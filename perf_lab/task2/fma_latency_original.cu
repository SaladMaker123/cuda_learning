
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "common/cuda_check.h"

// 实验配置
constexpr int N_BLOCKS = 1;
constexpr int N_THREADS = 1;
constexpr int N_ITER = 200000;
constexpr int MAX_ILP = 8;


// ============================================================
// Kernel 1: Warmup
// 从初始状态开始执行 FMA，并保存每条依赖链的最终状态
// ============================================================

template <int D, int I>
__global__ void kernel_warmup(
    const float* __restrict__ c_init,
    float* __restrict__ c_state,
    int n_iter)
{
    float c[I];

    #pragma unroll
    for (int i = 0; i < I; ++i)
    {
        c[i] = c_init[i];
    }

    for (int iter = 0; iter < n_iter; ++iter)
    {
        #pragma unroll
        for (int d = 0; d < D; ++d)
        {
            #pragma unroll
            for (int i = 0; i < I; ++i)
            {
                c[i] = fma(3.f, 7.f, c[i]);
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < I; ++i)
    {
        c_state[i] = c[i];
    }
}


// ============================================================
// Kernel 2: Timed
// 从预热后的状态继续计算，并将 I 条链的结果求和
// ============================================================

template <int D, int I>
__global__ void kernel_timed(
    const float* __restrict__ c_state,
    float* __restrict__ sink_out,
    int n_iter)
{
    float c[I];

    #pragma unroll
    for (int i = 0; i < I; ++i)
    {
        c[i] = c_state[i];
    }

    for (int iter = 0; iter < n_iter; ++iter)
    {
        #pragma unroll
        for (int d = 0; d < D; ++d)
        {
            #pragma unroll
            for (int i = 0; i < I; ++i)
            {
                c[i] = fma(3.f, 7.f, c[i]);
            }
        }
    }

    float sum = 0.f;

    #pragma unroll
    for (int i = 0; i < I; ++i)
    {
        sum += c[i];
    }

    sink_out[threadIdx.x] = sum;
}


// ============================================================
// 测量一组 D 和 I
// ============================================================

template <int D, int I>
bool run_case(
    float* h_init,
    float* d_init,
    float* d_c_state,
    float* h_sink,
    float* d_sink,
    int n_iter,
    int n_blocks,
    int n_threads,
    cudaEvent_t ev_start,
    cudaEvent_t ev_stop,
    int clock_rate_khz)
{
    // 1. 初始化输入
    for (int i = 0; i < I; ++i)
    {
        h_init[i] = static_cast<float>(i + 1);
    }

    CUDA_CHECK(cudaMemcpy(
        d_init,
        h_init,
        I * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    // 2. Warmup Kernel
    kernel_warmup<D, I><<<n_blocks, n_threads>>>(
        d_init,
        d_c_state,
        n_iter
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3. 开始计时
    CUDA_CHECK(cudaEventRecord(ev_start));

    // 4. Timed Kernel
    kernel_timed<D, I><<<n_blocks, n_threads>>>(
        d_c_state,
        d_sink,
        n_iter
    );

    CUDA_CHECK(cudaGetLastError());

    // 5. 结束计时
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms = 0.f;

    CUDA_CHECK(cudaEventElapsedTime(
        &ms,
        ev_start,
        ev_stop
    ));

    // ms * kHz = cycles
    double cyc =
        static_cast<double>(ms) *
        clock_rate_khz /
        n_iter;

    // 6. 读取结果
    CUDA_CHECK(cudaMemcpy(
        h_sink,
        d_sink,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // 7. 输出本组实验结果
    printf("%6d %6d %14.1f\n", D, I, cyc);

    return true;
}


// ============================================================
// Experiment 1
// 固定 D，依次测量 ILP = 1 ~ 8
// ============================================================

template <int D>
bool run_ilp_sweep(
    float* h_init,
    float* d_init,
    float* d_c_state,
    float* h_sink,
    float* d_sink,
    int n_iter,
    int n_blocks,
    int n_threads,
    cudaEvent_t ev_start,
    cudaEvent_t ev_stop,
    int clock_rate_khz)
{
    bool ok = true;

    #define RUN_CASE(I) \
        ok = run_case<D, I>( \
            h_init, d_init, d_c_state, h_sink, d_sink, \
            n_iter, n_blocks, n_threads, \
            ev_start, ev_stop, clock_rate_khz \
        ) && ok

    RUN_CASE(1);
    RUN_CASE(2);
    RUN_CASE(3);
    RUN_CASE(4);
    RUN_CASE(5);
    RUN_CASE(6);
    RUN_CASE(7);
    RUN_CASE(8);

    #undef RUN_CASE

    return ok;
}


// ============================================================
// Experiment 2
// 空循环基线：D = 0，ILP = 1
// ============================================================

bool run_empty_baseline(
    float* h_init,
    float* d_init,
    float* d_c_state,
    float* h_sink,
    float* d_sink,
    int n_blocks,
    int n_threads,
    cudaEvent_t ev_start,
    cudaEvent_t ev_stop,
    int clock_rate_khz)
{
    printf("\n=== Experiment 2: Empty Loop Baseline ===\n\n");

    printf("%6s %6s %14s\n",
           "D", "ILP", "Cycles/iter");

    return run_case<0, 1>(
        h_init,
        d_init,
        d_c_state,
        h_sink,
        d_sink,
        N_ITER,
        n_blocks,
        n_threads,
        ev_start,
        ev_stop,
        clock_rate_khz
    );
}


// ============================================================
// Experiment 3
// 固定 D = 4、ILP = 1，改变迭代次数
// ============================================================

bool run_niter_sensitivity(
    float* h_init,
    float* d_init,
    float* d_c_state,
    float* d_sink,
    int n_blocks,
    int n_threads,
    cudaEvent_t ev_start,
    cudaEvent_t ev_stop,
    int clock_rate_khz)
{
    printf("\n=== Experiment 3: Iteration Sensitivity ===\n\n");

    printf("%10s %14s\n",
           "n_iter", "Cycles/iter");

    const int iters[] = {
        1000,
        5000,
        20000,
        50000,
        100000,
        200000,
        500000
    };

    for (int ni : iters)
    {
        h_init[0] = 1.f;

        CUDA_CHECK(cudaMemcpy(
            d_init,
            h_init,
            sizeof(float),
            cudaMemcpyHostToDevice
        ));

        kernel_warmup<4, 1><<<n_blocks, n_threads>>>(
            d_init,
            d_c_state,
            ni
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(ev_start));

        kernel_timed<4, 1><<<n_blocks, n_threads>>>(
            d_c_state,
            d_sink,
            ni
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms = 0.f;

        CUDA_CHECK(cudaEventElapsedTime(
            &ms,
            ev_start,
            ev_stop
        ));

        double cyc =
            static_cast<double>(ms) *
            clock_rate_khz /
            ni;

        printf("%10d %14.1f\n", ni, cyc);
    }

    return true;
}


// ============================================================
// Main
// ============================================================

int main()
{
    // 1. 选择 GPU
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};

    CUDA_CHECK(cudaGetDeviceProperties(
        &prop,
        0
    ));

    printf("GPU: %s\n", prop.name);
    printf("Reference Clock: %d kHz\n", prop.clockRate);

    // 2. 分配 CPU 内存
    float* h_init = static_cast<float*>(
        std::malloc(MAX_ILP * sizeof(float))
    );

    float* h_sink = static_cast<float*>(
        std::malloc(sizeof(float))
    );

    if (h_init == nullptr || h_sink == nullptr)
    {
        fprintf(stderr, "Host allocation failed\n");

        std::free(h_init);
        std::free(h_sink);

        return EXIT_FAILURE;
    }

    // 3. 分配 GPU 显存
    float* d_init = nullptr;
    float* d_c_state = nullptr;
    float* d_sink = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_init,
        MAX_ILP * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_c_state,
        MAX_ILP * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_sink,
        sizeof(float)
    ));

    // 4. 创建 CUDA Event
    cudaEvent_t ev_start;
    cudaEvent_t ev_stop;

    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // 5. Experiment 1
    printf("\n=== Experiment 1: D x ILP Sweep ===\n\n");

    printf("%6s %6s %14s\n",
           "D", "ILP", "Cycles/iter");

    #define RUN_SWEEP(D) \
        run_ilp_sweep<D>( \
            h_init, d_init, d_c_state, h_sink, d_sink, \
            N_ITER, N_BLOCKS, N_THREADS, \
            ev_start, ev_stop, prop.clockRate \
        )

    bool ok = true;

    ok = RUN_SWEEP(2) && ok;
    ok = RUN_SWEEP(4) && ok;
    ok = RUN_SWEEP(6) && ok;
    ok = RUN_SWEEP(8) && ok;
    ok = RUN_SWEEP(10) && ok;

    ok = RUN_SWEEP(1) && ok;
    ok = RUN_SWEEP(12) && ok;
    ok = RUN_SWEEP(16) && ok;
    ok = RUN_SWEEP(20) && ok;

    #undef RUN_SWEEP

    // 6. Experiment 2
    bool baseline_ok = run_empty_baseline(
        h_init,
        d_init,
        d_c_state,
        h_sink,
        d_sink,
        N_BLOCKS,
        N_THREADS,
        ev_start,
        ev_stop,
        prop.clockRate
    );

    ok = baseline_ok && ok;

    // 7. Experiment 3
    bool sensitivity_ok = run_niter_sensitivity(
        h_init,
        d_init,
        d_c_state,
        d_sink,
        N_BLOCKS,
        N_THREADS,
        ev_start,
        ev_stop,
        prop.clockRate
    );

    ok = sensitivity_ok && ok;

    // 8. 销毁 CUDA Event
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    // 9. 释放 GPU 显存
    CUDA_CHECK(cudaFree(d_init));
    CUDA_CHECK(cudaFree(d_c_state));
    CUDA_CHECK(cudaFree(d_sink));

    // 10. 释放 CPU 内存
    std::free(h_init);
    std::free(h_sink);

    printf("\nOverall: %s\n",
           ok ? "COMPLETED" : "FAIL");

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
