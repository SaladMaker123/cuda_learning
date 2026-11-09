#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/std/utility>
#include "common/cuda_check.h"

constexpr int M = 1024;
constexpr int N = 256;
constexpr int TILE = 16;
constexpr int PHASES = 8;
constexpr int STAGES = 2;
constexpr int RUNS = 30;
// CPU、GPU 使用相同计算
__host__ __device__ int calculate(int value, int work)
{
    for (int i = 0; i < work; ++i)
    {
        value = (value * 3 + 7) % 1000003;
    }

    return value + 1;
}
// ============================================
// V10：Warp Specialization
// ============================================
__global__ void tmaWarpSpecializationV10(
    const __grid_constant__ CUtensorMap inputMap,
    int* output,
    int work
)
{
    __shared__ __align__(128) int sData[STAGES][TILE][TILE];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> ready[STAGES];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> filled[STAGES];
     // 96 个消费者线程 + 1 个生产者线程
    if(threadIdx.x == 0)
    {
        for(int s = 0;s < STAGES;++s)
        {
            init(&ready[s],97);
            init(&filled[s],97);
        }
        cuda::device::experimental::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    int startCol = blockIdx.x * TILE;
    int startRow = blockIdx.y * TILE * PHASES;
    int warpId = threadIdx.x / 32;
    // ========================================
    // Warp 0：Producer
    // ========================================
    if(warpId == 0)
    {
        // 由 Thread 0 发起 TMA
        if(threadIdx.x == 0)
        {
            for(int phase = 0;phase < PHASES;++phase)
            {
                int current = phase % STAGES;
                // 等待 Buffer 可以被重新写入
                ready[current].arrive_and_wait();
                 // 向当前 Buffer 发起 TMA 加载
                cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
                    sData[current],
                    &inputMap,
                    startCol,
                    startRow + phase * TILE,
                    filled[current]
                );
                // 登记事务并报告生产者到达
                (void)cuda::device::barrier_arrive_tx(
                    filled[current],
                    1,
                    sizeof(sData[current])
                );
            }
        }
    }
    // ========================================
    // Warp 1～3：Consumer
    // ========================================
    else
    {
        // 初始两个 Buffer 都可以写入
        ready[0].arrive();
        ready[1].arrive();
        for(int phase = 0;phase < PHASES;++phase)
        {
            int current = phase % STAGES;
            // 等待当前 Buffer 的 TMA 加载完成
            auto token = filled[current].arrive();
            filled[current].wait(cuda::std::move(token));
             // 96 个消费者线程负责 256 个元素
            for (int idx = threadIdx.x - 32;
                 idx < TILE * TILE;
                 idx += blockDim.x - 32)
            {
                int row = idx / TILE;
                int col = idx % TILE;

                int globalRow = startRow + phase * TILE + row;
                int globalCol = startCol + col;

                int value = sData[current][row][col];

                output[globalRow * N + globalCol] =
                    calculate(value, work);
            }
            // 当前 Buffer 已消费完成，可以复用
            if (phase + STAGES < PHASES)
            {
                ready[current].arrive();
            }
        }
    }
}
// ============================================
// Host：Tensor Map + Benchmark
// ============================================
int main()
{
    int hInput[M * N];
    int hOutput[M * N];

    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            hInput[row * N + col] = row % 100 + col;
        }
    }

    int* dInput = nullptr;
    int* dOutput = nullptr;

    CUDA_CHECK(cudaMalloc(&dInput, sizeof(hInput)));
    CUDA_CHECK(cudaMalloc(&dOutput, sizeof(hOutput)));

    CUDA_CHECK(cudaMemcpy(
        dInput, hInput, sizeof(hInput), cudaMemcpyHostToDevice
    ));

    // ========================================
    // 创建 Tensor Map
    // ========================================
    alignas(64) CUtensorMap inputMap{};

    const cuuint64_t globalDim[2] = {N, M};
    const cuuint64_t globalStrides[1] = {N * sizeof(int)};
    const cuuint32_t boxDim[2] = {TILE, TILE};
    const cuuint32_t elementStrides[2] = {1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &inputMap,
        CU_TENSOR_MAP_DATA_TYPE_INT32,
        2,
        dInput,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (result != CUDA_SUCCESS)
    {
        const char* errorName = nullptr;
        cuGetErrorName(result, &errorName);

        std::cerr << "Tensor Map creation failed: "
                  << (errorName ? errorName : "Unknown")
                  << std::endl;

        CUDA_CHECK(cudaFree(dInput));
        CUDA_CHECK(cudaFree(dOutput));

        return 1;
    }

    dim3 grid(N / TILE, M / (TILE * PHASES));

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int works[4] = {0, 32, 128, 256};

    for (int w = 0; w < 4; ++w)
    {
        int work = works[w];

        // 预热
        for (int i = 0; i < 5; ++i)
        {
            tmaWarpSpecializationV10<<<grid, 128>>>(
                inputMap, dOutput, work
            );
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark
        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < RUNS; ++i)
        {
            tmaWarpSpecializationV10<<<grid, 128>>>(
                inputMap, dOutput, work
            );
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &elapsed, start, stop
        ));

        float timeUs = elapsed * 1000.0f / RUNS;

        // 单独执行一次，检查正确性
        CUDA_CHECK(cudaMemset(
            dOutput, 0, sizeof(hOutput)
        ));

        tmaWarpSpecializationV10<<<grid, 128>>>(
            inputMap, dOutput, work
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(
            hOutput, dOutput, sizeof(hOutput),
            cudaMemcpyDeviceToHost
        ));

        int errors = 0;

        for (int i = 0; i < M * N; ++i)
        {
            int expected = calculate(hInput[i], work);

            if (hOutput[i] != expected)
            {
                ++errors;
            }
        }

        std::cout
            << "Work = " << work
            << "  V10: "
            << (errors == 0 ? "PASS" : "FAIL")
            << "  Time: " << timeUs << " us"
            << std::endl;

        if (errors != 0)
        {
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            CUDA_CHECK(cudaFree(dInput));
            CUDA_CHECK(cudaFree(dOutput));

            return 1;
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dOutput));

    return 0;
}



