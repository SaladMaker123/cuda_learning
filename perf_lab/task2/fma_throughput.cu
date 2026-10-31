#include <cuda_runtime.h>
#include <cstdio>

#include "common/cuda_check.h"

// 原始 Warmup 实验参数
constexpr int BLOCK_SIZE = 256;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int ILP = 4;
constexpr int N_ITER = 40000;
// ============================================================
// Kernel：执行大量独立的 FP32 FMA 运算
// ============================================================
__global__ void peak_flops_kernel(
    float* __restrict__ out,
    int n,
    bool flag
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //每个线程的累加器
    
    float c[TM * TN];
    //初始化64个累加器
    #pragma unroll 
    for(int tm = 0 ;tm < TM ; tm++)
    {
        #pragma unroll
        for(int tn = 0 ; tn < TN ; tn++)
        {
            c[tm * TN + tn] = static_cast<float>(tm * TN + tn);
        }
    }
    //大量重复FMA
    for(int i = 0; i< n;++i)
    {
        #pragma unroll
        for(int ilp = 0; ilp < ILP; ++ilp)
        {
            #pragma unroll
            for(int tm = 0; tm < TM;++tm)
            {
                #pragma unroll
                for(int tn = 0; tn < TN; tn++)
                {
                    c[tm * TN + tn] = fma(3.f,7.f,c[tm * TN + tn]);
                }
            }
        }
    }
    //写回结果，避免计算被优化掉
    if(flag)
    {
        float sum = 0.0f;
        for(int tm = 0;tm < TM; tm++)
        {
            for(int tn = 0; tn < TN; tn++)
            {
                sum += c[tm * TN + tn];
            }
        }
        out[idx] = sum;
    }
}

// ============================================================
// Main
// ============================================================
int main()
{
    //选择GPU
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    int sm_count = prop.multiProcessorCount;
    printf("GPU: %s\n",prop.name);
    printf("SM Count: %d\n",sm_count);
    // 2. 配置 Grid 和 Block
    const int n_blocks = sm_count * 28;
    const int total_threads = n_blocks * BLOCK_SIZE;
    // 3. 分配 GPU 显存
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(
        &d_out,
        static_cast<size_t>(total_threads) * sizeof(float)
    ));
    // 4. 创建 CUDA Event
    cudaEvent_t ev_start;
    cudaEvent_t ev_stop;

    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
     // 5. Warmup
    peak_flops_kernel<<<n_blocks, BLOCK_SIZE>>>(
        d_out,
        N_ITER,
        true
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

     // 6. 正式计时
    CUDA_CHECK(cudaEventRecord(ev_start));

    peak_flops_kernel<<<n_blocks, BLOCK_SIZE>>>(
        d_out,
        N_ITER,
        true
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    // 7. 获取执行时间
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(
        &ms,
        ev_start,
        ev_stop
    ));
    // 8. 计算总浮点运算次数
    double total_flops =
        static_cast<double>(total_threads) *
        N_ITER *
        TM *
        TN *
        ILP *
        2.0;
    // 9. 计算 TFLOPS
    double tflops =
        total_flops /
        (static_cast<double>(ms) * 1e-3) /
        1e12;
    // 10. 输出结果
    printf("\n=== FMA Throughput ===\n\n");

    printf("Blocks:    %d\n", n_blocks);
    printf("Threads:   %d\n", total_threads);
    printf("Time:      %.3f ms\n", ms);
    printf("Peak FP32: %.2f TFLOPS\n", tflops);

    // 11. 销毁 CUDA Event
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    // 12. 释放显存
    CUDA_CHECK(cudaFree(d_out));

    return 0;

}