#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include "common/cuda_check.h"

namespace wmma = nvcuda::wmma;
constexpr int M = 512;
constexpr int N = 512;
constexpr int K = 256;
constexpr int BK = 16;
// ============================================
// Kernel：WMMA Double Buffering
// ============================================
__global__ void wmmaDoubleBuffer(
    const __half *A,
    const __half *B,
    float* C
)
{
    // 两组 Shared Memory
    __shared__ __align__(32) __half sA[2][32 * BK];
    __shared__ __align__(32) __half sB[2][BK * 32];
    // 当前线程负责的 Shared Memory 起始下标
    // 每个线程复制 4 个 __half，即 8 字节
    int idx = threadIdx.x * 4;
    //warp编号和局部位置
    int warpIdx = threadIdx.x / 32;
    int warpRow = warpIdx / 2;
    int warpCol = warpIdx % 2;
    //warp负责的C矩阵块起始位置
    int row = blockIdx.y * 32 + warpRow * 16;
    int col = blockIdx.x * 32 + warpCol * 16;
    //wmma fragment
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bFrag;
    wmma::fragment<wmma::accumulator,16,16,16,float> cFrag;
    wmma::fill_fragment(cFrag,0.0f);
    // ========================================
    // 1. 预先加载 Phase 0 到 Buffer 0
    // ========================================
    __pipeline_memcpy_async(sA[0] + idx, A + (blockIdx.y * 32 + idx / BK) * K + idx % BK,8);
    __pipeline_memcpy_async(sB[0] + idx, B + (blockIdx.x * 32 + idx / BK) * K + idx % BK,8);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    // ========================================
    // 2. K Phase 主循环
    // ========================================
    for(int phase = 0;phase < K / BK; phase++)
    {
        int current = phase % 2;
        int next = (phase + 1) % 2;
        //提前搬运下一阶段的数据
        if(phase + 1 < K / BK)
        {
            int nextk = (phase + 1) * BK;
            //Global A -> 下一 Buffer
            __pipeline_memcpy_async(sA[next] + idx,A + (blockIdx.y * 32 + idx / BK) * K + nextk + idx % BK,8);
            //Global B -> 下一 Buffer
            __pipeline_memcpy_async(sB[next] + idx,
                
                B + (blockIdx.x * 32 + idx / BK) * K + nextk + idx % BK,8);
            __pipeline_commit();
        }
        //从当前buffer加载A fragment
        wmma::load_matrix_sync(aFrag,sA[current] + warpRow * 16 * BK,16);
        // 从当前 Buffer 加载 B Fragment
        wmma::load_matrix_sync(bFrag,sB[current] + warpCol * 16 * BK,16);
        // 执行 WMMA 乘法累加
        wmma::mma_sync(cFrag,aFrag,bFrag,cFrag);
        // 等待下一阶段数据搬运完成
        if(phase + 1 < K / BK)
        {
            __pipeline_wait_prior(0);
            __syncthreads();
        }
    }
    // ========================================
    // 3. 写回最终 C
    // ========================================
    wmma::store_matrix_sync(C + row * N + col,cFrag,N,wmma::mem_row_major);
}
// ============================================
// Host：初始化、运行、验证
// ============================================
int main()
{
    __half hA[M * K];
    __half hB[K * N];
    float hC[M * N];

    // 初始化 A：Row-Major
    for (int row = 0; row < M; ++row)
    {
        for (int k = 0; k < K; ++k)
        {
            hA[row * K + k] =
                __half((row + k) % 5 - 2);
        }
    }

    // 初始化 B：Column-Major
    for (int col = 0; col < N; ++col)
    {
        for (int k = 0; k < K; ++k)
        {
            hB[col * K + k] =
                __half((k + 2 * col) % 5 - 2);
        }
    }

    // GPU 指针
    __half* dA;
    __half* dB;
    float* dC;

    // 分配 GPU 内存
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(hC)));

    // CPU -> GPU
    CUDA_CHECK(cudaMemcpy(
        dA, hA, sizeof(hA),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dB, hB, sizeof(hB),
        cudaMemcpyHostToDevice
    ));

    // 16 个 Block，每个 Block 128 个线程
    dim3 grid(N / 32, M / 32);

    
    // ========================================
    // Benchmark：预热 + CUDA Event 计时
    // ========================================

    constexpr int WARMUP = 20;
    constexpr int REPEAT = 1000;

    // 1. Warmup：预热
    for (int i = 0; i < WARMUP; ++i)
    {
        wmmaDoubleBuffer<<<grid, 128>>>(dA, dB, dC);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2. 创建 CUDA Events
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 3. 正式计时
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEAT; ++i)
    {
        wmmaDoubleBuffer<<<grid, 128>>>(dA, dB, dC);
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // 4. 计算平均耗时
    float totalMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &totalMs,
        start,
        stop
    ));

    float avgMs = totalMs / REPEAT;

    std::cout << "Average Kernel Time: "
              << avgMs << " ms" << std::endl;

    // 5. 释放 CUDA Events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));


    // GPU -> CPU
    CUDA_CHECK(cudaMemcpy(
        hC, dC, sizeof(hC),
        cudaMemcpyDeviceToHost
    ));

    // CPU 参考计算和正确性验证
    int errors = 0;
    float maxDiff = 0.0f;

    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float expected = 0.0f;

            for (int k = 0; k < K; ++k)
            {
                expected +=
                    float(hA[row * K + k]) *
                    float(hB[col * K + k]);
            }

            float actual = hC[row * N + col];
            float diff = std::fabs(actual - expected);

            if (diff > maxDiff)
            {
                maxDiff = diff;
            }

            if (diff > 1e-3f)
            {
                ++errors;
            }
        }
    }

    std::cout << "Max difference: "
              << maxDiff << std::endl;

    if (errors == 0)
    {
        std::cout << "WMMA Double Buffer: PASS"
                  << std::endl;
    }
    else
    {
        std::cout << "WMMA Double Buffer: FAIL ("
                  << errors << " errors)"
                  << std::endl;
    }

    // 释放 GPU 内存
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return errors == 0 ? 0 : 1;
}