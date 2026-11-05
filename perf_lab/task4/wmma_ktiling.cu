#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "common/cuda_check.h"

namespace wmma = nvcuda::wmma;

constexpr int M = 512;
constexpr int N = 512;
constexpr int K = 256;
constexpr int BK = 16;

// ============================================
// Kernel：WMMA K Tiling
// ============================================
__global__ void wmmaKTiling(
    const __half* A,
    const __half* B,
    float* C
)
{
    //每个block只保存当前的K Tile
    __shared__ __align__(32) __half sA[32 * BK];
    __shared__ __align__(32) __half sB[BK * 32];
    //warp编号及位置
    int warpIdx = threadIdx.x / 32;
    int warpRow = warpIdx / 2;
    int warpCol = warpIdx % 2;
    //当前warp在完整C中的起点
    int row = blockIdx.y * 32 + warpRow * 16;
    int col = blockIdx.x * 32 + warpCol * 16;
    //声明Fragment
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator,16,16,16,float> c_frag;
    wmma::fill_fragment(c_frag,0.0f);
    //K方向分阶段搬运与计算
    for(int phase = 0;phase < K / BK; phase++)
    {
        int k0 = phase * BK;
        //globalA -> shared A
        for(int idx = threadIdx.x;idx < 32 * BK;idx += blockDim.x)
        {
            sA[idx] = A[(blockIdx.y * 32 + idx / BK) * K + k0 + idx % BK];
        }
        //globalB -> shared B
        for(int idx = threadIdx.x;idx < BK * 32;idx += blockDim.x)
        {
            sB[idx] = B[(blockIdx.x * 32 + idx / BK) * K + k0 + idx % BK];
        }
        // 等待当前阶段数据搬运完成
        __syncthreads();
        // Shared A -> A Fragment
        wmma::load_matrix_sync(a_frag,sA + warpRow * 16 * BK,BK);
        // Shared B -> B Fragment
        wmma::load_matrix_sync(b_frag,sB + warpCol * 16 * BK,BK);
        // Tensor Core 矩阵乘加
        wmma::mma_sync(
            c_frag,
            a_frag,
            b_frag,
            c_frag
        );

        // 所有 Warp 用完当前 Shared Memory
        __syncthreads();

    }
     // 将累加结果写回 Global Memory
    wmma::store_matrix_sync(
        C + row * N + col,
        c_frag,
        N,
        wmma::mem_row_major
    );
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

    // Grid：4 x 4 = 16 Blocks
    // 每个 Block：128 Threads = 4 Warps
    dim3 grid(N / 32, M / 32);

    
    // ========================================
    // Benchmark：Warmup + CUDA Event
    // ========================================

    constexpr int WARMUP = 20;
    constexpr int REPEAT = 1000;

    // 1. 预热
    for (int i = 0; i < WARMUP; ++i)
    {
        wmmaKTiling<<<grid, 128>>>(dA, dB, dC);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2. 创建 CUDA Events
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 3. 正式测量
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEAT; ++i)
    {
        wmmaKTiling<<<grid, 128>>>(dA, dB, dC);
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // 4. 计算平均时间
    float totalMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &totalMs,
        start,
        stop
    ));

    float avgMs = totalMs / REPEAT;

    std::cout << "Average Kernel Time: "
              << avgMs << " ms" << std::endl;

    // 5. 释放 Events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));


    // GPU -> CPU
    CUDA_CHECK(cudaMemcpy(
        hC, dC, sizeof(hC),
        cudaMemcpyDeviceToHost
    ));

    // CPU 参考计算
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
        std::cout << "WMMA K Tiling: PASS"
                  << std::endl;
    }
    else
    {
        std::cout << "WMMA K Tiling: FAIL ("
                  << errors << " errors)"
                  << std::endl;
    }

    // 释放 GPU 内存
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return errors == 0 ? 0 : 1;
}