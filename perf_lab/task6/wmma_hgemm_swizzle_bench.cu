#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "common/cuda_check.h"

namespace wmma = nvcuda::wmma;
constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 64;
constexpr int TILE = 128;
constexpr int SWIZZLE_COLS = 2;
constexpr int SWIZZLE_ROWS = 4;
// ============================================
// Kernel：WMMA HGEMM + Block Swizzle
// ============================================
__global__ void wmmaHGEMMSwizzle(
    const __half* A,
    const __half* B,
    float* C
)
{
    // 1. 计算逻辑 Tile 数量
    int tilesM = M / TILE;
    int tilesN = N / TILE;
    // 2. 计算每行有多少个 Swizzle Group
    int groupsPerRow = (tilesN + SWIZZLE_COLS -1) / SWIZZLE_COLS;
    // 3. 当前 CUDA Block 的编号
    int groupId = blockIdx.y;
    int localId = blockIdx.x;
    // 4. Group 在整个矩阵中的位置
    int groupCol = groupId % groupsPerRow;
    int groupRow = groupId / groupsPerRow;
    //5. 转换成真正的逻辑 Block 坐标
    int blockCol = groupCol * SWIZZLE_COLS + localId % SWIZZLE_COLS;
    int blockRow = groupRow * SWIZZLE_ROWS + localId / SWIZZLE_COLS;
    // 6. 边界检查
    if(blockRow >= tilesM || blockCol >= tilesN)
    {
        return;
    }
    // ========================================
    // 7. Shared Memory
    // ========================================
    __shared__ __align__(32) __half sA[TILE * K];
    __shared__ __align__(32) __half sB[K * TILE];
     // Global A -> Shared A
    for(int idx = threadIdx.x;idx < TILE * K;idx += blockDim.x)
    {
        sA[idx] = A[blockRow * TILE * K + idx];
    }
    // Global B -> Shared B
    for(int idx = threadIdx.x;idx < TILE * K;idx += blockDim.x)
    {
        sB[idx] = B[blockCol * TILE * K + idx];
    }
    __syncthreads();
    // ========================================
    // 8. Warp 计算分工
    // ========================================
    int warpIdx = threadIdx.x / 32;

    // 每个 Warp 计算 2×8 个 WMMA Tile
    for(int tileRow = 0;tileRow < 2;++tileRow)
    {
        for(int tileCol = 0;tileCol < 8;++tileCol)
        {
            //当前 Warp 的局部 Tile 起点
            int localRow = warpIdx * 32 + tileRow * 16;
            int localCol = tileCol * 16;
            // 当前 Tile 在完整 C 中的起点
            int row = blockRow * TILE + localRow;
            int col = blockCol * TILE + localCol;
            // WMMA Fragment
            wmma::fragment<
                wmma::matrix_a,16,16,16,
                __half,wmma::row_major
            > aFrag;

            wmma::fragment<
                wmma::matrix_b,16,16,16,
                __half,wmma::col_major
            > bFrag;

            wmma::fragment<
                wmma::accumulator,16,16,16,float
            > cFrag;

            wmma::fill_fragment(cFrag,0.0f);
            // ====================================
            // 9. K Phase
            // ====================================
            for (int phase = 0; phase < K / 16; ++phase)
            {
                int k0 = phase * 16;

                // Shared A -> Fragment
                wmma::load_matrix_sync(
                    aFrag,
                    sA + localRow * K + k0,
                    K
                );

                // Shared B -> Fragment
                wmma::load_matrix_sync(
                    bFrag,
                    sB + localCol * K + k0,
                    K
                );

                // Tensor Core MMA
                wmma::mma_sync(
                    cFrag,
                    aFrag,
                    bFrag,
                    cFrag);
            }
             // 10. 当前输出 Tile 写回 C
            wmma::store_matrix_sync(
                C + row * N + col,
                cFrag,
                N,
                wmma::mem_row_major);

        }
    }
}
// ============================================
// Host：初始化、启动、正确性验证
// ============================================
int main()
{
    __half* hA = new __half[M * K];
    __half* hB = new __half[K * N];
    float* hC = new float[M * N];
    // A：Row-Major
    for (int row = 0; row < M; ++row)
    {
        for (int k = 0; k < K; ++k)
        {
            hA[row * K + k] =
                __half((row + k) % 5 - 2);
        }
    }

    // B：Column-Major
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

    CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));

    // CPU -> GPU
    CUDA_CHECK(cudaMemcpy(
        dA, hA, M * K * sizeof(__half),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dB, hB, K * N * sizeof(__half),
        cudaMemcpyHostToDevice
    ));
    // ========================================
    // Grid：按 Swizzle Group 组织
    // ========================================
    int tilesM = M / TILE;
    int tilesN = N / TILE;
    int groupsPerRow =
        (tilesN + SWIZZLE_COLS - 1) / SWIZZLE_COLS;

    int groupsPerCol =
        (tilesM + SWIZZLE_ROWS - 1) / SWIZZLE_ROWS;
    int groupCount = groupsPerRow * groupsPerCol;
    dim3 grid(
        SWIZZLE_COLS * SWIZZLE_ROWS,
        groupCount
    );
    // 启动 Kernel
    constexpr int WARMUP = 20;
    constexpr int REPEAT = 1000;

    // 预热
    for (int i = 0; i < WARMUP; ++i)
    {
        wmmaHGEMMSwizzle<<<grid,128>>>(dA,dB,dC);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // CUDA Event
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 正式计时
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEAT; ++i)
    {
        wmmaHGEMMSwizzle<<<grid,128>>>(dA,dB,dC);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // 计算平均耗时
    float totalMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &totalMs, start, stop
    ));

    float avgMs = totalMs / REPEAT;

    std::cout << "Average Kernel Time: "
            << avgMs << " ms" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
     // GPU -> CPU
    CUDA_CHECK(cudaMemcpy(
        hC, dC, M * N * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // ========================================
    // CPU 参考计算
    // ========================================

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
        std::cout << "WMMA HGEMM Swizzle V1: PASS"
                  << std::endl;
    }
    else
    {
        std::cout << "WMMA HGEMM Swizzle V1: FAIL ("
                  << errors << " errors)"
                  << std::endl;
    }

    // 释放 GPU 内存
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    delete[] hA;
    delete[] hB;
    delete[] hC;

    return errors == 0 ? 0 : 1;
}