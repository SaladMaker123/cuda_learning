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
constexpr int BK = 16;
constexpr int HALF_TILE = TILE / 2;

constexpr int SWIZZLE_COLS = 2;
constexpr int SWIZZLE_ROWS = 4;
constexpr int WARMUP = 20;
constexpr int REPEAT = 1000;
// ==================================================
// 1. PTX：异步复制 8 字节
// ==================================================
__device__ __forceinline__
void cpAsync8(__half* dst, const __half* src)
{
    unsigned int sharedAddr =
        static_cast<unsigned int>(
            __cvta_generic_to_shared(dst)
        );

    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 8;\n"
        :
        : "r"(sharedAddr), "l"(src)
        : "memory"
    );
}
// ==================================================
// 2. PTX：提交当前线程的异步复制批次
// ==================================================
__device__ __forceinline__
void cpAsyncCommit()
{
    asm volatile(
        "cp.async.commit_group;\n"
        ::: "memory"
    );
}
// ==================================================
// 3. PTX：等待当前线程已提交的复制完成
// ==================================================
__device__ __forceinline__
void cpAsyncWait()
{
    asm volatile(
        "cp.async.wait_group 0;\n"
        ::: "memory"
    );
}
// ==================================================
// 4. Kernel：WMMA + Swizzle + Warp Partition
// ==================================================
__global__ void wmmaHGEMMWarpPartitionV4(
    const __half* A,
    const __half* B,
    float* C
)
{
    // ----------------------------------------------
    // 4.1 Block Swizzle：沿用 V3
    // ----------------------------------------------
    int tilesM = M / TILE;
    int tilesN = N / TILE;
    int groupsPerRow =
        (tilesN + SWIZZLE_COLS - 1) / SWIZZLE_COLS;
    int groupId = blockIdx.y;
    int localId = blockIdx.x;
     
    int groupRow = groupId / groupsPerRow;
    int groupCol = groupId % groupsPerRow;
    int blockRow =
        groupRow * SWIZZLE_ROWS
        + localId / SWIZZLE_COLS;

    int blockCol =
        groupCol * SWIZZLE_COLS
        + localId % SWIZZLE_COLS;

    if (blockRow >= tilesM || blockCol >= tilesN)
    {
        return;
    }
    // ----------------------------------------------
    // 4.2 V4：Warp 分工
    // ----------------------------------------------
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    // Warp 0、1 负责前半部分
    // Warp 2、3 负责后半部分
    int part = warpId / 2;
    // Warp 0、2 加载 A
    // Warp 1、3 加载 B
    bool loadA = (warpId % 2 == 0);
    // ----------------------------------------------
    // 4.3 Shared Memory 双缓冲
    // ----------------------------------------------
    __shared__ __align__(32)
    __half sA[2][TILE * BK];

    __shared__ __align__(32)
    __half sB[2][BK * TILE];
    // ----------------------------------------------
    // 4.4 每个 Warp 的 16 个累加器
    // ----------------------------------------------

    wmma::fragment<
        wmma::accumulator,
        16, 16, 16,
        float
    > cFrag[16];

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        wmma::fill_fragment(cFrag[i], 0.0f);
    }
    // ----------------------------------------------
    // 4.5 Phase 0：Warp 分工加载到 Buffer 0
    // ----------------------------------------------
    for(int i = laneId * 4;i < HALF_TILE * BK;i += 32 * 4)
    {
        // 将当前 Warp 的局部索引换算到完整 Tile
        int idx = part * HALF_TILE * BK + i;
        if(loadA)
        {
            // Warp 0、2：Global A -> Shared A
            int globalARow =
                blockRow * TILE + idx / BK;

            int globalACol = idx % BK;

            cpAsync8(
                sA[0] + idx,
                A + globalARow * K + globalACol
            );
        }
        else
        {
            // Warp 1、3：Global B -> Shared B
            // B 在 Global Memory 中按列连续存储

            int globalBRow = idx % BK;

            int globalBCol =
                blockCol * TILE + idx / BK;

            cpAsync8(
                sB[0] + idx,
                B + globalBCol * K + globalBRow
            );
        }
    }

    cpAsyncCommit();
    cpAsyncWait();

    // 每个线程完成自己的异步复制后，
    // 等待其他 Warp 的数据也准备完成
    __syncthreads();

    // ----------------------------------------------
    // 4.6 K Phase 双缓冲主循环
    // ----------------------------------------------

    for (int phase = 0; phase < K / BK; ++phase)
    {
        int current = phase % 2;
        int next = (phase + 1) % 2;

        // ------------------------------------------
        // 4.6.1 预加载下一 Phase
        // ------------------------------------------

        if (phase + 1 < K / BK)
        {
            int nextK = (phase + 1) * BK;

            for (int i = laneId * 4;
                 i < HALF_TILE * BK;
                 i += 32 * 4)
            {
                int idx =
                    part * HALF_TILE * BK + i;

                if (loadA)
                {
                    int globalARow =
                        blockRow * TILE + idx / BK;

                    int globalACol =
                        nextK + idx % BK;

                    cpAsync8(
                        sA[next] + idx,
                        A + globalARow * K + globalACol
                    );
                }
                else
                {
                    int globalBRow =
                        nextK + idx % BK;

                    int globalBCol =
                        blockCol * TILE + idx / BK;

                    cpAsync8(
                        sB[next] + idx,
                        B + globalBCol * K + globalBRow
                    );
                }
            }

            cpAsyncCommit();
        }

        // ------------------------------------------
        // 4.6.2 当前 Phase：WMMA 计算
        // 保持与 V3 完全相同的计算组织
        // ------------------------------------------

        #pragma unroll
        for (int tileRow = 0; tileRow < 2; ++tileRow)
        {
            #pragma unroll
            for (int tileCol = 0; tileCol < 8; ++tileCol)
            {
                int localRow =
                    warpId * 32 + tileRow * 16;

                int localCol =
                    tileCol * 16;

                int fragIdx =
                    tileRow * 8 + tileCol;

                wmma::fragment<
                    wmma::matrix_a,
                    16, 16, 16,
                    __half,
                    wmma::row_major
                > aFrag;

                wmma::fragment<
                    wmma::matrix_b,
                    16, 16, 16,
                    __half,
                    wmma::col_major
                > bFrag;

                // Shared A -> Fragment
                wmma::load_matrix_sync(
                    aFrag,
                    sA[current] + localRow * BK,
                    BK
                );

                // Shared B -> Fragment
                wmma::load_matrix_sync(
                    bFrag,
                    sB[current] + localCol * BK,
                    BK
                );

                // Tensor Core 矩阵乘加
                wmma::mma_sync(
                    cFrag[fragIdx],
                    aFrag,
                    bFrag,
                    cFrag[fragIdx]
                );
            }
        }

        // ------------------------------------------
        // 4.6.3 等待下一 Phase 的加载完成
        // ------------------------------------------

        if (phase + 1 < K / BK)
        {
            cpAsyncWait();
            __syncthreads();
        }
    }

    // ----------------------------------------------
    // 4.7 将 C 写回 Global Memory
    // ----------------------------------------------

    #pragma unroll
    for (int tileRow = 0; tileRow < 2; ++tileRow)
    {
        #pragma unroll
        for (int tileCol = 0; tileCol < 8; ++tileCol)
        {
            int localRow =
                warpId * 32 + tileRow * 16;

            int localCol =
                tileCol * 16;

            int row =
                blockRow * TILE + localRow;

            int col =
                blockCol * TILE + localCol;

            int fragIdx =
                tileRow * 8 + tileCol;

            wmma::store_matrix_sync(
                C + row * N + col,
                cFrag[fragIdx],
                N,
                wmma::mem_row_major
            );
        }
    }
}

// ==================================================
// 5. Host：初始化、Benchmark、正确性验证
// ==================================================

int main()
{
    // ----------------------------------------------
    // 5.1 Host 内存
    // ----------------------------------------------

    __half* hA = new __half[M * K];
    __half* hB = new __half[K * N];
    float* hC = new float[M * N];

    // ----------------------------------------------
    // 5.2 初始化 A（Row-Major）
    // ----------------------------------------------

    for (int row = 0; row < M; ++row)
    {
        for (int k = 0; k < K; ++k)
        {
            hA[row * K + k] =
                __half((row + k) % 5 - 2);
        }
    }

    // ----------------------------------------------
    // 5.3 初始化 B（Column-Major）
    // ----------------------------------------------

    for (int col = 0; col < N; ++col)
    {
        for (int k = 0; k < K; ++k)
        {
            hB[col * K + k] =
                __half((k + 2 * col) % 5 - 2);
        }
    }

    // ----------------------------------------------
    // 5.4 Device 内存
    // ----------------------------------------------

    __half* dA = nullptr;
    __half* dB = nullptr;
    float* dC = nullptr;

    CUDA_CHECK(cudaMalloc(
        &dA, M * K * sizeof(__half)
    ));

    CUDA_CHECK(cudaMalloc(
        &dB, K * N * sizeof(__half)
    ));

    CUDA_CHECK(cudaMalloc(
        &dC, M * N * sizeof(float)
    ));

    // ----------------------------------------------
    // 5.5 Host -> Device
    // ----------------------------------------------

    CUDA_CHECK(cudaMemcpy(
        dA,
        hA,
        M * K * sizeof(__half),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dB,
        hB,
        K * N * sizeof(__half),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemset(
        dC,
        0,
        M * N * sizeof(float)
    ));

    // ----------------------------------------------
    // 5.6 Grid 配置：保持 V3 的 Swizzle
    // ----------------------------------------------

    int tilesM = M / TILE;
    int tilesN = N / TILE;

    int groupsPerRow =
        (tilesN + SWIZZLE_COLS - 1) / SWIZZLE_COLS;

    int groupsPerCol =
        (tilesM + SWIZZLE_ROWS - 1) / SWIZZLE_ROWS;

    int groupCount =
        groupsPerRow * groupsPerCol;

    dim3 grid(
        SWIZZLE_COLS * SWIZZLE_ROWS,
        groupCount
    );

    // ----------------------------------------------
    // 5.7 Warmup：预热
    // ----------------------------------------------

    for (int i = 0; i < WARMUP; ++i)
    {
        wmmaHGEMMWarpPartitionV4<<<grid, 128>>>(
            dA, dB, dC
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ----------------------------------------------
    // 5.8 CUDA Event 计时
    // ----------------------------------------------

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEAT; ++i)
    {
        wmmaHGEMMWarpPartitionV4<<<grid, 128>>>(
            dA, dB, dC
        );
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &totalMs, start, stop
    ));

    float avgMs = totalMs / REPEAT;

    std::cout << "Average Kernel Time: "
              << avgMs << " ms"
              << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // ----------------------------------------------
    // 5.9 Device -> Host
    // ----------------------------------------------

    CUDA_CHECK(cudaMemcpy(
        hC,
        dC,
        M * N * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // ----------------------------------------------
    // 5.10 CPU 正确性验证
    // ----------------------------------------------

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

            float diff =
                std::fabs(actual - expected);

            if (diff > maxDiff)
            {
                maxDiff = diff;
            }

            if (!std::isfinite(actual) ||
                diff > 1e-3f)
            {
                ++errors;
            }
        }
    }

    // ----------------------------------------------
    // 5.11 输出结果
    // ----------------------------------------------

    std::cout << "Max difference: "
              << maxDiff << std::endl;

    if (errors == 0)
    {
        std::cout
            << "WMMA HGEMM Warp Partition V4: PASS"
            << std::endl;
    }
    else
    {
        std::cout
            << "WMMA HGEMM Warp Partition V4: FAIL ("
            << errors << " errors)"
            << std::endl;
    }

    // ----------------------------------------------
    // 5.12 释放资源
    // ----------------------------------------------

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    delete[] hA;
    delete[] hB;
    delete[] hC;

    return errors == 0 ? 0 : 1;
}




