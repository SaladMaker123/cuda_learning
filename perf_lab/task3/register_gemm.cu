
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>

#include "common/cuda_check.h"

// 矩阵尺寸
constexpr int M = 37;
constexpr int N = 50;
constexpr int K = 29;

// Block Tile
constexpr int BM = 16;
constexpr int BN = 16;
constexpr int BK = 16;

// Thread Tile
constexpr int TM = 2;
constexpr int TN = 2;

// ============================================================
// Kernel: Register Tiling GEMM
// 每个线程计算 2 x 2 个 C 元素
// ============================================================
__global__ void registerGEMM(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    __shared__ float tileA[BM][BK];
    __shared__ float tileB[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * BM + ty * TM;
    int col = blockIdx.x * BN + tx * TN;

    float sum00 = 0.0f;
    float sum01 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;

    int numPhases = (K + BK - 1) / BK;

    for (int phase = 0; phase < numPhases; ++phase)
    {
        // 协作加载 A
        #pragma unroll
        for (int i = 0; i < TM; ++i)
        {
            #pragma unroll
            for (int j = 0; j < TN; ++j)
            {
                int aRow = blockIdx.y * BM + ty * TM + i;
                int aCol = phase * BK + tx * TN + j;

                if (aRow < M && aCol < K)
                {
                    tileA[ty * TM + i][tx * TN + j] =
                        A[aRow * K + aCol];
                }
                else
                {
                    tileA[ty * TM + i][tx * TN + j] = 0.0f;
                }
            }
        }

        // 协作加载 B
        #pragma unroll
        for (int i = 0; i < TM; ++i)
        {
            #pragma unroll
            for (int j = 0; j < TN; ++j)
            {
                int bRow = phase * BK + ty * TM + i;
                int bCol = blockIdx.x * BN + tx * TN + j;

                if (bRow < K && bCol < N)
                {
                    tileB[ty * TM + i][tx * TN + j] =
                        B[bRow * N + bCol];
                }
                else
                {
                    tileB[ty * TM + i][tx * TN + j] = 0.0f;
                }
            }
        }

        __syncthreads();

        // 四个寄存器累加器
        #pragma unroll
        for (int k = 0; k < BK; ++k)
        {
            sum00 += tileA[ty * TM][k] *
                     tileB[k][tx * TN];

            sum01 += tileA[ty * TM][k] *
                     tileB[k][tx * TN + 1];

            sum10 += tileA[ty * TM + 1][k] *
                     tileB[k][tx * TN];

            sum11 += tileA[ty * TM + 1][k] *
                     tileB[k][tx * TN + 1];
        }

        __syncthreads();
    }

    // 所有 Phase 结束后写回 C
    if (row < M && col < N)
        C[row * N + col] = sum00;

    if (row < M && col + 1 < N)
        C[row * N + col + 1] = sum01;

    if (row + 1 < M && col < N)
        C[(row + 1) * N + col] = sum10;

    if (row + 1 < M && col + 1 < N)
        C[(row + 1) * N + col + 1] = sum11;
}

// ============================================================
// Main
// ============================================================
int main()
{
    CUDA_CHECK(cudaSetDevice(0));

    // 1. 查询 GPU 属性
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // 2. CPU 矩阵
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_ref(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i)
        h_A[i] = static_cast<float>(i % 7) / 7.0f;

    for (int i = 0; i < K * N; ++i)
        h_B[i] = static_cast<float>(i % 11) / 11.0f;

    // 3. 分配 GPU 显存
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    // 4. 复制输入矩阵
    CUDA_CHECK(cudaMemcpy(
        d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice
    ));

    // 5. 配置 Grid 和 Block
    dim3 block(BN / TN, BM / TM);

    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    printf("GPU: %s\n", prop.name);
    printf("A: %d x %d\n", M, K);
    printf("B: %d x %d\n", K, N);
    printf("C: %d x %d\n", M, N);
    printf("Grid:  (%u, %u)\n", grid.x, grid.y);
    printf("Block: (%u, %u)\n", block.x, block.y);

    // ========================================================
    // 6. 新增：Occupancy 资源分析
    // ========================================================
    int threadsPerBlock = block.x * block.y;
    int activeBlocksPerSM = 0;

    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSM,
        registerGEMM,
        threadsPerBlock,
        0
    ));

    int warpsPerBlock =
        threadsPerBlock / prop.warpSize;

    int activeWarpsPerSM =
        activeBlocksPerSM * warpsPerBlock;

    int maxWarpsPerSM =
        prop.maxThreadsPerMultiProcessor / prop.warpSize;

    double occupancy =
        100.0 * activeWarpsPerSM / maxWarpsPerSM;

    printf("\n=== Occupancy Analysis ===\n");
    printf("Threads/Block: %d\n", threadsPerBlock);
    printf("Warps/Block: %d\n", warpsPerBlock);
    printf("Active Blocks/SM: %d\n", activeBlocksPerSM);
    printf("Active Warps/SM: %d\n", activeWarpsPerSM);
    printf("Max Warps/SM: %d\n", maxWarpsPerSM);
    printf("Theoretical Occupancy: %.2f%%\n", occupancy);

    // 7. Warmup
    registerGEMM<<<grid, block>>>(
        d_A, d_B, d_C, M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 8. 创建 CUDA Event
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 9. 正式计时
    CUDA_CHECK(cudaEventRecord(start));

    registerGEMM<<<grid, block>>>(
        d_A, d_B, d_C, M, N, K
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(
        &elapsedMs, start, stop
    ));

    // 10. 读取 GPU 结果
    CUDA_CHECK(cudaMemcpy(
        h_C.data(), d_C, bytesC, cudaMemcpyDeviceToHost
    ));

    // 11. CPU Reference
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k)
            {
                sum += h_A[row * K + k] *
                       h_B[k * N + col];
            }

            h_ref[row * N + col] = sum;
        }
    }

    // 12. 正确性验证
    bool passed = true;
    float maxError = 0.0f;

    for (int i = 0; i < M * N; ++i)
    {
        float error = std::fabs(h_C[i] - h_ref[i]);

        if (error > maxError)
            maxError = error;

        if (error > 1e-3f)
            passed = false;
    }

    // 13. 输出 GEMM 结果
    printf("\n=== Register Tiling GEMM ===\n");
    printf("Block Tile: %d x %d\n", BM, BN);
    printf("K Tile: %d\n", BK);
    printf("Thread Tile: %d x %d\n", TM, TN);
    printf("K Phases: %d\n", (K + BK - 1) / BK);
    printf("Kernel Time: %.4f ms\n", elapsedMs);
    printf("C[0][0]: %.6f\n", h_C[0]);
    printf("Max Error: %.8f\n", maxError);
    printf("Check: %s\n", passed ? "PASS" : "FAIL");

    // 14. 释放资源
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed ? 0 : 1;
}
