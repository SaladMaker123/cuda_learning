#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>

#include "common/cuda_check.h"

// 矩阵尺寸
constexpr int M = 37;
constexpr int N = 50;
constexpr int K = 29;
// 一个 Block 负责 16 x 16 个 C 元素
constexpr int TILE = 16;

__global__ void sharedGEMM(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K  
)
{
    //声明两块shared memory
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    //当前线程在block内部的坐标
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    //当前线程负责的C元素
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    //当前线程的累加器
    float sum = 0.0f;
    //沿K方向逐个处理TILE
    int numPhases = (K + TILE -1)/TILE;
    for(int phase = 0;phase < numPhases; ++phase)
    {
        //当前phase对应的global K索引
        int kA = phase * TILE + tx;
        int kB = phase * TILE + ty;
        //将A和B的tile加载到shared memory
        if(row < M && kA < K)
        {
            tileA[ty][tx] = A[row * K + kA];
        }
        else
        {
            tileA[ty][tx] = 0.0f;
        }
        if(kB < K && col < N)
        {
            tileB[ty][tx] = B[kB * N + col];
        }
        else
        {
            tileB[ty][tx] = 0.0f;
        }
        __syncthreads();
        // 9. 使用 Shared Memory 中的数据计算
        #pragma unroll
        for(int k = 0; k < TILE; ++k)
        {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        // 10. 等待整个 Block 完成当前 Phase
        __syncthreads();
    }
    // 11. 将结果写回全局内存
    if(row < M && col < N)
    {
        C[row * N + col ] = sum;
    }
}
// ============================================================
// Main
// ============================================================
int main()
{
    // 1. 创建 CPU 矩阵
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_ref(M * N, 0.0f);

    // 2. 初始化输入
    for (int i = 0; i < M * K; ++i)
    {
        h_A[i] = static_cast<float>(i % 7) / 7.0f;
    }

    for (int i = 0; i < K * N; ++i)
    {
        h_B[i] = static_cast<float>(i % 11) / 11.0f;
    }

    // 3. 计算显存大小
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    // 4. 分配 GPU 显存
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    // 5. 复制输入矩阵
    CUDA_CHECK(cudaMemcpy(
        d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice
    ));

    // 6. 配置线程
    dim3 block(TILE, TILE);

    dim3 grid(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    printf("A: %d x %d\n", M, K);
    printf("B: %d x %d\n", K, N);
    printf("C: %d x %d\n", M, N);

    printf("Grid:  (%u, %u)\n", grid.x, grid.y);
    printf("Block: (%u, %u)\n", block.x, block.y);

    // 7. Warmup
    sharedGEMM<<<grid, block>>>(
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

    sharedGEMM<<<grid, block>>>(
        d_A, d_B, d_C, M, N, K
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &elapsedMs, start, stop
    ));

    // 10. 将计算结果复制回 CPU
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

    // 12. 检查误差
    bool passed = true;
    float maxError = 0.0f;

    for (int i = 0; i < M * N; ++i)
    {
        float error = std::fabs(h_C[i] - h_ref[i]);

        if (error > maxError)
        {
            maxError = error;
        }

        if (error > 1e-3f)
        {
            passed = false;
        }
    }

    // 13. 输出结果
    printf("\n=== Shared Memory GEMM ===\n");
    printf("Tile Size: %d x %d\n", TILE, TILE);
    printf("K Phases: %d\n", (K + TILE - 1) / TILE);
    printf("Kernel Time: %.4f ms\n", elapsedMs);
    printf("C[0][0]: %.6f\n", h_C[0]);
    printf("Max Error: %.8f\n", maxError);
    printf("Check: %s\n", passed ? "PASS" : "FAIL");

    // 14. 清理资源
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed ? 0 : 1;
}