#include<cuda_runtime.h>
#include<cstdio>
#include<vector>
#include<cmath>

#include "common/cuda_check.h"

//矩阵形状
constexpr int M = 37;
constexpr int N = 50;
constexpr int K = 29;
// 每个 Block 的线程尺寸
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;
// ============================================================
// Kernel: 每个线程计算 C 的一个元素
// ============================================================
__global__ void naiveGEMM(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N)
    {
        float sum = 0.0f;
        for(int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
// ============================================================
// Main
// ============================================================
int main()
{
    //创建CPU端矩阵
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_ref(M * N, 0.0f);

    //初始化A和B
    for(int i = 0;i < M * K; ++i)
    {
        h_A[i] = static_cast<float>(i%7)/7.0f;
    }
    for(int i = 0;i < K * N; ++i)
    {
        h_B[i] = static_cast<float>(i%11)/11.0f;
    }
    //计算数组需要的显存大小
    size_t bytesA =  M * K * sizeof(float);
    size_t bytesB =  K * N * sizeof(float);
    size_t bytesC =  M * N * sizeof(float);

    //分配GPU显存
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A,bytesA));
    CUDA_CHECK(cudaMalloc(&d_B,bytesB));
    CUDA_CHECK(cudaMalloc(&d_C,bytesC));

    //A,B拷贝到GPU
    CUDA_CHECK(cudaMemcpy(d_A,h_A.data(),bytesA,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B,h_B.data(),bytesB,cudaMemcpyHostToDevice));

    //计算Grid和Block的尺寸
    dim3 block(BLOCK_X,BLOCK_Y);
    dim3 grid(
        (N + BLOCK_X - 1) / BLOCK_X,
        (M + BLOCK_Y - 1) / BLOCK_Y
    );
    printf("A: %d x %d\n", M, K);
    printf("B: %d x %d\n", K, N);
    printf("C: %d x %d\n", M, N);

    printf("Grid:  (%u, %u)\n", grid.x, grid.y);
    printf("Block: (%u, %u)\n", block.x, block.y);
    
    // 7. 预热
    naiveGEMM<<<grid, block>>>(
        d_A, d_B, d_C, M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // 8. 创建计时事件
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 9. 正式计时
    CUDA_CHECK(cudaEventRecord(start));

    naiveGEMM<<<grid, block>>>(
        d_A, d_B, d_C, M, N, K
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &elapsedMs, start, stop
    ));

    // 10. 将 GPU 结果复制回 CPU
    CUDA_CHECK(cudaMemcpy(
        h_C.data(), d_C, bytesC, cudaMemcpyDeviceToHost
    ));

    // 11. 使用 CPU 计算参考结果
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
    // 12. 比较 CPU 和 GPU 的结果
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
    printf("\n=== Naive GEMM ===\n");
    printf("Kernel Time: %.4f ms\n", elapsedMs);
    printf("C[0][0]: %.6f\n", h_C[0]);
    printf("Max Error: %.8f\n", maxError);
    printf("Check: %s\n", passed ? "PASS" : "FAIL");

    // 14. 销毁计时事件
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 15. 释放 GPU 显存
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed ? 0 : 1;
}




