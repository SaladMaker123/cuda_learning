
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#include "common/cuda_check.h"

constexpr int TILE = 32;
constexpr int TM = 4;
constexpr int TN = 4;
// ==================================================
// 1. Naive GEMM：一个线程计算一个 C 元素
// ==================================================
__global__ void naiveGEMM(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;

    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

// ==================================================
// 2. Shared GEMM：一个 Block 计算 16×16 个 C 元素
// ==================================================
__global__ void sharedGEMM(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    int numPhases = (K + TILE - 1) / TILE;

    for (int phase = 0; phase < numPhases; ++phase) {
        int aCol = phase * TILE + tx;
        int bRow = phase * TILE + ty;

        tileA[ty][tx] =
            (row < M && aCol < K)
            ? A[row * K + aCol] : 0.0f;

        tileB[ty][tx] =
            (bRow < K && col < N)
            ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ==================================================
// 3. Register GEMM：一个线程计算 2×2 个 C 元素
// ==================================================
__global__ void registerGEMM(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //当前线程负责的4X4输出区域左上角
    int row = blockIdx.y * TILE + ty * TM;
    int col = blockIdx.x * TILE + tx * TN;
    //16个累加器，全部初始化为0
    float sum[TM][TN] = {};
    int numPhase = (K + TILE -1) / TILE;
    for(int phase = 0;phase < numPhase;phase++)
    {
        //1 将A的数据加载到shared memory
        #pragma unroll
        for(int i = 0;i< TM;i++)
        {
            #pragma unroll
            for(int j = 0;j < TN;j++)
            {
                int aRow = blockIdx.y * TILE + ty * TM + i;
                int aCol = phase * TILE + tx * TN + j;
                tileA[ty * TM + i][tx * TN + j] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
            }
        }
        // 2. 将 B 的数据加载到 Shared Memory
        #pragma unroll
        for(int i = 0;i< TM;i++)
        {
            #pragma unroll
            for (int j = 0; j < TN; ++j)
            {
                int bRow = phase * TILE + ty * TM + i;
                int bCol = blockIdx.x * TILE + tx * TN + j;
                 tileB[ty * TM + i][tx * TN + j] =(bRow < K && bCol < N)? B[bRow * N + bCol] : 0.0f;
            }
        }
        __syncthreads();
        //3 沿K方向执行矩阵乘法
        #pragma unroll
        for(int k = 0;k < TILE;k++)
        {
            #pragma unroll
            for(int i = 0;i< TM; i++)
            {
                #pragma unroll
                for(int j = 0;j < TN;j++)
                {
                    sum[i][j] += tileA[ty * TM + i][k] * tileB[k][tx * TN + j];
                }
            }
        }
        __syncthreads();
    }
    // 4. 将 4×4 个结果写回 Global Memory
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {

            if (row + i < M && col + j < N) {
                C[(row + i) * N + col + j] = sum[i][j];
            }
        }
    }
}

// ==================================================
// 4. main：初始化、测试、检查结果
// ==================================================
int main()
{
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;

    constexpr int WARMUP = 10;
    constexpr int REPEATS = 100;

    CUDA_CHECK(cudaSetDevice(0));

    // 1. CPU 内存
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];
    float* h_naive = new float[M * N];

    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<float>(i % 7) / 7.0f;
    }

    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<float>(i % 11) / 11.0f;
    }

    // 2. GPU 内存
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_A, h_A, M * K * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B, h_B, K * N * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    // 3. 设置 Block 和 Grid
    dim3 blockNaive(16, 16);
    dim3 gridNaive(
        (N + 15) / 16,
        (M + 15) / 16
    );

    dim3 blockShared(32, 32);
    dim3 gridShared(
        (N + 31) / 32,
        (M + 31) / 32
    );

    dim3 blockRegister(8, 8);
    dim3 gridRegister(
        (N + 31) / 32,
        (M + 31) / 32
    );

    // 4. CUDA Event
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float totalMs = 0.0f;
    double naiveMs = 0.0;
    double sharedMs = 0.0;
    double registerMs = 0.0;

    bool passed = true;

    const double flops = 2.0 * M * N * K;

    // ==================================================
    // 5. Benchmark Naive GEMM
    // ==================================================
    for (int i = 0; i < WARMUP; ++i) {
        naiveGEMM<<<gridNaive, blockNaive>>>(
            d_A, d_B, d_C, M, N, K
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEATS; ++i) {
        naiveGEMM<<<gridNaive, blockNaive>>>(
            d_A, d_B, d_C, M, N, K
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));

    naiveMs = totalMs / REPEATS;

    CUDA_CHECK(cudaMemcpy(
        h_naive, d_C, M * N * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // 抽查 16 个输出位置，与 CPU 计算结果比较
    int testRows[4] = {0, 1, M / 2, M - 1};
    int testCols[4] = {0, 1, N / 2, N - 1};

    float maxCpuError = 0.0f;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            int row = testRows[i];
            int col = testCols[j];

            float expected = 0.0f;

            for (int k = 0; k < K; ++k) {
                expected += h_A[row * K + k]
                          * h_B[k * N + col];
            }

            float diff = std::fabs(
                h_naive[row * N + col] - expected
            );

            if (diff > maxCpuError) {
                maxCpuError = diff;
            }
        }
    }

    bool naivePass = maxCpuError < 1e-2f;
    passed = passed && naivePass;

    std::printf(
        "Naive GEMM    : %.5f ms | %.3f TFLOP/s | %s\n",
        naiveMs,
        flops / (naiveMs * 1e9),
        naivePass ? "PASS" : "FAIL"
    );

    // ==================================================
    // 6. Benchmark Shared GEMM
    // ==================================================
    for (int i = 0; i < WARMUP; ++i) {
        sharedGEMM<<<gridShared, blockShared>>>(
            d_A, d_B, d_C, M, N, K
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEATS; ++i) {
        sharedGEMM<<<gridShared, blockShared>>>(
            d_A, d_B, d_C, M, N, K
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));

    sharedMs = totalMs / REPEATS;

    CUDA_CHECK(cudaMemcpy(
        h_C, d_C, M * N * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    float maxSharedError = 0.0f;

    for (int i = 0; i < M * N; ++i) {
        float diff = std::fabs(h_C[i] - h_naive[i]);

        if (diff > maxSharedError) {
            maxSharedError = diff;
        }
    }

    bool sharedPass = maxSharedError < 1e-2f;
    passed = passed && sharedPass;

    std::printf(
        "Shared GEMM   : %.5f ms | %.3f TFLOP/s | %s\n",
        sharedMs,
        flops / (sharedMs * 1e9),
        sharedPass ? "PASS" : "FAIL"
    );

    // ==================================================
    // 7. Benchmark Register GEMM
    // ==================================================
    for (int i = 0; i < WARMUP; ++i) {
        registerGEMM<<<gridRegister, blockRegister>>>(
            d_A, d_B, d_C, M, N, K
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEATS; ++i) {
        registerGEMM<<<gridRegister, blockRegister>>>(
            d_A, d_B, d_C, M, N, K
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));

    registerMs = totalMs / REPEATS;

    CUDA_CHECK(cudaMemcpy(
        h_C, d_C, M * N * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    float maxRegisterError = 0.0f;

    for (int i = 0; i < M * N; ++i) {
        float diff = std::fabs(h_C[i] - h_naive[i]);

        if (diff > maxRegisterError) {
            maxRegisterError = diff;
        }
    }

    bool registerPass = maxRegisterError < 1e-2f;
    passed = passed && registerPass;

    std::printf(
        "Register GEMM : %.5f ms | %.3f TFLOP/s | %s\n",
        registerMs,
        flops / (registerMs * 1e9),
        registerPass ? "PASS" : "FAIL"
    );

    // ==================================================
    // 8. 输出性能比较
    // ==================================================
    std::printf("\nPerformance comparison:\n");
    std::printf("Shared / Naive   : %.2fx\n",
                naiveMs / sharedMs);
    std::printf("Register / Naive : %.2fx\n",
                naiveMs / registerMs);
    std::printf("Register / Shared: %.2fx\n",
                sharedMs / registerMs);

    std::printf("\nNaive CPU sample max error: %.8f\n",
                maxCpuError);
    std::printf("Shared vs Naive max error: %.8f\n",
                maxSharedError);
    std::printf("Register vs Naive max error: %.8f\n",
                maxRegisterError);

    // 9. 释放资源
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_naive;


    int activeBlocks = 0;

    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocks,
        registerGEMM,
        64,
        0
    ));

    printf("Register 4x4 Blocks/SM: %d\n", activeBlocks);
    printf("Register 4x4 Occupancy: %.2f%%\n",
        activeBlocks * 2.0f / 48.0f * 100.0f);

    return passed ? 0 : 1;
}























