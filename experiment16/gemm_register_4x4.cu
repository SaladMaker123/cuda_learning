#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "../common/cuda_check.h"


constexpr int TILE_M = 64;
constexpr int TILE_K = 16;
constexpr int TILE_N = 64;
// =====================================================
// Register Tiling GEMM Kernel
// 一个线程计算4X4的 C 元素
// =====================================================

__global__ void gemmRegister4x4(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) 
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    //一个线程负责计算C的4x4区域
    int row0 = blockIdx.y * TILE_M + ty * 4;
    int col0 = blockIdx.x * TILE_N + tx * 4;
    //shared memory
    __shared__ float tileA[TILE_M][TILE_K];
    __shared__ float tileB[TILE_K][TILE_N];

    //每个线程维护16个累加器
    float sum[4][4] = {};
    //沿K方向分阶段计算
    for(int phase = 0;phase < (K+TILE_K-1)/TILE_K;phase++)
    {
        int offset = phase * TILE_K;
        //加载A：每线程负责4行
        for(int i = 0;i < 4;i++)
        {
            tileA[ty*4 + i][tx] = (row0 + i< M && offset + tx < K) ? A[(row0+i)*K + offset + tx] : 0.0f;
        }
        //加载B：每线程负责四列
        for(int i = 0;i < 4;i++)
        {
            tileB[ty][tx*4 + i] = (offset + ty < K && (col0 + i) < N) ? B[(offset + ty) * N + col0 + i] : 0.0f;
        }
        __syncthreads();
        //计算4X4个输出
        for(int k = 0;k<TILE_K;++k)
        {
            #pragma unroll
            for(int i = 0;i<4;++i)
            {
                #pragma unroll
                for(int j = 0;j<4;++j)
                {
                    sum[i][j] += tileA[ty*4 + i][k] * tileB[k][tx*4 + j];
                }
            }
        }
        __syncthreads();
    }
    for(int i = 0; i<4;++i)
    {
        for(int j = 0;j<4;++j)
        {
            if(row0 + i < M && col0 + j < N)
            {
                C[(row0+i)*N + col0 + j] = sum[i][j];
            }
        }
    }
    
}
// =====================================================
// CPU 参考计算
// =====================================================
void gemmCPU(
    const std::vector<float> &A,
    const std::vector<float> &B,
    std::vector<float> &C,
    int M,
    int N,
    int K
)
{
    for(int row = 0;row < M;++row){
        for(int col = 0;col < N;++col){
            float sum = 0.0f;
            for(int k = 0;k<K;++k){
                sum += A[row * K + k] * B[k*N + col];
            }
            C[row * N + col] = sum;
        }
        
    }
}
// =====================================================
// 测试一个矩阵尺寸
// =====================================================
bool runTest(int M,int N,int K)
{
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    std::vector<float> hA(M * K);
    std::vector<float> hB(K * N);
    std::vector<float> hC(M * N, 0.0f);
    std::vector<float> expected(M * N, 0.0f);
    // 初始化测试数据
    for (int i = 0; i < M * K; ++i) {
        hA[i] = static_cast<float>((i % 7) - 3);
    }
    for (int i = 0; i < K * N; ++i) {
        hB[i] = static_cast<float>((i % 5) - 2);
    }
    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    CUDA_CHECK(cudaMemcpy(
        dA, hA.data(), bytesA, cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dB, hB.data(), bytesB, cudaMemcpyHostToDevice
    ));

    //配置kernel
    dim3 block(16,16);
    dim3 grid(
        (N+TILE_N-1)/TILE_N,
        (M+TILE_M-1)/TILE_M
    );
    //gpu计算
    gemmRegister4x4<<<grid,block>>>(
        dA,dB,dC,M,N,K
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    //取回GPU计算结果
    CUDA_CHECK(cudaMemcpy(hC.data(),dC,bytesC,cudaMemcpyDeviceToHost));
    // CPU 参考计算
    gemmCPU(hA, hB, expected, M, N, K);
    
    // 逐元素比较
    bool passed = true;
    float maxError = 0.0f;
    
    for (int i = 0; i < M * N; ++i) {

        float error = std::fabs(hC[i] - expected[i]);

        if (error > maxError) {
            maxError = error;
        }

        if (error > 1e-3f) {
            passed = false;

            std::cout << "Mismatch at index " << i
                    << ": GPU=" << hC[i]
                    << ", CPU=" << expected[i]
                    << std::endl;

            break;
        }
    }

    std::cout << "M=" << M
            << " N=" << N
            << " K=" << K
            << " | " << (passed ? "PASSED" : "FAILED")
            << " | Max error=" << maxError
            << std::endl;

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return passed;
}

void runBenchmark(int M, int N, int K)
{
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    std::vector<float> hA(M * K);
    std::vector<float> hB(K * N);

    for (int i = 0; i < M * K; ++i) {
        hA[i] = static_cast<float>((i % 7) - 3);
    }

    for (int i = 0; i < K * N; ++i) {
        hB[i] = static_cast<float>((i % 5) - 2);
    }

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    CUDA_CHECK(cudaMemcpy(
        dA, hA.data(), bytesA, cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dB, hB.data(), bytesB, cudaMemcpyHostToDevice
    ));

    dim3 block(16, 16);

    dim3 grid(
        (N + TILE_N - 1) / TILE_N,
        (M + TILE_M - 1) / TILE_M
    );

    // 1. 预热
    constexpr int warmup = 10;

    for (int i = 0; i < warmup; ++i) {
        gemmRegister4x4<<<grid, block>>>(
            dA, dB, dC, M, N, K
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2. 创建 CUDA Event
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 3. 正式计时
    constexpr int repeat = 100;

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        gemmRegister4x4<<<grid, block>>>(
            dA, dB, dC, M, N, K
        );
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // 4. 计算平均执行时间
    float totalMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &totalMs, start, stop
    ));

    float avgMs = totalMs / repeat;

    // 5. 计算 GFLOPS
    double flops = 2.0 * M * N * K;
    double gflops = flops / (avgMs * 1e6);

    std::cout << "Register 4X4 Benchmark\n"
              << "M=" << M
              << " N=" << N
              << " K=" << K << "\n"
              << "Average time: " << avgMs << " ms\n"
              << "Performance: " << gflops << " GFLOPS"
              << std::endl;

    // 6. 释放资源
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}
int main()
{
    bool allPassed = true;
    allPassed &= runTest(2, 4, 2);
    allPassed &= runTest(7, 10, 5);
    allPassed &= runTest(37, 53, 29);
    allPassed &= runTest(70, 95, 29);
    if (allPassed) {
    std::cout << "All tests PASSED!" << std::endl;
    } else {
        std::cout << "Some tests FAILED!" << std::endl;
    }
    if (allPassed) {
    runBenchmark(1024, 1024, 1024);


    // =====================================================
    // 实验 14.4.3：查询 Register Tiling 理论 Occupancy
    // =====================================================

    // 1. 获取当前 GPU 设备编号
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    // 2. 获取 GPU 硬件属性
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // 3. 查询每个 SM 最多可以驻留多少个 Block
    int activeBlocks = 0;

    int threadsPerBlock = 16 * 16;

    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocks,
        gemmRegister4x4,
        threadsPerBlock,
        0
    ));

    // 4. 计算每个 SM 的理论驻留线程数量
    int activeThreads = activeBlocks * threadsPerBlock;

    // 5. 计算理论 Occupancy（百分比）
    double occupancy =
        100.0 * activeThreads / prop.maxThreadsPerMultiProcessor;

    // 6. 输出结果
    std::cout << "Register Tiling Occupancy:" << std::endl;

    std::cout << "Active blocks per SM: "
            << activeBlocks << std::endl;

    std::cout << "Active threads per SM: "
            << activeThreads << std::endl;

    std::cout << "Theoretical occupancy: "
            << occupancy << "%" << std::endl;
        
    return allPassed ? 0 : 1;

    }
}