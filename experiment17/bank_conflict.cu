#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "../common/cuda_check.h"

constexpr int TILE = 32;
constexpr int BLOCK_ROWS = 8;
template<int PAD>
__global__ void transposeBank(
    const float* A,
    float* B,
    int M,
    int N
)
{
    __shared__ float tile[TILE][TILE+PAD];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    //输入矩阵A的全局坐标
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    //第一阶段，读取A
    if(row < M && col < N)
    {
        tile[ty][tx] = A[row * N + col];
    }else{
        tile[ty][tx] = 0.0f;
    }
    __syncthreads();
    //第二阶段：确定输出位置
    int outRow = blockIdx.x * TILE + ty;
    int outCol = blockIdx.y * TILE + tx;
    //第三阶段：写入B
    if(outRow < N && outCol < M)
    {
        B[outRow * M + outCol] = tile [tx][ty];
    }
}

template<int PAD>
__global__ void transposeBank256(
    const float* A,
    float* B,
    int M,
    int N
)
{
    __shared__ float tile[TILE][TILE + PAD];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    //输入A的全局坐标
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    //第一阶段：每个线程加载四个元素
    for(int j = 0;j<TILE;j+=BLOCK_ROWS)
    {
        if(row + j < M && col < N)
        {
            tile[ty + j][tx] =A[(row + j) * N + col];
        }else
        {
            tile[ty + j][tx] = 0.0f;
        }
    }
    __syncthreads();
    //输出B的起始坐标
    int outRow = blockIdx.x * TILE + ty;
    int outCol = blockIdx.y * TILE + tx;
    //第二阶段，每线程转置写回四个元素
    for(int j = 0;j < TILE; j+= BLOCK_ROWS)
    {
        if(outRow + j < N && outCol < M)
        {
            B[(outRow + j) * M + outCol] = tile[tx][ty + j];
        }
    }
}
bool runTest(int M,int N)
{
    size_t count = static_cast<size_t>(M)*N;
    size_t bytes = count * sizeof(float);

    std::vector<float> h_A(count);

    std::vector<float> h_B(count);
    std::vector<float> expected(count);

    //初始化输入矩阵
    for(size_t i = 0;i < count;i++)
    {
        h_A[i] = static_cast<float>(i+1);
    }
    //cpu计算正确的转置结果
    for(int row = 0;row < M;row++)
    {
        for(int col = 0;col < N;col++)
        {
            expected[col*M + row] = h_A[row * N + col];
        }
    }
    float* d_A = nullptr;
    float* d_B = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A,bytes));
    CUDA_CHECK(cudaMalloc(&d_B,bytes));

    CUDA_CHECK(cudaMemcpy(d_A,h_A.data(),bytes,cudaMemcpyHostToDevice));

    dim3 block(TILE,TILE);
    dim3 grid(
        (N+TILE-1)/TILE,
        (M+TILE-1)/TILE
    );
    dim3 block256(TILE, BLOCK_ROWS);
    bool allPassed = true;
    //运行普通版本
    CUDA_CHECK(cudaMemset(d_B, 0, bytes));
    transposeBank<0><<<grid,block>>>(d_A,d_B,M,N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_B.data(), d_B, bytes, cudaMemcpyDeviceToHost));
    bool normalPassed = (h_B == expected);
    //运行padding版本
    CUDA_CHECK(cudaMemset(d_B, 0, bytes));
    transposeBank<1><<<grid, block>>>(d_A, d_B, M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_B.data(), d_B, bytes, cudaMemcpyDeviceToHost));
    bool paddingPassed = (h_B == expected);
    // 新版本：256线程，不使用Padding
    CUDA_CHECK(cudaMemset(d_B, 0, bytes));
    transposeBank256<0><<<grid, block256>>>(d_A, d_B, M, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_B.data(), d_B, bytes, cudaMemcpyDeviceToHost
    ));

    bool normal256Passed = (h_B == expected);


    // 新版本：256线程，使用Padding
    CUDA_CHECK(cudaMemset(d_B, 0, bytes));
    transposeBank256<1><<<grid, block256>>>(d_A, d_B, M, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_B.data(), d_B, bytes, cudaMemcpyDeviceToHost
    ));

    bool padding256Passed = (h_B == expected);

    allPassed = normalPassed && paddingPassed
         && normal256Passed && padding256Passed;
    std::cout << "M=" << M << " N=" << N
          << " | 1024 Normal: " << (normalPassed ? "PASS" : "FAIL")
          << " | 1024 Padding: " << (paddingPassed ? "PASS" : "FAIL")
          << " | 256 Normal: " << (normal256Passed ? "PASS" : "FAIL")
          << " | 256 Padding: " << (padding256Passed ? "PASS" : "FAIL")
          << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
        return allPassed;
}

template<int PAD>
float benchmarkKernel(
    const float* d_A,
    float* d_B,
    int M,
    int N,
    dim3 grid,
    dim3 block,
    bool use256)
{
    constexpr int WARMUP = 10;
    constexpr int REPEAT = 100;

    // 预热
    for (int i = 0; i < WARMUP; ++i)
    {
        if (use256)
        {
        transposeBank256<PAD><<<grid, block>>>(d_A, d_B, M, N);
        }
        else
        {
        transposeBank<PAD><<<grid, block>>>(d_A, d_B, M, N);
        }
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEAT; ++i)
    {
         if (use256)
        {
        transposeBank256<PAD><<<grid, block>>>(d_A, d_B, M, N);
        }
        else
        {
        transposeBank<PAD><<<grid, block>>>(d_A, d_B, M, N);
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return totalMs / REPEAT;
}
void runBenchmark()
{
    constexpr int M = 2048;
    constexpr int N = 2048;

    size_t bytes = static_cast<size_t>(M) * N * sizeof(float);

    float* d_A = nullptr;
    float* d_B = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));

    CUDA_CHECK(cudaMemset(d_A, 0, bytes));

    dim3 block1024(TILE, TILE);
    dim3 block256(TILE, BLOCK_ROWS);
    dim3 grid((N + TILE - 1) / TILE,
              (M + TILE - 1) / TILE);

    // 1024线程，普通版本
    float normal1024Ms = benchmarkKernel<0>(
        d_A, d_B, M, N, grid, block1024, false);

    // 1024线程，Padding版本
    float padding1024Ms = benchmarkKernel<1>(
        d_A, d_B, M, N, grid, block1024, false);

    // 256线程，普通版本
    float normal256Ms = benchmarkKernel<0>(
        d_A, d_B, M, N, grid, block256, true);

    // 256线程，Padding版本
    float padding256Ms = benchmarkKernel<1>(
        d_A, d_B, M, N, grid, block256, true);

    std::cout << "\nBank Conflict Benchmark\n";
    std::cout << "M=" << M << " N=" << N << '\n';

    std::cout << "1024 Normal:  " << normal1024Ms << " ms\n";
    std::cout << "1024 Padding: " << padding1024Ms << " ms\n";

    std::cout << "256 Normal:   " << normal256Ms << " ms\n";
    std::cout << "256 Padding:  " << padding256Ms << " ms\n";

    std::cout << "1024 Padding Speedup: "
            << normal1024Ms / padding1024Ms << "x\n";

    std::cout << "256 Padding Speedup: "
            << normal256Ms / padding256Ms << "x\n";

    std::cout << "Thread Config Speedup (Padding): "
            << padding1024Ms / padding256Ms << "x\n";

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
}
int main()
{
    bool allPassed = true;

    allPassed &= runTest(8, 8);
    allPassed &= runTest(64, 64);
    allPassed &= runTest(37, 53);
    allPassed &= runTest(2048, 2048);

    std::cout << (allPassed ? "All tests PASSED!"
                            : "Some tests FAILED!")
              << std::endl;
    if(allPassed)
    {
        runBenchmark();
    }

    return allPassed ? 0 : 1;
}
