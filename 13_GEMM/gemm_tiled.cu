#include <iostream>
#include <vector>
#include "../common/cuda_check.h"

#define TILE_SIZE 16

__global__ void gemmTiled(
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

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for(int phase = 0; phase<(K+TILE_SIZE-1)/TILE_SIZE;++phase)
    {
        int offset = phase * TILE_SIZE;
        //加载A tile
        if(row < M && (offset + tx)< K)
        {
            tileA[ty][tx] = A[row * K +(offset + tx)];
        }else{
            tileA[ty][tx] = 0.0f;
        }

        //加载B tile
        if((offset + ty) < K && col < N)
        {
            tileB[ty][tx] = B[(offset + ty) * N + col];
        }else{
            tileB[ty][tx] = 0.0f;
        }

        //确保当前tile加载完成
        __syncthreads();

        //使用shared memory进行乘法累加
        for(int k = 0; k < TILE_SIZE; k++)
        {
            sum += tileA[ty][k]*tileB[k][tx];
        }
        //确保所有线程完成计算后再覆盖旧tile
        __syncthreads();
    }
    if(row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}
int main() {
    //cpu端初始化数组
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<float>(i % 5 + 1);
    }

    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<float>(i % 3 + 1);
    }
    
    //GPU端
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A,bytesA));
    CUDA_CHECK(cudaMalloc(&d_B,bytesB));
    CUDA_CHECK(cudaMalloc(&d_C,bytesC));
    //cpu矩阵复制到GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice));
    //启动kernel
    dim3 block(16, 16);
    dim3 grid(
    (N + block.x - 1) / block.x,
    (M + block.y - 1) / block.y
    );
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // 1. 预热 Kernel
    gemmTiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    //2.正式性能测试
    const int repeat = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0 ;i < repeat; i++)
    {
        gemmTiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    // 3. 计算平均执行时间
    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs,start,stop));
    float avgMs = elapsedMs/repeat;
    std::cout << "Tiled GEMM average time: "<< avgMs << " ms" << std::endl;
    double flops = 2.0 * M * N * K;
    double gflops = flops/(avgMs*1e6);
    std::cout << "Tiled GEMM GFLOPS: "<< gflops << std::endl;

    
    //数据从GPU传回CPU
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytesC, cudaMemcpyDeviceToHost));
    //打印矩阵C
    // std::cout<<"GEMM result:"<<std::endl;
    // for(int row = 0; row < M; ++row)
    // {
    //     for(int col = 0; col < N; ++col)
    //     {
    //         std::cout<<h_C[row * N + col]<<"\t";
    //     }
    //     std::cout<<std::endl;
    // }

    //参考结果与计算结果进行比较
    bool passed = true;
    for(int row = 0;row <M;++row)
    {
        for(int col = 0; col < N;++col)
        {
            float expected = 0.0f;
            for (int k = 0; k < K; ++k) 
            {
            expected += h_A[row * K + k] * h_B[k * N + col];
            }
            if(h_C[row * N +col] != expected)
            {
                passed = false;
            }
        }
    }

    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
        
    return 0;
}