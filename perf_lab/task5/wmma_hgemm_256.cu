#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "common/cuda_check.h"

namespace wmma = nvcuda::wmma;
constexpr int M = 256;
constexpr int N = 256;
constexpr int K = 64;
// ============================================
// Kernel：一个 Block 计算 128×128 C Tile
// ============================================
__global__ void wmmaHGEMM256(
    const __half* A,
    const __half* B,
    float* C
)
{
    // 1. 整个 Block 共享 A、B
    __shared__ __align__(32) __half sA[128 * K];
    __shared__ __align__(32) __half sB[K * 128];
    //2. 所有线程共同搬运 A
    for(int idx = threadIdx.x;idx < 128 * K;idx += blockDim.x)
    {
        sA[idx] = A[blockIdx.y * 128 * K+ idx];
    }
    // 3. 所有线程共同搬运 B
    for(int idx = threadIdx.x;idx < K * 128;idx += blockDim.x)
    {
        sB[idx] = B[blockIdx.x * 128 * K + idx];
    }
    __syncthreads();
    //4.当前warp的编号
    int warpIdx = threadIdx.x / 32;
    // 5. 每个 Warp 负责 2×8 个输出 Tile
    for(int tileRow = 0; tileRow < 2;++tileRow)
    {
        for(int tileCol = 0;tileCol < 8;++tileCol)
        {
            //当前tile在shared memory中的局部坐标
            int localRow = warpIdx * 32 + tileRow * 16;
            int localCol = tileCol * 16;
            //当前tile在完整C中的起始坐标
            int row = blockIdx.y * 128 + localRow;
            int col = blockIdx.x * 128 + localCol;
            //声明wmma fragment
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> aFrag;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bFrag;
            wmma::fragment<wmma::accumulator,16,16,16,float> cFrag;
            wmma::fill_fragment(cFrag,0.0f);
            // 7. 沿 K 方向累加
            for(int phase = 0;phase < K/16;++phase)
            {
                int k0 = phase * 16;
                // Shared A -> A Fragment
                wmma::load_matrix_sync(aFrag,sA + localRow * K + k0,K);
                // Shared B -> B Fragment
                wmma::load_matrix_sync(bFrag,sB + localCol * K + k0,K);
                // Tensor Core 矩阵乘加
                wmma::mma_sync(cFrag,aFrag,bFrag,cFrag);

            }
             // 8. 当前 Tile 完成后写回 C
            wmma::store_matrix_sync(C + row * N + col,cFrag,N,wmma::mem_row_major);
        }
    }
}

// ============================================
// Host：初始化、运行、正确性验证
// ============================================
int main()
{
    __half hA[M * K];
    __half hB[K * N];
    float hC[M * N];
     // 初始化 A：Row-Major
    for(int row = 0 ;row < M;++row)
    {
        for(int col = 0; col < K;++col)
        {
            hA[row * K + col] = __half((row + col) % 5 -2);
        }
    }
    // 初始化 B：Column-Major
    for(int col = 0;col < N;col++)
    {
        for(int row = 0;row < K;row++)
        {
            hB[col * K + row] = __half((row + 2 * col) % 5 -2);
        }
    }
    //gpu指针
    __half* dA;
    __half* dB;
    float* dC;
    //分配GPU内存
    CUDA_CHECK(cudaMalloc(&dA,sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB,sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dC,sizeof(hC)));
    // CPU -> GPU
    CUDA_CHECK(cudaMemcpy(dA,hA,sizeof(hA),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,hB,sizeof(hB),cudaMemcpyHostToDevice));
    // 一个 Block，128 个线程
    dim3 grid(N / 128, M / 128);

    wmmaHGEMM256<<<grid,128>>>(dA,dB,dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // GPU -> CPU
    CUDA_CHECK(cudaMemcpy(hC,dC,sizeof(hC),cudaMemcpyDeviceToHost));
    // CPU 参考计算
    int errors = 0;
    float maxDiff = 0.0f;
    for(int row = 0;row < M;++row)
    {
        for(int col = 0;col < N;++col)
        {
            float expected = 0.0f;
            for(int k = 0;k < K;k++)
            {
                expected += float(hA[row * K + k]) * float(hB[col * K + k]);
            }
            float actual = hC[row * N + col];
            float diff = std::fabs(actual - expected);
            if(diff > maxDiff)
            {
                maxDiff = diff;
            }
            if(diff > 1e-3f)
            {
                ++errors;
            }
        }
    }
    std::cout<<"Max difference:"<<maxDiff<<std::endl;
    if(errors == 0)
    {
        std::cout<<"WMMA HGEMM 256: PASS"<<std::endl;
    }else
    {std::cout << "WMMA HGEMM 256: FAIL ("
                  << errors << " errors)"
                  << std::endl;
    }
        
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    
    return errors == 0? 0:1;
}