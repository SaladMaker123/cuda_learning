#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "common/cuda_check.h"

namespace wmma = nvcuda::wmma;

constexpr int M = 32;
constexpr int N = 32;
constexpr int K = 16;

// ============================================
// 1. Multi-Warp WMMA Kernel
// ============================================
__global__ void wmmaMultiWarp(
    const __half* A,
    const __half* B,
    float* C
)
{
    //当前线程属于哪一个WARP
    int warpIdx = threadIdx.x / 32;
    //warp在2X2网格种的位置
    int warpRow = warpIdx / 2;
    int warpCol = warpIdx % 2;
    //当前warp负责的C TILE 的起始位置
    int row = warpRow * 16;
    int col = warpCol * 16;
    //A fragment:16x16,fp16,行主序
    wmma::fragment<
        wmma::matrix_a,16,16,16,__half,wmma::row_major> a_frag;
    //B fragment:16x16,fp16,列主序
    wmma::fragment<
        wmma::matrix_b,16,16,16,__half,wmma::col_major> b_frag;
    //C fragment:16x16,fp32,累加器
    wmma::fragment<
        wmma::accumulator,16,16,16,float> c_frag;
    //累加器初始化
    wmma::fill_fragment(c_frag,0.0f);
    //加载当前warp对应的A tile
    wmma::load_matrix_sync(a_frag,A + row * K,K);
    //加载当前warp对应的B tile
    wmma::load_matrix_sync(b_frag,B + col * K,K);
    //tensor core执行矩阵乘法累加
    wmma::mma_sync(c_frag,a_frag,b_frag,c_frag);
    //将累加器写回全局内存
    wmma::store_matrix_sync(C + row * N + col,c_frag,N,wmma::mem_row_major);
}
// ============================================
// 2. 主函数
// ============================================
int main()
{
    //cpu输入输出矩阵
    __half hA[M * K];
    __half hB[K * N];
    float hC[M * N];
    //初始化A 行主序
    for(int row = 0;row < M;row++)
    {
        for(int col = 0;col < K;col++)
        {
            hA[row * K + col] = __half((row + col) % 5 -2);
        }
    }
    //初始化B 列主序
    for(int col = 0;col < N;col++)
    {
        for(int row = 0;row < K;row++)
        {
            hB[col * K + row] = __half((col * 2 + row) % 5 -2);
        }
    }
    //分配GPU内存
    __half* dA;
    __half* dB;
    float* dC;

    CUDA_CHECK(cudaMalloc(&dA,M * K *sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dB,K * N *sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dC,M * N *sizeof(float)));
    //将数据从CPU拷贝到GPU
    CUDA_CHECK(cudaMemcpy(dA,hA,M * K *sizeof(__half),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,hB,K * N *sizeof(__half),cudaMemcpyHostToDevice));
    //启动WMMA kernel
    wmmaMultiWarp<<<1,128>>>(dA,dB,dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // GPU -> CPU
    CUDA_CHECK(cudaMemcpy(
        hC, dC, sizeof(hC),
        cudaMemcpyDeviceToHost
    ));
    // CPU 参考计算和正确性验证
    int errors = 0;
    float maxDiff = 0.0f;
     for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {

            float expected = 0.0f;

            for (int k = 0; k < K; ++k) {
                expected +=
                    float(hA[row * K + k]) *
                    float(hB[col * K + k]);
            }

            float actual = hC[row * N + col];
            float diff = std::fabs(actual - expected);

            if (diff > maxDiff) {
                maxDiff = diff;
            }

            if (diff > 1e-3f) {
                ++errors;
            }
        }
    }

    std::printf("Max difference: %.6f\n", maxDiff);

    if (errors == 0) {
        std::printf("Multi-Warp WMMA: PASS\n");
    } else {
        std::printf(
            "Multi-Warp WMMA: FAIL (%d errors)\n",
            errors
        );
    }

    // 释放 GPU 内存
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return errors == 0 ? 0 : 1;
}
