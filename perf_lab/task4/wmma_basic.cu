#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "common/cuda_check.h"

namespace wmma = nvcuda::wmma;

constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;
// ============================================
// 1. Tensor Core WMMA Kernel
// ============================================
__global__ void wmmaBasic(
    const __half* A,
    const __half* B,
    float* C
)
{
    //A fragment:16x16,fp16,行主序
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> aFrag;
    //B fragment:16x16,fp16,列主序
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bFrag;
    // C Fragment：16x16，FP32 累加器
    wmma::fragment<wmma::accumulator,16,16,16,float> cFrag;
    // 1. 累加器初始化为 0
    wmma::fill_fragment(cFrag,0.0f);
    //2.加载A和B
    wmma::load_matrix_sync(aFrag,A,16);
    wmma::load_matrix_sync(bFrag,B,16);
    //3.tensor core矩阵相加
    wmma::mma_sync(
        cFrag,
        aFrag,
        bFrag,
        cFrag
    );
    //4.将计算结果写回C
    wmma::store_matrix_sync(C,cFrag,16,wmma::mem_row_major);
}
// ============================================
// 2. 主函数
// ============================================
int main()
{
    //CPU输入和输出矩阵
    __half hA[M * K];
    __half hB[K * N];
    float hC[M * N];
    // ========================================
    // 初始化 A：行主序
    // ========================================
    for(int row = 0; row<M; row++)
    {
        for(int col = 0;col<K;col++)
        {
            float value = float((row + col) % 5 - 2);
            hA[row * K + col] = __float2half(value);
        }
    }
    // ========================================
    // 初始化 B：列主序
    // ========================================
    for(int col = 0;col<N;++col)
    {
        for(int row = 0;row<K;++row)
        {
            float value = float((row + 2 * col) % 5 - 2);
            hB[row + col * K] = __float2half(value);
        }
    }
    //分配GPU内存
    __half* dA = nullptr;
    __half* dB = nullptr;
    float* dC = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(hC)));
    // ========================================
    // CPU -> GPU
    // ========================================
    CUDA_CHECK(cudaMemcpy(
        dA,
        hA,
        sizeof(hA),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        dB,
        hB,
        sizeof(hB),
        cudaMemcpyHostToDevice
    ));
    // ========================================
    // 启动 Kernel：1 个 Block，32 个线程
    // ========================================
    wmmaBasic<<<1, 32>>>(dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // ========================================
    // GPU -> CPU
    // ========================================
    CUDA_CHECK(cudaMemcpy(
        hC,
        dC,
        sizeof(hC),
        cudaMemcpyDeviceToHost
    ));
    // ========================================
    // CPU 参考结果和正确性检查
    // ========================================
    int errors = 0;
    float maxDiff = 0.0f;

    for(int row = 0;row<M;++row)
    {
        for(int col = 0;col<N;++col)
        {
            float expected = 0.0f;
            for(int k = 0;k<K;++k)
            {
                expected +=
                    __half2float(hA[row * K + k]) *
                    __half2float(hB[col * K + k]);
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
        std::printf("WMMA correctness: PASS\n");
    } else {
        std::printf(
            "WMMA correctness: FAIL (%d errors)\n",
            errors
        );
    }
    // ========================================
    // 释放 GPU 内存
    // ========================================

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return errors == 0 ? 0 : 1;

}