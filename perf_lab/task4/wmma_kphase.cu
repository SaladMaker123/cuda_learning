#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "common/cuda_check.h"

namespace wmma = nvcuda::wmma;
constexpr int M = 32;
constexpr int N = 32;
constexpr int K = 64;
// ============================================
// 1. Multi-Warp + K Phase WMMA Kernel
// ============================================
__global__ void wmmaKPhase(
    const __half* A,
    const __half* B,
    float* C
)
{
    // 当前线程属于哪个 Warp
    int warpIdx = threadIdx.x / 32;
    // 当前 Warp 在 2x2 网格中的位置
    int warpRow = warpIdx / 2;
    int warpCol = warpIdx % 2;
    // 当前 Warp 负责的 C Tile 起点
    int row = warpRow * 16;
    int col = warpCol * 16;
    // A Fragment
    wmma::fragment<
        wmma::matrix_a,16,16,16,
        __half,wmma::row_major
    > a_frag;
    // B Fragment
    wmma::fragment<
        wmma::matrix_b,16,16,16,
        __half,wmma::col_major
    > b_frag;
    // FP32 累加器 Fragment
    wmma::fragment<
        wmma::accumulator,16,16,16,float
    > c_frag;
    // 累加器只初始化一次
    wmma::fill_fragment(c_frag,0.0f);
    // ========================================
    // K Phase 循环
    // ========================================
    for(int phase = 0;phase < K /16;phase++)
    {
        //当前阶段的K起点
        int k0 = phase * 16;
        //A沿列方向移动
        wmma::load_matrix_sync(a_frag,A + row * K + k0,K);
        //B沿行方向移动
        wmma::load_matrix_sync(b_frag,B + col * K + k0,K);
        // 累加当前 K Phase 的矩阵乘积
        wmma::mma_sync(
            c_frag,
            a_frag,
            b_frag,
            c_frag
        );
    }

    //全部计算完成之后写回C
    wmma::store_matrix_sync(
        C + row * N + col,
        c_frag,
        N,
        wmma::mem_row_major
    );
}
// ============================================
// 2. 主函数
// ============================================
int main()
{
    // CPU 输入输出矩阵
    __half hA[M * K];
    __half hB[K * N];
    float hC[M * N];

    // 初始化 A：行主序
    for (int row = 0; row < M; ++row)
    {
        for (int k = 0; k < K; ++k)
        {
            hA[row * K + k] =
                __half((row + k) % 5 - 2);
        }
    }

    // 初始化 B：列主序
    for (int col = 0; col < N; ++col)
    {
        for (int k = 0; k < K; ++k)
        {
            hB[col * K + k] =
                __half((k + 2 * col) % 5 - 2);
        }
    }

    // 分配 GPU 内存
    __half* dA;
    __half* dB;
    float* dC;

    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(hC)));

    // CPU -> GPU
    CUDA_CHECK(cudaMemcpy(
        dA, hA, sizeof(hA),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dB, hB, sizeof(hB),
        cudaMemcpyHostToDevice
    ));

    // 一个 Block，四个 Warp
    wmmaKPhase<<<1,128>>>(dA,dB,dC);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // GPU -> CPU
    CUDA_CHECK(cudaMemcpy(
        hC, dC, sizeof(hC),
        cudaMemcpyDeviceToHost
    ));

    // ========================================
    // CPU 参考计算与正确性验证
    // ========================================

    int errors = 0;
    float maxDiff = 0.0f;

    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float expected = 0.0f;

            for (int k = 0; k < K; ++k)
            {
                expected +=
                    float(hA[row * K + k]) *
                    float(hB[col * K + k]);
            }

            float actual = hC[row * N + col];
            float diff = std::fabs(actual - expected);

            if (diff > maxDiff)
            {
                maxDiff = diff;
            }

            if (diff > 1e-3f)
            {
                ++errors;
            }
        }
    }

    std::printf("Max difference: %.6f\n", maxDiff);

    if (errors == 0)
    {
        std::printf("WMMA K Phase: PASS\n");
    }
    else
    {
        std::printf(
            "WMMA K Phase: FAIL (%d errors)\n",
            errors
        );
    }

    // 释放 GPU 内存
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return errors == 0 ? 0 : 1;
}