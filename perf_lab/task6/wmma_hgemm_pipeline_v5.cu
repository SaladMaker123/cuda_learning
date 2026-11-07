#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include "common/cuda_check.h"
namespace wmma = nvcuda::wmma;
constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 64;
constexpr int TILE = 128;
constexpr int BK = 16;
constexpr int STAGES = 3;
constexpr int PHASES = K / BK;

constexpr int SWIZZLE_COLS = 2;
constexpr int SWIZZLE_ROWS = 4;
constexpr int WARMUP = 20;
constexpr int REPEAT = 1000;
// ==================================================
// 1. Kernel：WMMA HGEMM + 3-Stage Pipeline
// ==================================================
__global__ void wmmaHGEMMPipelineV5(
    const __half* A,
    const __half* B,
    float* C
)
{
    // ==============================================
    // 1.1 Block Swizzle
    // ==============================================
    int tilesM = M / TILE;
    int tilesN = N / TILE;
    int groupsPerRow = (tilesN + SWIZZLE_COLS -1) / SWIZZLE_COLS;
    int groupId = blockIdx.y;
    int localId = blockIdx.x;
     
    int groupRow = groupId / groupsPerRow;
    int groupCol = groupId % groupsPerRow;

    int blockRow = groupRow * SWIZZLE_ROWS + localId / SWIZZLE_COLS;
    int blockCol = groupCol * SWIZZLE_COLS + localId % SWIZZLE_COLS;
    if(blockRow >= tilesM || blockCol >= tilesN)
    {
        return;
    }
    int warpIdx = threadIdx.x / 32;
    // ==============================================
    // 1.2 Shared Memory：三级缓冲
    // ==============================================
    __shared__ __align__(32) __half sA[STAGES][TILE * BK];
    __shared__ __align__(32) __half sB[STAGES][BK * TILE];
    // ==============================================
    // 1.3 每个 Warp 的 16 个累加器
    // ==============================================
    wmma::fragment<wmma::accumulator,16,16,16,float> cFrag[16];
    #pragma unroll
    for(int i = 0;i<16;++i)
    {
        wmma::fill_fragment(cFrag[i],0.0f);
    }
    // ==============================================
    // 1.4 Prologue：预加载 Phase 0、1
    // ==============================================
    for(int p = 0;p < 2 && p<PHASES;++p)
    {
        int buffer = p % STAGES;
        int kStart = p * BK;
        for(int idx = threadIdx.x * 4;idx<TILE * BK;idx += blockDim.x * 4)
        {
            // Global A -> Shared A
            // A：行主序
            int globalARow = blockRow * TILE + idx / BK;
            int globalACol = kStart + idx % BK;
            __pipeline_memcpy_async(sA[buffer] + idx, A + globalARow * K + globalACol,8);
            // Global B -> Shared B
            // B：列主序
            int globalBRow = kStart + idx % BK;
            int globalBCol = blockCol * TILE + idx / BK;
            __pipeline_memcpy_async(sB[buffer] + idx, B + globalBCol * K + globalBRow,8);  
        }
        // 每个 Phase 单独提交一组复制
        __pipeline_commit();
    }
    // 已提交 P0、P1
    // 允许最近的 P1 尚未完成，但确保 P0 完成
    __pipeline_wait_prior(1);
    __syncthreads();
    // ==============================================
    // 1.5 Main Loop：三级流水线
    // ==============================================
    for(int phase = 0; phase < PHASES;++phase)
    {
        int current = phase % STAGES;
        int prefetch = phase + 2;
        // ------------------------------------------
        // 1.5.1 预加载后面第二个 Phase
        // ------------------------------------------
        if(prefetch < PHASES)
        {
            int nextbuffer = prefetch % STAGES;
            int nextK = prefetch * BK;
            for(int idx = threadIdx.x * 4; idx < TILE * BK; idx += blockDim.x * 4)
            {
                // Global A -> Shared A
                int globalARow = blockRow * TILE + idx / BK;
                int globalACol = nextK + idx % BK;
                __pipeline_memcpy_async(sA[nextbuffer] + idx,A + globalARow * K + globalACol,8);
                 // Global B -> Shared B
                int globalBRow = nextK + idx % BK;
                int globalBCol = blockCol * TILE + idx /BK;
                __pipeline_memcpy_async(sB[nextbuffer] + idx,B + globalBCol * K + globalBRow,8);
            }
            __pipeline_commit();
        }
        // ------------------------------------------
        // 1.5.2 当前 Phase：WMMA 计算
        // ------------------------------------------
        #pragma unroll
        for(int tileRow = 0;tileRow < 2;++tileRow)
        {
            #pragma unroll
            for(int tileCol = 0;tileCol < 8;++tileCol)
            {
                int localRow = warpIdx * 32 + tileRow * 16;
                int localCol = tileCol * 16;
                int fragIdx = tileRow * 8 + tileCol;
                wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> aFrag;
                wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bFrag;
                // Shared A -> Fragment
                wmma::load_matrix_sync(aFrag,sA[current] + localRow * BK,BK);
                // Shared B -> Fragment
                wmma::load_matrix_sync(bFrag,sB[current] + localCol * BK,BK);
                //tensor core矩阵相加
                wmma::mma_sync(cFrag[fragIdx],aFrag,bFrag,cFrag[fragIdx]);
            }
        }
        // ------------------------------------------
        // 1.5.3 确保下一 Phase 的数据已准备好
        // ------------------------------------------
        if(phase + 1 < PHASES)
        {
            if(prefetch < PHASES)
            {
                //保留最新一组异步复制
                __pipeline_wait_prior(1);
            }
            else
            {
                // 最后的剩余复制全部等待完成
                __pipeline_wait_prior(0);
            }
            // 当前计算结束，并确保下一阶段
            // 所需的数据在整个 Block 中可安全读取
            __syncthreads();
        }
    }
    // ==============================================
    // 1.6 将结果写回 Global C
    // ==============================================
    #pragma unroll
    for (int tileRow = 0; tileRow < 2; ++tileRow)
    {
        #pragma unroll
        for (int tileCol = 0; tileCol < 8; ++tileCol)
        {
            int localRow =
                warpIdx * 32 + tileRow * 16;

            int localCol =
                tileCol * 16;

            int row =
                blockRow * TILE + localRow;

            int col =
                blockCol * TILE + localCol;

            int fragIdx =
                tileRow * 8 + tileCol;

            wmma::store_matrix_sync(
                C + row * N + col,
                cFrag[fragIdx],
                N,
                wmma::mem_row_major
            );
        }
    }
}
int main()
{
    // ==============================================
    // 2.1 Host 内存
    // ==============================================
    __half* hA = new __half[M * K];
    __half* hB = new __half[K * N];
    float* hC = new float[M * N];

     // ==============================================
    // 2.2 初始化 A（行主序）
    // ==============================================

    for (int row = 0; row < M; ++row)
    {
        for (int k = 0; k < K; ++k)
        {
            hA[row * K + k] =
                __half((row + k) % 5 - 2);
        }
    }

    // ==============================================
    // 2.3 初始化 B（列主序）
    // ==============================================

    for (int col = 0; col < N; ++col)
    {
        for (int k = 0; k < K; ++k)
        {
            hB[col * K + k] =
                __half((k + 2 * col) % 5 - 2);
        }
    }
    // ==============================================
    // 2.4 Device 内存
    // ==============================================
    __half* dA = nullptr;
    __half* dB = nullptr;
    float* dC = nullptr;

    CUDA_CHECK(cudaMalloc(
        &dA, M * K * sizeof(__half)
    ));

    CUDA_CHECK(cudaMalloc(
        &dB, K * N * sizeof(__half)
    ));

    CUDA_CHECK(cudaMalloc(
        &dC, M * N * sizeof(float)
    ));

    // ==============================================
    // 2.5 Host -> Device
    // ==============================================

    CUDA_CHECK(cudaMemcpy(
        dA,
        hA,
        M * K * sizeof(__half),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dB,
        hB,
        K * N * sizeof(__half),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemset(
        dC,
        0,
        M * N * sizeof(float)
    ));
    // ==============================================
    // 2.6 Grid 配置：沿用 V3 Swizzle
    // ==============================================
    int tilesM = M / TILE;
    int tilesN = N / TILE;
    int groupsPerRow = (tilesN + SWIZZLE_COLS -1) / SWIZZLE_COLS;
    int groupsPerCol = (tilesM + SWIZZLE_ROWS -1) / SWIZZLE_ROWS;
    int groupCount = groupsPerRow * groupsPerCol;
    dim3 grid(SWIZZLE_COLS * SWIZZLE_ROWS,groupCount);

     // ==============================================
    // 2.7 Warmup：预热 20 次
    // ==============================================

    for (int i = 0; i < WARMUP; ++i)
    {
        wmmaHGEMMPipelineV5<<<grid, 128>>>(
            dA, dB, dC
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ==============================================
    // 2.8 CUDA Event 计时
    // ==============================================

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEAT; ++i)
    {
        wmmaHGEMMPipelineV5<<<grid, 128>>>(
            dA, dB, dC
        );
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &totalMs, start, stop
    ));

    float avgMs = totalMs / REPEAT;

    std::cout << "Average Kernel Time: "
              << avgMs << " ms"
              << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // ==============================================
    // 2.9 Device -> Host
    // ==============================================

    CUDA_CHECK(cudaMemcpy(
        hC,
        dC,
        M * N * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // ==============================================
    // 2.10 CPU 正确性验证
    // ==============================================

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

            float diff =
                std::fabs(actual - expected);

            if (diff > maxDiff)
            {
                maxDiff = diff;
            }

            if (!std::isfinite(actual) ||
                diff > 1e-3f)
            {
                ++errors;
            }
        }
    }

    // ==============================================
    // 2.11 输出结果
    // ==============================================

    std::cout << "Max difference: "
              << maxDiff << std::endl;

    if (errors == 0)
    {
        std::cout
            << "WMMA HGEMM 3-Stage Pipeline V5: PASS"
            << std::endl;
    }
    else
    {
        std::cout
            << "WMMA HGEMM 3-Stage Pipeline V5: FAIL ("
            << errors << " errors)"
            << std::endl;
    }

    // ==============================================
    // 2.12 释放资源
    // ==============================================

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    delete[] hA;
    delete[] hB;
    delete[] hC;

    return errors == 0 ? 0 : 1;

}