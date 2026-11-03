
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#include <cstdio>
#include <cmath>
#include <cstdlib>

#include "common/cuda_check.h"

constexpr int TILE = 32;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int STRIDE_A = TILE + 4;

// ==================================================
// 1. Single Buffer GEMM：单缓冲基线
// ==================================================

__global__ void singleBufferGEMM(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K)
{
    __shared__ __align__(16) float tileA[TILE][STRIDE_A];
    __shared__ __align__(16) float tileB[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty * TM;
    int col = blockIdx.x * TILE + tx * TN;

    float sum[TM][TN] = {};

    int numPhases = K / TILE;

    for (int phase = 0; phase < numPhases; ++phase) {

        // 加载 A Tile
        #pragma unroll
        for (int i = 0; i < TM; ++i) {

            int aRow = row + i;
            int aCol = phase * TILE + tx * TN;

            float4 values =
                *reinterpret_cast<const float4*>(
                    A + aRow * K + aCol
                );

            tileA[ty * TM + i][tx * TN]     = values.x;
            tileA[ty * TM + i][tx * TN + 1] = values.y;
            tileA[ty * TM + i][tx * TN + 2] = values.z;
            tileA[ty * TM + i][tx * TN + 3] = values.w;
        }

        // 加载 B Tile
        #pragma unroll
        for (int i = 0; i < TM; ++i) {

            int bRow = phase * TILE + ty * TM + i;
            int bCol = col;

            float4 values =
                *reinterpret_cast<const float4*>(
                    B + bRow * N + bCol
                );

            tileB[ty * TM + i][tx * TN]     = values.x;
            tileB[ty * TM + i][tx * TN + 1] = values.y;
            tileB[ty * TM + i][tx * TN + 2] = values.z;
            tileB[ty * TM + i][tx * TN + 3] = values.w;
        }

        __syncthreads();

        // Register Tiling：每线程计算 4×4
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {

            #pragma unroll
            for (int i = 0; i < TM; ++i) {

                #pragma unroll
                for (int j = 0; j < TN; ++j) {

                    sum[i][j] +=
                        tileA[ty * TM + i][k] *
                        tileB[k][tx * TN + j];
                }
            }
        }

        __syncthreads();
    }

    // 写回 C
    #pragma unroll
    for (int i = 0; i < TM; ++i) {

        #pragma unroll
        for (int j = 0; j < TN; ++j) {

            C[(row + i) * N + col + j] = sum[i][j];
        }
    }
}

// ==================================================
// 2. Double Buffer GEMM：CUDA API 异步双缓冲
// ==================================================

__global__ void doubleBufferGEMM(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K)
{
    // A 和 B 分别拥有两个 Shared Memory Buffer
    __shared__ __align__(16)
    float tileA[2][TILE][STRIDE_A];

    __shared__ __align__(16)
    float tileB[2][TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty * TM;
    int col = blockIdx.x * TILE + tx * TN;

    float sum[TM][TN] = {};

    int numPhases = K / TILE;

    // ==============================================
    // 第一步：预加载 Phase 0 到 Buffer 0
    // ==============================================

    #pragma unroll
    for (int i = 0; i < TM; ++i) {

        // 异步加载 A
        __pipeline_memcpy_async(
            &tileA[0][ty * TM + i][tx * TN],
            &A[(row + i) * K + tx * TN],
            16
        );

        // 异步加载 B
        __pipeline_memcpy_async(
            &tileB[0][ty * TM + i][tx * TN],
            &B[(ty * TM + i) * N + col],
            16
        );
    }

    // 提交当前线程的异步拷贝
    __pipeline_commit();

    // 等待当前线程的拷贝完成
    __pipeline_wait_prior(0);

    // 等待整个 Block
    __syncthreads();

    // ==============================================
    // 第二步：双缓冲主循环
    // ==============================================

    for (int phase = 0; phase < numPhases; ++phase) {

        int current = phase % 2;
        int next = (phase + 1) % 2;

        // ------------------------------------------
        // A. 提前加载下一阶段
        // ------------------------------------------

        if (phase + 1 < numPhases) {

            int nextK = (phase + 1) * TILE;

            #pragma unroll
            for (int i = 0; i < TM; ++i) {

                // 加载下一阶段 A Tile
                __pipeline_memcpy_async(
                    &tileA[next][ty * TM + i][tx * TN],
                    &A[(row + i) * K + nextK + tx * TN],
                    16
                );

                // 加载下一阶段 B Tile
                __pipeline_memcpy_async(
                    &tileB[next][ty * TM + i][tx * TN],
                    &B[(nextK + ty * TM + i) * N + col],
                    16
                );
            }

            // 提交异步拷贝，不立即等待
            __pipeline_commit();
        }

        // ------------------------------------------
        // B. 使用当前 Buffer 执行 GEMM
        // ------------------------------------------

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {

            #pragma unroll
            for (int i = 0; i < TM; ++i) {

                #pragma unroll
                for (int j = 0; j < TN; ++j) {

                    sum[i][j] +=
                        tileA[current][ty * TM + i][k] *
                        tileB[current][k][tx * TN + j];
                }
            }
        }

        // ------------------------------------------
        // C. 等待下一阶段数据搬运完成
        // ------------------------------------------

        if (phase + 1 < numPhases) {

            __pipeline_wait_prior(0);
        }

        // 等待整个 Block 完成当前阶段
        __syncthreads();
    }

    // ==============================================
    // 第三步：写回 C
    // ==============================================

    #pragma unroll
    for (int i = 0; i < TM; ++i) {

        #pragma unroll
        for (int j = 0; j < TN; ++j) {

            C[(row + i) * N + col + j] = sum[i][j];
        }
    }
}

// ==================================================
// 3. 启动 Kernel
// ==================================================

void runGEMM(
    int mode,
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M, int N, int K)
{
    dim3 block(TILE / TN, TILE / TM);

    dim3 grid(
        N / TILE,
        M / TILE
    );

    if (mode == 0) {

        singleBufferGEMM<<<grid, block>>>(
            d_A, d_B, d_C, M, N, K
        );

    } else {

        doubleBufferGEMM<<<grid, block>>>(
            d_A, d_B, d_C, M, N, K
        );
    }

    CUDA_CHECK(cudaGetLastError());
}

// ==================================================
// 4. Benchmark：10 次预热，100 次计时
// ==================================================

float benchmark(
    int mode,
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M, int N, int K)
{
    constexpr int WARMUP = 10;
    constexpr int REPEATS = 100;

    // 预热
    for (int i = 0; i < WARMUP; ++i) {

        runGEMM(mode, d_A, d_B, d_C, M, N, K);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 正式计时
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEATS; ++i) {

        runGEMM(mode, d_A, d_B, d_C, M, N, K);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &totalMs, start, stop
    ));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return totalMs / REPEATS;
}

// ==================================================
// 5. main：内存、正确性验证、性能测试
// ==================================================

int main(int argc, char* argv[])
{
    int M = 1024;
    int N = 1024;
    int K = 1024;

    // 命令行可以指定方阵规模
    if (argc > 1) {

        M = atoi(argv[1]);
        N = M;
        K = M;
    }

    // 当前实验要求尺寸是 32 的整数倍
    if (M <= 0 || M > 4096 || M % TILE != 0) {

        std::printf(
            "Matrix size must be a multiple of 32, <= 4096\n"
        );

        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));

    std::printf("Matrix: %d x %d x %d\n", M, N, K);
    std::printf(
        "TILE=%d TM=%d TN=%d STRIDE_A=%d\n",
        TILE, TM, TN, STRIDE_A
    );

    // ==============================================
    // 1. 分配 CPU 内存
    // ==============================================

    float* h_A = new float[M * K];
    float* h_B = new float[K * N];

    float* h_single = new float[M * N];
    float* h_double = new float[M * N];

    // 与之前的实验保持相同输入
    for (int i = 0; i < M * K; ++i) {

        h_A[i] = (float)(i % 7) / 7.0f;
    }

    for (int i = 0; i < K * N; ++i) {

        h_B[i] = (float)(i % 11) / 11.0f;
    }

    // ==============================================
    // 2. 分配 GPU 内存
    // ==============================================

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

    // ==============================================
    // 3. 运行单缓冲
    // ==============================================

    runGEMM(0, d_A, d_B, d_C, M, N, K);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_single, d_C, M * N * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // ==============================================
    // 4. 运行双缓冲
    // ==============================================

    runGEMM(1, d_A, d_B, d_C, M, N, K);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_double, d_C, M * N * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // ==============================================
    // 5. 完整比较两个 GPU 输出
    // ==============================================

    float maxGpuError = 0.0f;
    int gpuErrors = 0;

    for (int i = 0; i < M * N; ++i) {

        float diff =
            std::fabs(h_single[i] - h_double[i]);

        float tolerance =
            1e-2f + 1e-4f * std::fabs(h_single[i]);

        if (!std::isfinite(h_single[i]) ||
            !std::isfinite(h_double[i]) ||
            diff > tolerance) {

            gpuErrors++;

            if (gpuErrors <= 3) {

                std::printf(
                    "GPU mismatch at %d: %.6f vs %.6f\n",
                    i, h_single[i], h_double[i]
                );
            }
        }

        if (diff > maxGpuError) {
            maxGpuError = diff;
        }
    }

    // ==============================================
    // 6. CPU 参考结果验证
    // ==============================================

    // 256及以下完整验证，大矩阵抽查16个位置
    int checks = 16;

    if (M <= 256) {
        checks = M * N;
    }

    int testRows[4] = {0, 1, M / 2, M - 1};
    int testCols[4] = {0, 1, N / 2, N - 1};

    float maxCpuError = 0.0f;
    int cpuErrors = 0;

    for (int t = 0; t < checks; ++t) {

        int row;
        int col;

        if (M <= 256) {

            row = t / N;
            col = t % N;

        } else {

            row = testRows[t / 4];
            col = testCols[t % 4];
        }

        float expected = 0.0f;

        for (int k = 0; k < K; ++k) {

            expected +=
                h_A[row * K + k] *
                h_B[k * N + col];
        }

        int index = row * N + col;

        float errorSingle =
            std::fabs(h_single[index] - expected);

        float errorDouble =
            std::fabs(h_double[index] - expected);

        float tolerance =
            1e-2f + 1e-4f * std::fabs(expected);

        if (errorSingle > maxCpuError) {
            maxCpuError = errorSingle;
        }

        if (errorDouble > maxCpuError) {
            maxCpuError = errorDouble;
        }

        if (!std::isfinite(h_single[index]) ||
            !std::isfinite(h_double[index]) ||
            errorSingle > tolerance ||
            errorDouble > tolerance) {

            cpuErrors++;

            if (cpuErrors <= 3) {

                std::printf(
                    "CPU mismatch at (%d, %d)\n",
                    row, col
                );
            }
        }
    }

    // ==============================================
    // 7. 正确性结果
    // ==============================================

    bool passed = (gpuErrors == 0 && cpuErrors == 0);

    std::printf(
        "\nValidation: %s\n",
        passed ? "PASS" : "FAIL"
    );

    std::printf(
        "GPU max error: %.8f\n",
        maxGpuError
    );

    std::printf(
        "CPU checked elements: %d\n",
        checks
    );

    std::printf(
        "CPU max error: %.8f\n",
        maxCpuError
    );

    // ==============================================
    // 8. Benchmark
    // ==============================================

    if (passed) {

        float singleMs = benchmark(
            0, d_A, d_B, d_C, M, N, K
        );

        float doubleMs = benchmark(
            1, d_A, d_B, d_C, M, N, K
        );

        double flops = 2.0 * M * N * K;

        std::printf("\nPerformance comparison:\n");

        std::printf(
            "Single Buffer: %.5f ms | %.3f TFLOP/s\n",
            singleMs,
            flops / (singleMs * 1e9)
        );

        std::printf(
            "Double Buffer: %.5f ms | %.3f TFLOP/s\n",
            doubleMs,
            flops / (doubleMs * 1e9)
        );

        std::printf(
            "Speedup: %.3fx\n",
            singleMs / doubleMs
        );
    }

    // ==============================================
    // 9. 释放资源
    // ==============================================

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_single;
    delete[] h_double;

    return passed ? 0 : 1;
}
