#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/std/utility>
#include "common/cuda_check.h"

constexpr int M = 8192;
constexpr int N = 32;
constexpr int TILE = 32;
constexpr int REPEAT = 64;
constexpr int RUNS = 30;

// ============================================
// Kernel：TMA + Shared Memory Bank Conflict
// ============================================
__global__ void tmaBankV4(
    const __grid_constant__ CUtensorMap tensorMap,
    int* output,
    int swizzle
)
{
    __shared__ __align__(1024) int sData[TILE][TILE];

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;

    if (threadIdx.x == 0)
    {
        init(&bar, 1);
        cuda::device::experimental::fence_proxy_async_shared_cta();
    }

    __syncthreads();

    int startCol = blockIdx.x * TILE;
    int startRow = blockIdx.y * TILE;

    if (threadIdx.x == 0)
    {
        cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
            sData,
            &tensorMap,
            startCol,
            startRow,
            bar
        );

        auto token = cuda::device::barrier_arrive_tx(
            bar, 1, sizeof(sData)
        );

        bar.wait(cuda::std::move(token));
    }

    __syncthreads();

    // 每4个线程处理同一行
    int row = threadIdx.x / 4;
    int colInGroup = threadIdx.x % 4;

    // 保证重复访问实际执行
    volatile int* sPtr = &sData[0][0];

    int sum = 0;

    for (int repeat = 0; repeat < REPEAT; ++repeat)
    {
        for (int group = 0; group < TILE / 4; ++group)
        {
            int col = group * 4 + colInGroup;

            int physicalCol = col;

            if (swizzle == 1)
            {
                physicalCol =
                    ((row % 8) ^ (col / 4)) * 4 + col % 4;
            }

            sum += sPtr[row * TILE + physicalCol];
        }
    }

    // 每个线程写回一个累加结果
    output[(startRow + row) * 4 + colInGroup] = sum;
}

// ============================================
// Host：创建 Tensor Map、运行与验证
// ============================================
int main()
{
    int hInput[M * N];
    int hOutput[M * 4];

    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            hInput[row * N + col] = row % 100 + col;
        }
    }

    int* dInput = nullptr;
    int* dOutput = nullptr;

    CUDA_CHECK(cudaMalloc(&dInput, sizeof(hInput)));
    CUDA_CHECK(cudaMalloc(&dOutput, sizeof(hOutput)));

    CUDA_CHECK(cudaMemcpy(
        dInput, hInput, sizeof(hInput), cudaMemcpyHostToDevice
    ));

    // ========================================
    // 创建两个 Tensor Map
    // ========================================
    alignas(64) CUtensorMap tensorMapNormal{};
    alignas(64) CUtensorMap tensorMapSwizzle{};

    const cuuint64_t globalDim[2] = {N, M};
    const cuuint64_t globalStrides[1] = {N * sizeof(int)};
    const cuuint32_t boxDim[2] = {TILE, TILE};
    const cuuint32_t elementStrides[2] = {1, 1};

    CUtensorMap* maps[2] = {
        &tensorMapNormal,
        &tensorMapSwizzle
    };

    for (int mode = 0; mode < 2; ++mode)
    {
        CUresult result = cuTensorMapEncodeTiled(
            maps[mode],
            CU_TENSOR_MAP_DATA_TYPE_INT32,
            2,
            dInput,
            globalDim,
            globalStrides,
            boxDim,
            elementStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            mode == 0
                ? CU_TENSOR_MAP_SWIZZLE_NONE
                : CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );

        if (result != CUDA_SUCCESS)
        {
            const char* errorName = nullptr;
            cuGetErrorName(result, &errorName);

            std::cerr << "Tensor Map creation failed: "
                      << (errorName ? errorName : "Unknown")
                      << std::endl;

            CUDA_CHECK(cudaFree(dInput));
            CUDA_CHECK(cudaFree(dOutput));
            return 1;
        }
    }

    // ========================================
    // Benchmark
    // ========================================
    dim3 grid(N / TILE, M / TILE);

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float times[2] = {0.0f, 0.0f};

    for (int mode = 0; mode < 2; ++mode)
    {
        // 预热
        for (int i = 0; i < 5; ++i)
        {
            tmaBankV4<<<grid, 128>>>(
                *maps[mode], dOutput, mode
            );
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 正式计时
        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < RUNS; ++i)
        {
            tmaBankV4<<<grid, 128>>>(
                *maps[mode], dOutput, mode
            );
        }

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &elapsed, start, stop
        ));

        times[mode] = elapsed * 1000.0f / RUNS;

        // ====================================
        // 正确性验证
        // ====================================
        CUDA_CHECK(cudaMemcpy(
            hOutput, dOutput, sizeof(hOutput),
            cudaMemcpyDeviceToHost
        ));

        int errors = 0;

        for (int row = 0; row < M; ++row)
        {
            for (int c = 0; c < 4; ++c)
            {
                int expected = 0;

                for (int g = 0; g < TILE / 4; ++g)
                {
                    expected += hInput[row * N + g * 4 + c];
                }

                expected *= REPEAT;

                if (hOutput[row * 4 + c] != expected)
                {
                    ++errors;
                }
            }
        }

        if (mode == 0)
        {
            std::cout << "Normal: "
                      << (errors == 0 ? "PASS" : "FAIL")
                      << "  Time: " << times[mode]
                      << " us" << std::endl;
        }
        else
        {
            std::cout << "Swizzle: "
                      << (errors == 0 ? "PASS" : "FAIL")
                      << "  Time: " << times[mode]
                      << " us" << std::endl;
        }

        if (errors != 0)
        {
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            CUDA_CHECK(cudaFree(dInput));
            CUDA_CHECK(cudaFree(dOutput));
            return 1;
        }
    }

    std::cout << "Speedup: "
              << times[0] / times[1]
              << "x" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dOutput));

    return 0;
}