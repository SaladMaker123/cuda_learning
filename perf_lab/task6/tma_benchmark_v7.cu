
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/std/utility>
#include "common/cuda_check.h"

constexpr int M = 1024;
constexpr int N = 256;
constexpr int TILE = 16;
constexpr int PHASES = 4;
constexpr int RUNS = 30;

// CPU、GPU 使用相同的计算
__host__ __device__ int calculate(int value, int work)
{
    for (int i = 0; i < work; ++i)
    {
        value = (value * 3 + 7) % 1000003;
    }

    return value + 1;
}

// ============================================
// V7 Single Buffer：单缓冲
// ============================================
__global__ void tmaSingleV7(
    const __grid_constant__ CUtensorMap inputMap,
    int* output,
    int work
)
{
    __shared__ __align__(128) int sData[TILE][TILE];

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;

    if (threadIdx.x == 0)
    {
        init(&bar, 1);
        cuda::device::experimental::fence_proxy_async_shared_cta();
    }

    __syncthreads();

    int startCol = blockIdx.x * TILE;
    int startRow = blockIdx.y * TILE * PHASES;

    for (int phase = 0; phase < PHASES; ++phase)
    {
        // Thread 0 加载当前 Tile
        if (threadIdx.x == 0)
        {
            cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
                sData,
                &inputMap,
                startCol,
                startRow + phase * TILE,
                bar
            );

            auto token = cuda::device::barrier_arrive_tx(
                bar, 1, sizeof(sData)
            );

            bar.wait(cuda::std::move(token));
        }

        // 先等待当前 Tile 加载完成
        __syncthreads();

        if (threadIdx.x != 0)
        {
            for (int idx = threadIdx.x - 1;
                 idx < TILE * TILE;
                 idx += blockDim.x - 1)
            {
                int row = idx / TILE;
                int col = idx % TILE;

                int globalRow = startRow + phase * TILE + row;
                int globalCol = startCol + col;

                int value = sData[row][col];

                output[globalRow * N + globalCol] =
                    calculate(value, work);
            }
        }

        // 当前计算结束，才能加载下一批
        __syncthreads();
    }
}

// ============================================
// V7 Double Buffer：双缓冲
// ============================================
__global__ void tmaDoubleV7(
    const __grid_constant__ CUtensorMap inputMap,
    int* output,
    int work
)
{
    __shared__ __align__(128) int sData[2][TILE][TILE];

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> bar[2];

    if (threadIdx.x == 0)
    {
        init(&bar[0], 1);
        init(&bar[1], 1);

        cuda::device::experimental::fence_proxy_async_shared_cta();
    }

    __syncthreads();

    int startCol = blockIdx.x * TILE;
    int startRow = blockIdx.y * TILE * PHASES;

    // 预加载 Tile 0
    if (threadIdx.x == 0)
    {
        cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
            sData[0],
            &inputMap,
            startCol,
            startRow,
            bar[0]
        );

        auto token = cuda::device::barrier_arrive_tx(
            bar[0], 1, sizeof(sData[0])
        );

        bar[0].wait(cuda::std::move(token));
    }

    __syncthreads();

    // 双缓冲主循环
    for (int phase = 0; phase < PHASES; ++phase)
    {
        int current = phase % 2;
        int next = (phase + 1) % 2;

        if (threadIdx.x == 0)
        {
            // Thread 0 加载下一批 Tile
            if (phase + 1 < PHASES)
            {
                cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
                    sData[next],
                    &inputMap,
                    startCol,
                    startRow + (phase + 1) * TILE,
                    bar[next]
                );

                auto token = cuda::device::barrier_arrive_tx(
                    bar[next], 1, sizeof(sData[next])
                );

                bar[next].wait(cuda::std::move(token));
            }
        }
        else
        {
            // 其他线程计算当前 Tile
            for (int idx = threadIdx.x - 1;
                 idx < TILE * TILE;
                 idx += blockDim.x - 1)
            {
                int row = idx / TILE;
                int col = idx % TILE;

                int globalRow = startRow + phase * TILE + row;
                int globalCol = startCol + col;

                int value = sData[current][row][col];

                output[globalRow * N + globalCol] =
                    calculate(value, work);
            }
        }

        __syncthreads();
    }
}

// ============================================
// Host：创建 Tensor Map、Benchmark
// ============================================
int main()
{
    int hInput[M * N];
    int hOutput[M * N];

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

    alignas(64) CUtensorMap inputMap{};

    const cuuint64_t globalDim[2] = {N, M};
    const cuuint64_t globalStrides[1] = {N * sizeof(int)};
    const cuuint32_t boxDim[2] = {TILE, TILE};
    const cuuint32_t elementStrides[2] = {1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &inputMap,
        CU_TENSOR_MAP_DATA_TYPE_INT32,
        2,
        dInput,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
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

    dim3 grid(N / TILE, M / (TILE * PHASES));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int works[4] = {0, 32, 128, 256};

    for (int w = 0; w < 4; ++w)
    {
        int work = works[w];
        float times[2] = {0.0f, 0.0f};

        std::cout << "Work = " << work << std::endl;

        for (int mode = 0; mode < 2; ++mode)
        {
            // 预热
            for (int i = 0; i < 5; ++i)
            {
                if (mode == 0)
                {
                    tmaSingleV7<<<grid, 128>>>(inputMap, dOutput, work);
                }
                else
                {
                    tmaDoubleV7<<<grid, 128>>>(inputMap, dOutput, work);
                }
            }

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Benchmark
            CUDA_CHECK(cudaEventRecord(start));

            for (int i = 0; i < RUNS; ++i)
            {
                if (mode == 0)
                {
                    tmaSingleV7<<<grid, 128>>>(inputMap, dOutput, work);
                }
                else
                {
                    tmaDoubleV7<<<grid, 128>>>(inputMap, dOutput, work);
                }
            }

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float elapsed = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));

            times[mode] = elapsed * 1000.0f / RUNS;

            CUDA_CHECK(cudaMemcpy(
                hOutput, dOutput, sizeof(hOutput), cudaMemcpyDeviceToHost
            ));

            // 正确性验证
            int errors = 0;

            for (int i = 0; i < M * N; ++i)
            {
                int expected = calculate(hInput[i], work);

                if (hOutput[i] != expected)
                {
                    ++errors;
                }
            }

            std::cout
                << (mode == 0 ? "Single: " : "Double: ")
                << (errors == 0 ? "PASS" : "FAIL")
                << "  Time: " << times[mode] << " us"
                << std::endl;

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
                  << times[0] / times[1] << "x"
                  << std::endl << std::endl;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dOutput));

    return 0;
}
