#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/std/utility>
#include "common/cuda_check.h"

constexpr int M = 1024;
constexpr int N = 256;
constexpr int TILE = 16;
constexpr int PHASES = 8;
constexpr int STAGES = 3;
constexpr int RUNS = 30;

// CPU、GPU 使用相同计算
__host__ __device__ int calculate(int value, int work)
{
    for (int i = 0; i < work; ++i)
    {
        value = (value * 3 + 7) % 1000003;
    }

    return value + 1;
}

// ============================================
// V9：TMA Multi-Stage Pipeline
// ============================================
__global__ void tmaMultiStageV9(
    const __grid_constant__ CUtensorMap inputMap,
    int* output,
    int work
)
{
    // 三个 Shared Memory Buffer
    __shared__ __align__(128) int sData[STAGES][TILE][TILE];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> bar[STAGES];
     // ========================================
    // 第一步：初始化三个 Barrier
    // ========================================
    if(threadIdx.x == 0)
    {
        for(int s = 0;s < STAGES;++s)
        {
            init(&bar[s],blockDim.x);
        }
        cuda::device::experimental::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    // 当前 Block 负责的区域起点
    int startCol = blockIdx.x * TILE;
    int startRow = blockIdx.y * TILE * PHASES;
    // ========================================
    // 第二步：提前加载 Tile 0、1、2
    // ========================================
    if(threadIdx.x == 0)
    {
        for(int s = 0;s < STAGES;++s)
        {
            cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
                sData[s],
                &inputMap,
                startCol,
                startRow + s * TILE,
                bar[s]
            );
            cuda::device::barrier_expect_tx(bar[s],sizeof(sData[s]));
        }
    }
    // 确保三个 TMA 请求及事务登记都已完成
    __syncthreads();
    // ========================================
    // 第三步：多阶段流水线
    // ========================================
    for(int phase = 0;phase < PHASES;++phase)
    {
        int current = phase % STAGES;
        //等待当前TILE的TMA加载完成
        auto token = bar[current].arrive();
        bar[current].wait(cuda::std::move(token));
         // Thread 1～127 计算当前 Tile
        if(threadIdx.x != 0)
        {
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
        // 所有计算线程必须结束对当前 Buffer 的读取
        __syncthreads();
        // ====================================
        // 第四步：复用当前 Buffer
        // ====================================
        if (threadIdx.x == 0)
        {
            int fetch = phase + STAGES;

            if (fetch < PHASES)
            {
                // 将未来的 Tile 加载到刚使用完的 Buffer
                cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
                    sData[current],
                    &inputMap,
                    startCol,
                    startRow + fetch * TILE,
                    bar[current]
                );

                // 为 Barrier 新阶段登记搬运字节数
                cuda::device::barrier_expect_tx(
                    bar[current], sizeof(sData[current])
                );
            }
        }
        // 确保新一轮 TMA 请求和事务登记已经发起
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

    // ========================================
    // 创建 Tensor Map
    // ========================================
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

    // ========================================
    // 启动参数
    // ========================================
    dim3 grid(N / TILE, M / (TILE * PHASES));

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int works[4] = {0, 32, 128, 256};

    // ========================================
    // 不同计算强度的 Benchmark
    // ========================================
    for (int w = 0; w < 4; ++w)
    {
        int work = works[w];

        // 预热
        for (int i = 0; i < 5; ++i)
        {
            tmaMultiStageV9<<<grid, 128>>>(
                inputMap, dOutput, work
            );
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================
        // Benchmark
        // ====================================
        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < RUNS; ++i)
        {
            tmaMultiStageV9<<<grid, 128>>>(
                inputMap, dOutput, work
            );
        }

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &elapsed, start, stop
        ));

        float timeUs = elapsed * 1000.0f / RUNS;

        // ====================================
        // 单独运行一次，检查正确性
        // ====================================
        CUDA_CHECK(cudaMemset(
            dOutput, 0, sizeof(hOutput)
        ));

        tmaMultiStageV9<<<grid, 128>>>(
            inputMap, dOutput, work
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(
            hOutput, dOutput, sizeof(hOutput),
            cudaMemcpyDeviceToHost
        ));

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
            << "Work = " << work
            << "  V9: "
            << (errors == 0 ? "PASS" : "FAIL")
            << "  Time: " << timeUs << " us"
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

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dOutput));

    return 0;
}