#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/std/utility>
#include "common/cuda_check.h"
constexpr int M = 128;
constexpr int N = 64;
constexpr int TILE = 16;
constexpr int PHASES = 4;
// ============================================
// Kernel：TMA Double Buffer
// ============================================
__global__ void tmaDoubleBufferV6(
    const __grid_constant__ CUtensorMap inputMap,
    int* output
)
{
    // 两个 Shared Memory Buffer
    __shared__ __align__(128) int sData[2][TILE][TILE];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> bar[2];
    //初始化两个barrier
    if(threadIdx.x  == 0)
    {
        init(&bar[0],1);
        init(&bar[1],1);
        cuda::device::experimental::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    // 当前 Block 负责的区域起点
    int startCol = blockIdx.x * TILE;
    int startRow = blockIdx.y * TILE * PHASES;
    // ========================================
    // 第一阶段：预加载 Tile 0 到 Buffer 0
    // ========================================
    if(threadIdx.x == 0)
    {
        cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
            sData[0],
            &inputMap,
            startCol,
            startRow,
            bar[0]
        );
        auto token = cuda::device::barrier_arrive_tx(bar[0],1,sizeof(sData[0]));
        bar[0].wait(cuda::std::move(token));
    }
    __syncthreads();
    // ========================================
    // 第二阶段：双缓冲主循环
    // ========================================
    for(int phase = 0;phase < PHASES;++phase)
    {
        int current = phase % 2;
        int next = (phase + 1) % 2;
        //thread0:预取下一批tile
        if(threadIdx.x == 0)
        {
            if(phase + 1 < PHASES)
            {
                cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
                    sData[next],
                    &inputMap,
                    startCol,
                    startRow + (phase + 1) * TILE,
                    bar[next]
                );
                auto token = cuda::device::barrier_arrive_tx(bar[next],1,sizeof(sData[next]));
                bar[next].wait(cuda::std::move(token));
            }
        }
        // 其他 127 个线程：计算当前 Tile
        else{
            for(int idx = threadIdx.x -1;idx < TILE * TILE;idx += blockDim.x -1)
            {
                int col = idx % TILE;
                int row = idx / TILE;
                int globalRow = startRow + phase * TILE + row;
                int globalCol = startCol + col;
                output[globalRow * N + globalCol] = sData[current][row][col] + 1;
            }
        }
        //等待当前计算和下一批TMA LOAD完成
        __syncthreads();
    }
}
// ============================================
// Host：创建 Tensor Map、运行与验证
// ============================================
int main()
{
    int hInput[M * N];
    int hOutput[M * N];
    for(int row = 0;row < M;row++)
    {
        for(int col = 0;col < N;col++)
        {
            hInput[row * N + col] = row * 1000 + col;
        }
    }
    int* dInput = nullptr;
    int* dOutput = nullptr;
    CUDA_CHECK(cudaMalloc(&dInput, sizeof(hInput)));
    CUDA_CHECK(cudaMalloc(&dOutput, sizeof(hOutput)));
    CUDA_CHECK(cudaMemcpy(
        dInput, hInput, sizeof(hInput), cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemset(dOutput, 0, sizeof(hOutput)));
    // ========================================
    // 创建输入 Tensor Map
    // ========================================
    alignas(64) CUtensorMap inputMap{};
    const cuuint64_t globalDim[2] = {N,M};
    const cuuint64_t globalStrides[1] = {N * sizeof(int)};
    const cuuint32_t boxDim[2] = {TILE,TILE};
    const cuuint32_t elementStrides[2] = {1,1};

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
    // 启动 Kernel
    // ========================================
    dim3 grid(N / TILE, M / (TILE * PHASES));

    tmaDoubleBufferV6<<<grid, 128>>>(inputMap, dOutput);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        hOutput, dOutput, sizeof(hOutput), cudaMemcpyDeviceToHost
    ));

    // ========================================
    // 正确性验证
    // ========================================
    int errors = 0;

    for (int i = 0; i < M * N; ++i)
    {
        if (hOutput[i] != hInput[i] + 1)
        {
            ++errors;
        }
    }

    if (errors == 0)
    {
        std::cout << "TMA Double Buffer V6: PASS" << std::endl;
    }
    else
    {
        std::cout << "TMA Double Buffer V6: FAIL ("
                  << errors << " errors)" << std::endl;
    }

    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dOutput));

    return errors == 0 ? 0 : 1;
}