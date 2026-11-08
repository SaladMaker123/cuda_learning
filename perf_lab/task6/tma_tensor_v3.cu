#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/std/utility>
#include "common/cuda_check.h"

constexpr int M = 64;
constexpr int N = 64;
constexpr int TILE = 32;
// ============================================
// Kernel：TMA 2D + 128B Swizzle
// ============================================
__global__ void tmaTensorV3(
    const __grid_constant__ CUtensorMap tensorMap,
    int* output
)
{
    __shared__ __align__(1024) int sData[TILE][TILE];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    if(threadIdx.x == 0)
    {
        init(&bar,1);
        cuda::device::experimental::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    int startRow = blockIdx.y * TILE;
    int startCol = blockIdx.x * TILE;
    if(threadIdx.x == 0)
    {
        cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
            sData,
            &tensorMap,
            startCol,
            startRow,
            bar
        );
        auto token = cuda::device::barrier_arrive_tx(bar,1,sizeof(sData));
        bar.wait(cuda::std::move(token));
    }
    __syncthreads();
    // 根据 Swizzle 规则读取 Shared Memory
    for(int idx = threadIdx.x;idx < TILE * TILE;idx += blockDim.x)
    {
        int row = idx / TILE;
        int col = idx % TILE;
        int swizzledCol = ((row % 8)^(col / 4))* 4 + col % 4;
        output[(startRow + row) * N + startCol + col] = sData[row][swizzledCol] + 1;
    }
}
// ============================================
// Host：创建 Tensor Map、运行与验证
// ============================================
int main()
{
    int hInput[M * N];
    int hOutput[M * N];
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
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
    // 创建二维 Tensor Map
    // ========================================
    alignas(64) CUtensorMap tensorMap{};

    const cuuint64_t globalDim[2] = {N,M};
    const cuuint64_t globalStrides[1] = {N * sizeof(int)};
    const cuuint32_t boxDim[2] = {TILE,TILE};
    const cuuint32_t elementStrides[2] = {1,1};

    CUresult result = cuTensorMapEncodeTiled(
        &tensorMap,
        CU_TENSOR_MAP_DATA_TYPE_INT32,
        2,
        dInput,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if(result != CUDA_SUCCESS)
    {
        const char* errorName = nullptr;
        cuGetErrorName(result,&errorName);
         std::cerr << "Tensor Map creation failed: "
                  << (errorName ? errorName : "Unknown")
                  << std::endl;

        CUDA_CHECK(cudaFree(dInput));
        CUDA_CHECK(cudaFree(dOutput));
        return 1;
    }
    // 启动 Kernel
    // ========================================
    dim3 grid(N / TILE, M / TILE);

    tmaTensorV3<<<grid, 128>>>(tensorMap, dOutput);

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
        std::cout << "TMA Tensor V3: PASS" << std::endl;
    }
    else
    {
        std::cout << "TMA Tensor V3: FAIL ("
                  << errors << " errors)" << std::endl;
    }

    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dOutput));

    return errors == 0 ? 0 : 1;
 
}