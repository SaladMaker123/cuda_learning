#include <cuda_runtime.h>
#include <stdio.h>
__global__ void blockReduction(
    const int* input,
    int* partial
)
{
    __shared__ int sdata[8];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程加载一个元素
    sdata[tid] = input[idx];

    __syncthreads();

    // 每个 Block 内部进行树形归约
    for (int stride = 4; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    // 每个 Block 输出一个部分和
    if (tid == 0)
    {
        partial[blockIdx.x] = sdata[0];
    }
}
__global__ void finalReduction(
    const int* partial,
    int* result
)
{
    if (threadIdx.x == 0)
    {
        result[0] = partial[0] + partial[1];
    }
}
int main()
{
    const int N = 16;
    const int BLOCK_SIZE = 8;
    const int NUM_BLOCKS = N / BLOCK_SIZE;
    int* result = nullptr;



    int* input = nullptr;
    int* partial = nullptr;

    // 分配统一内存
    cudaMallocManaged(&input, N * sizeof(int));
    cudaMallocManaged(&partial, NUM_BLOCKS * sizeof(int));
    cudaMallocManaged(&result, sizeof(int));

    // 初始化输入数组
    for (int i = 0; i < N; i++)
    {
        input[i] = i + 1;
    }

    // 启动两个 Block，每个 Block 8 个线程
    blockReduction<<<NUM_BLOCKS, BLOCK_SIZE>>>(input, partial);

    // 等待 GPU 执行完成
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        printf("Kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(input);
        cudaFree(partial);
        return 1;
    }

    // 输出每个 Block 的部分和
    for (int i = 0; i < NUM_BLOCKS; i++)
    {
        printf("partial[%d] = %d\n", i, partial[i]);
    }

    // 第二阶段：合并两个 Block 的部分和
    finalReduction<<<1, 1>>>(partial, result);

    // 等待第二个 Kernel 完成
    cudaError_t err2 = cudaDeviceSynchronize();

    if (err2 != cudaSuccess)
    {
        printf("Final reduction failed: %s\n",
            cudaGetErrorString(err2));

        cudaFree(input);
        cudaFree(partial);
        cudaFree(result);
        return 1;
    }

    // 输出最终求和结果
    printf("Final sum = %d\n", result[0]);

    cudaFree(input);
    cudaFree(partial);
    cudaFree(result);

    return 0;
}