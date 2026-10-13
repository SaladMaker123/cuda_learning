#include <cuda_runtime.h>
#include <stdio.h>

__global__ void atomicReduction(
    const int* input,
    int* result
)
{
    __shared__ int sdata[8];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //每个线程读取一个元素
    sdata[tid] = input[idx];
    __syncthreads();

    //block内部树形规约
    for(int stride = 4;stride>0;stride/=2)
    {
        if(tid<stride)
        {
            sdata[tid] += sdata[tid+stride];
        }
        __syncthreads();
    }

    //每个block的thread0执行原子累加
    if(tid == 0)
    {
        atomicAdd(result,sdata[0]);
    }
}

int main()
{
    const int N = 16;
    const int BLOCK_SIZE = 8;
    const int NUM_BLOCKS = N / BLOCK_SIZE;

    int* input = nullptr;
    int* result = nullptr;
    
    cudaError_t err;
    // 分配统一内存
    err = cudaMallocManaged(&input, N * sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Input allocation failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }

    err = cudaMallocManaged(&result, sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Result allocation failed: %s\n",
               cudaGetErrorString(err));
        cudaFree(input);
        return 1;
    }

    // 初始化输入数组
    for (int i = 0; i < N; i++)
    {
        input[i] = i + 1;
    }
    //原子累加前必须清零
    result[0] = 0;
    //启动kernel
    atomicReduction<<<NUM_BLOCKS, BLOCK_SIZE>>>(input, result);
     // 检查 Kernel 启动错误
    err = cudaGetLastError();

    if (err == cudaSuccess)
    {
        err = cudaDeviceSynchronize();
    }

    if (err != cudaSuccess)
    {
        printf("Kernel failed: %s\n",
               cudaGetErrorString(err));

        cudaFree(input);
        cudaFree(result);
        return 1;
    }

    printf("Final sum = %d\n", result[0]);

    cudaFree(input);
    cudaFree(result);

    return 0;
}