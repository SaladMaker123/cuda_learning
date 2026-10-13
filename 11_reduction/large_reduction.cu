#include <cuda_runtime.h>
#include <stdio.h>
#define BLOCK_SIZE 256

__global__ void largeReduction(
    const int* input,
    int* partial,
    int N
)
{
    __shared__ int sdata[BLOCK_SIZE];
    int tid = threadIdx.x;//block内部线程编号
    int idx = blockIdx.x * blockDim.x + threadIdx.x;//全局线程索引

    //加载数据，并处理数组边界
    if(idx<N)
    {
        sdata[tid] = input[idx];
    }
    else{
        sdata[tid] = 0;
    }
    __syncthreads();

    //block内部树形规约
    for(int stride = blockDim.x/2;stride > 0;stride/=2)
    {
        if(tid < stride)
        {
            sdata[tid] = sdata[tid] + sdata[tid + stride];
        }
        __syncthreads();
    }

    //每个block输出一个部分和
    if(tid == 0)
    {
        partial[blockIdx.x] = sdata[0];
    }
}

//第二次规约kernel
 
__global__ void finalReduction(
    const int* partial,
    int* result,
    int N
)
{

    __shared__ int sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    //每个线程计算自己的局部和
    int localSum = 0;
    for(int i = tid;i< N;i+=blockDim.x)
    {
        localSum += partial[i];
    }
    sdata[tid] = localSum;
    __syncthreads();
    // Shared Memory 树形归约
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    // Thread 0 写出最终结果
    if (tid == 0)
    {
        result[0] = sdata[0];
    }
}

int main()
{
    const int N = 1024 *1024;
    //向上取整，计算需要的block数量
    const int NUM_BLOCKS = (N+BLOCK_SIZE-1)/BLOCK_SIZE;
    
    int* input = nullptr;
    int* partial = nullptr;
    int* result = nullptr;

    //分配统一内存
    cudaError_t err = cudaMallocManaged(&input,N*sizeof(int));

    if (err != cudaSuccess)
    {
        printf("Input allocation failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }
    err = cudaMallocManaged(
        &partial,
        NUM_BLOCKS * sizeof(int)
    );

    if (err != cudaSuccess)
    {
        printf("Partial allocation failed: %s\n",
               cudaGetErrorString(err));
        cudaFree(input);
        return 1;
    }
    err = cudaMallocManaged(&result, sizeof(int));

    if (err != cudaSuccess)
    {
        printf("Result allocation failed: %s\n",
            cudaGetErrorString(err));

        cudaFree(input);
        cudaFree(partial);

        return 1;
    }

    // 初始化输入数据
    for (int i = 0; i < N; i++)
    {
        input[i] = 1;
    }

    // 启动第一阶段 Reduction
    largeReduction<<<NUM_BLOCKS, BLOCK_SIZE>>>(
        input, partial, N
    );

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
        cudaFree(partial);
        return 1;
    }

    //输出每个block的部分和
    for (int i = 0; i < NUM_BLOCKS && i < 4; i++)
    {
        printf("partial[%d] = %d\n", i, partial[i]);
    }

    // CPU 计算部分和的总和，用于验证
    int cpuSum = 0;

    for (int i = 0; i < NUM_BLOCKS; i++)
    {
        cpuSum += partial[i];
    }

    printf("CPU check sum = %d\n", cpuSum);

   

    // 第二阶段：GPU 归约 partial 数组
    finalReduction<<<1, BLOCK_SIZE>>>(
        partial, result, NUM_BLOCKS
    );

    // 检查第二个 Kernel 的启动错误
    err = cudaGetLastError();

    if (err == cudaSuccess)
    {
        err = cudaDeviceSynchronize();
    }

    if (err != cudaSuccess)
    {
        printf("Final reduction failed: %s\n",
            cudaGetErrorString(err));

        cudaFree(input);
        cudaFree(partial);
        cudaFree(result);

        return 1;
    }

    // 输出 GPU 最终结果
    printf("GPU Final sum = %d\n", result[0]);

    // 释放内存
    cudaFree(input);
    cudaFree(partial);
    cudaFree(result);

    return 0;
}


