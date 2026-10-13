#include <cuda_runtime.h>
#include <stdio.h>

__global__ void sharedReduction(
    const int* input,
    int* output
)
{
    __shared__ int sdata[8];
    int tid = threadIdx.x;
    //每个线程读取一个元素
    sdata[tid] = input[tid];
    //等待所有线程完成数据加载
    __syncthreads();
    //进行归约操作
    for(int stride = 4;stride>0;stride/=2)
    {
        if(tid<stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    //仅thread0将结果写回全局内存
    if(tid==0)
    {
        output[0] = sdata[0];
    }
}

int main()
{
    const int N = 8;

    int* input = nullptr;
    int* output = nullptr;

    // 分配统一内存
    cudaMallocManaged(&input, N * sizeof(int));
    cudaMallocManaged(&output, sizeof(int));

    // 初始化输入数据
    for (int i = 0; i < N; i++)
    {
        input[i] = i + 1;
    }
    //启动1个block，每个block 8个线程
    sharedReduction<<<1, 8>>>(input, output);

    // 等待 GPU 计算完成
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        printf("Kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(input);
        cudaFree(output);
        return 1;
    }

    // 输出最终结果
    printf("Final sum = %d\n", output[0]);

    // 释放统一内存
    cudaFree(input);
    cudaFree(output);

    return 0;
}