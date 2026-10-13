#include <stdio.h>

__global__ void sharedTest()
{
    //每个block都有自己独立的共享数组
    __shared__ int data[4];
    int tid = threadIdx.x;
    //每个线程写入自己的位置
    data[tid] = tid +100;
    //等待同一个block所有线程完成写入
    // __syncthreads();
    //  // 每个线程读取自己写入的位置
    // printf("Block %d, Thread %d, data[%d] = %d\n",
    //        blockIdx.x, tid, tid, data[tid]);

    // 读取下一个线程写入的数据
    int next = (tid + 1) % 4;

    printf("Block %d, Thread %d, read data[%d] = %d\n",
        blockIdx.x, tid, next, data[next]);
}

int main()
{
    // 启动 2 个 Block，每个 Block 4 个线程
    sharedTest<<<2, 4>>>();

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Kernel launch failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        printf("Kernel execution failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
