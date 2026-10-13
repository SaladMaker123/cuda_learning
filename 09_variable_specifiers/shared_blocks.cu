#include <stdio.h>
__global__ void sharedBlocks()
{
    __shared__ int data[4];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    data[tid] = bid *100 + tid;
    __syncthreads();
    printf("Block %d,Thread %d,data[%d] = %d\n",bid,tid,tid,data[tid]);
}

int main()
{
    //启动2个block,每个block4个thread
    sharedBlocks<<<2,4>>>();
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        printf("kernel launch failed:%s\n",cudaGetErrorString(err));
        return 1;
    }

    //等待GPU执行完毕
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        printf("kernel execution failed!:%s\n",cudaGetErrorString(err));
        return 1;
    }

    return 0;
}