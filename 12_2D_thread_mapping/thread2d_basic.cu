#include <cuda_runtime.h>
#include <stdio.h>

__global__ void print2DIndex()
{
    //计算线程在整个Grid中的二维坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    printf(
        "Block(%d,%d) Thread(%d,%d) -> Position(%d,%d)\n",
        blockIdx.x,
        blockIdx.y,
        threadIdx.x,
        threadIdx.y,
        x,
        y
    );
}

int main()
{
    //一个block中有4X4个线程
    dim3 block(4,4);
    //grid中有2X2个BLOCK
    dim3 grid(2,2);
    print2DIndex<<<grid,block>>>();
    cudaDeviceSynchronize();
    return 0;
}