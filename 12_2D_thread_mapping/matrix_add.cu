#include <cuda_runtime.h>
#include <stdio.h>
#define WIDTH 8
#define HEIGHT 8

//GPU kernel
__global__ void matrixAdd(
    const float* A,
    const float* B,
    float* C,
    int width,
    int height
)
{
    //当前线程负责的矩阵坐标
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    //边界检查
    if(row < height && col < width){
        int index = row * width + col;
        C[index] = A[index] + B[index];
    }
}

int main()
{
    int size = WIDTH * HEIGHT;
    size_t bytes = size * sizeof(float);

    //HOST(cpu)内存
    float h_A[size];
    float h_B[size];
    float h_C[size];

    //初始化矩阵
    for(int i =0; i<size;i++)
    {
        h_A[i] = i;
        h_B[i] = i * 10;
    }

    //Device(gpu)内存
    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A,bytes);
    cudaMalloc(&d_B,bytes);
    cudaMalloc(&d_C,bytes);

    //cpu->gpu
    cudaMemcpy(
        d_A,
        h_A,
        bytes,
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_B,
        h_B,
        bytes,
        cudaMemcpyHostToDevice
    );

    //设置二维线程结构
    dim3 block(4,4);
    dim3 grid(
        (WIDTH+block.x -1)/block.x,
        (HEIGHT + block.y - 1) / block.y
    );

    matrixAdd<<<grid,block>>>(
        d_A,
        d_B,
        d_C,
        WIDTH,
        HEIGHT
    );
    cudaDeviceSynchronize();
    // GPU -> CPU

    cudaMemcpy(
        h_C,
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    );

     // 输出结果

    printf("Matrix C:\n");


    for(int i=0;i<HEIGHT;i++)
    {
        for(int j=0;j<WIDTH;j++)
        {
            printf("%.1f ", h_C[i*WIDTH+j]);
        }

        printf("\n");
    }
}