#include <cuda_runtime.h>
#include <stdio.h>


#define WIDTH 8
#define HEIGHT 8

//矩阵转置kernel
__global__ void transpose(
    const float* A,
    float* B,
    int width,
    int height
)
{
    //当前线程的位置
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    //边界检查
    if(row < HEIGHT && col < WIDTH)
    {
         /*
            原矩阵:

            A[row][col]

            转置:

            B[col][row]
        */
        B[col*height+row] = A[row*width+col];
    }
}
int main()
{

    int size = WIDTH * HEIGHT;
    size_t bytes = size * sizeof(float);
    // CPU数组
    float h_A[size];

    float h_B[size];
    // 初始化A

    for(int i=0;i<size;i++)
    {
        h_A[i] = i;
    }
    float* d_A;
    float* d_B;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    // CPU -> GPU
    cudaMemcpy(
        d_A,
        h_A,
        bytes,
        cudaMemcpyHostToDevice
    );
    //设置线程结构
    dim3 block(4,4);
    dim3 grid(
        (WIDTH+block.x-1)/block.x,
        (HEIGHT+block.y-1)/block.y
    );
    transpose<<<grid,block>>>(
        d_A,
        d_B,
        WIDTH,
        HEIGHT
    );
    cudaDeviceSynchronize();
    // GPU -> CPU
    cudaMemcpy(
        h_B,
        d_B,
        bytes,
        cudaMemcpyDeviceToHost
    );
    // 输出原矩阵
    printf("Marix A:\n");
    for(int i = 0;i<HEIGHT;i++)
    {
        for(int j = 0;j<WIDTH;j++)
        {
            printf("%.0f ",h_A[i*WIDTH+j]);
        }
        printf("\n");
    }

    //输出转置后的矩阵
    printf("\nMatrix B (Transpose):\n");
    for(int i=0;i<WIDTH;i++)
    {
        for(int j=0;j<HEIGHT;j++)
        {
            printf("%.0f ",h_B[i*HEIGHT+j]);
        }

        printf("\n");
    }
    cudaFree(d_A);

    cudaFree(d_B);


    return 0;
}



