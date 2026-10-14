#include <cuda_runtime.h>
#include <stdio.h>


#define WIDTH 8
#define HEIGHT 8

#define TILE_SIZE 4



//====================================================
// Shared Memory Tiled Matrix Transpose
//====================================================

__global__ void transposeShared(
    const float* A,
    float* B,
    int width,
    int height
)
{

    // 每个Block拥有自己的Shared Memory
    __shared__ float tile[TILE_SIZE][TILE_SIZE];


    // 当前线程在线程块中的坐标

    int x = threadIdx.x;   // 列方向
    int y = threadIdx.y;   // 行方向



    //------------------------------------------------
    // 1. Global Memory -> Shared Memory
    //------------------------------------------------


    int row = blockIdx.y * TILE_SIZE + y;

    int col = blockIdx.x * TILE_SIZE + x;



    if(row < height && col < width)
    {

        /*
            A[row][col]

            保存到：

            tile[y][x]

        */

        tile[y][x] =
            A[row * width + col];

    }



    // 等待Block内所有线程加载完成

    __syncthreads();



    //------------------------------------------------
    // 2. Shared Memory -> Global Memory
    //    完成转置
    //------------------------------------------------



    /*
        Tile内部：

        tile[y][x]

        读取：

        tile[x][y]


        Block位置：

        (x,y)

        变成：

        (y,x)

    */


    int trans_row =
        blockIdx.x * TILE_SIZE + y;


    int trans_col =
        blockIdx.y * TILE_SIZE + x;



    if(trans_row < width && trans_col < height)
    {

        /*
            写入：

            B[trans_row][trans_col]


            数据：

            tile[x][y]

        */


        B[trans_row * height + trans_col]
            =
            tile[x][y];

    }

}




int main()
{

    int size = WIDTH * HEIGHT;

    size_t bytes = size * sizeof(float);



    //--------------------------------
    // CPU矩阵
    //--------------------------------

    float h_A[size];

    float h_B[size];



    for(int i=0;i<size;i++)
    {
        h_A[i] = i;
        h_B[i] = 0;
    }




    //--------------------------------
    // GPU内存
    //--------------------------------

    float* d_A;

    float* d_B;



    cudaMalloc(&d_A, bytes);

    cudaMalloc(&d_B, bytes);



    cudaMemcpy(
        d_A,
        h_A,
        bytes,
        cudaMemcpyHostToDevice
    );



    //--------------------------------
    // 设置Grid和Block
    //--------------------------------


    // 一个Block = 一个4×4 Tile

    dim3 block(
        TILE_SIZE,
        TILE_SIZE
    );



    /*
        8×8矩阵

        Tile大小4×4


        Grid:

        2×2 Block

    */

    dim3 grid(
        WIDTH / TILE_SIZE,
        HEIGHT / TILE_SIZE
    );



    //--------------------------------
    // Kernel启动
    //--------------------------------


    transposeShared<<<grid,block>>>(
        d_A,
        d_B,
        WIDTH,
        HEIGHT
    );



    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess)
    {
        printf("Kernel launch error: %s\n",
            cudaGetErrorString(err));
    }


    err = cudaDeviceSynchronize();

    if(err != cudaSuccess)
    {
        printf("Kernel execution error: %s\n",
            cudaGetErrorString(err));
    }



    //--------------------------------
    // GPU -> CPU
    //--------------------------------

    cudaMemcpy(
        h_B,
        d_B,
        bytes,
        cudaMemcpyDeviceToHost
    );



    //--------------------------------
    // 输出
    //--------------------------------


    printf("Matrix A:\n\n");


    for(int i=0;i<HEIGHT;i++)
    {
        for(int j=0;j<WIDTH;j++)
        {
            printf("%.0f ",
                h_A[i*WIDTH+j]);
        }

        printf("\n");
    }



    printf("\nTranspose Result B:\n\n");



    for(int i=0;i<WIDTH;i++)
    {
        for(int j=0;j<HEIGHT;j++)
        {
            printf("%.0f ",
                h_B[i*HEIGHT+j]);
        }

        printf("\n");
    }



    cudaFree(d_A);

    cudaFree(d_B);



    return 0;
}