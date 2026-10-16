
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#define WIDTH 2048
#define HEIGHT 2048
#define TILE_SIZE 16

// ========================================
// Kernel 1: Naive Transpose
// 朴素矩阵转置
// ========================================

__global__ void transposeNaive(
    const float* A,
    float* B,
    int width,
    int height
)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width)
    {
        B[col * height + row] = A[row * width + col];
    }
}


// ========================================
// Kernel 2: Shared Memory Transpose
// 普通共享内存矩阵转置
// ========================================

__global__ void transposeShared(
    const float* A,
    float* B,
    int width,
    int height
)
{
    // 普通 Tile: 16 x 16
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int x = threadIdx.x;
    int y = threadIdx.y;

    // 原矩阵 A 的全局坐标
    int row = blockIdx.y * TILE_SIZE + y;
    int col = blockIdx.x * TILE_SIZE + x;

    // Global Memory -> Shared Memory
    if (row < height && col < width)
    {
        tile[y][x] = A[row * width + col];
    }

    __syncthreads();

    // 输出矩阵 B 的全局坐标
    int trans_row = blockIdx.x * TILE_SIZE + y;
    int trans_col = blockIdx.y * TILE_SIZE + x;

    // Shared Memory -> Global Memory
    if (trans_row < width && trans_col < height)
    {
        B[trans_row * height + trans_col] = tile[x][y];
    }
}


// ========================================
// Kernel 3: Shared Memory + Padding
// 填充列优化的矩阵转置
// ========================================

__global__ void transposePadded(
    const float* A,
    float* B,
    int width,
    int height
)
{
    // Padding: 每行增加一个 float
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    int x = threadIdx.x;
    int y = threadIdx.y;

    // 原矩阵 A 的全局坐标
    int row = blockIdx.y * TILE_SIZE + y;
    int col = blockIdx.x * TILE_SIZE + x;

    // Global Memory -> Shared Memory
    if (row < height && col < width)
    {
        tile[y][x] = A[row * width + col];
    }

    __syncthreads();

    // 输出矩阵 B 的全局坐标
    int trans_row = blockIdx.x * TILE_SIZE + y;
    int trans_col = blockIdx.y * TILE_SIZE + x;

    // Shared Memory -> Global Memory
    if (trans_row < width && trans_col < height)
    {
        B[trans_row * height + trans_col] = tile[x][y];
    }
}


// ========================================
// main: Benchmark 测试程序
// 当前阶段仅测试 Naive 的正确性
// ========================================

int main()
{
    // ------------------------------------
    // 1. 计算矩阵元素数量与字节数
    // ------------------------------------

    const size_t count = WIDTH * HEIGHT;
    const size_t bytes = count * sizeof(float);


    // ------------------------------------
    // 2. 创建 CPU 输入与输出数组
    // ------------------------------------

    std::vector<float> h_A(count);
    std::vector<float> h_B(count, 0.0f);


    // ------------------------------------
    // 3. 初始化输入矩阵 A
    // ------------------------------------

    for (size_t i = 0; i < count; ++i)
    {
        h_A[i] = static_cast<float>(i);
    }


    // ------------------------------------
    // 4. 定义 GPU 指针
    // ------------------------------------

    float* d_A = nullptr;
    float* d_B = nullptr;


    // ------------------------------------
    // 5. 分配 GPU 显存
    // ------------------------------------

    cudaError_t err = cudaMalloc(&d_A, bytes);

    if (err != cudaSuccess)
    {
        printf("cudaMalloc d_A failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc(&d_B, bytes);

    if (err != cudaSuccess)
    {
        printf("cudaMalloc d_B failed: %s\n",
               cudaGetErrorString(err));

        cudaFree(d_A);
        return 1;
    }


    // ------------------------------------
    // 6. CPU -> GPU
    // 将输入矩阵 A 复制到 GPU
    // ------------------------------------

    err = cudaMemcpy(
        d_A,
        h_A.data(),
        bytes,
        cudaMemcpyHostToDevice
    );

    if (err != cudaSuccess)
    {
        printf("cudaMemcpy H2D failed: %s\n",
               cudaGetErrorString(err));

        cudaFree(d_A);
        cudaFree(d_B);
        return 1;
    }

    printf("GPU memory allocation and copy PASSED\n");


    // ------------------------------------
    // 7. 设置 Grid 和 Block
    // ------------------------------------

    dim3 block(TILE_SIZE, TILE_SIZE);

    dim3 grid(
        (WIDTH + TILE_SIZE - 1) / TILE_SIZE,
        (HEIGHT + TILE_SIZE - 1) / TILE_SIZE
    );


    // ------------------------------------
    // 8. 启动 Naive Kernel
    // ------------------------------------

    // ========================================
    // CUDA Event Benchmark
    // ========================================
    //创建两个cuda EVENT
    // cudaEvent_t start;
    // cudaEvent_t stop;
    
    // ========================================
    // Benchmark：依次测试三个 Kernel
    // ========================================

    const int repeat = 100;

    // 保存三个 Kernel 的名称
    const char* names[3] = {
        "Naive",
        "Shared",
        "Padded"
    };

    // 保存三个 Kernel 的平均执行时间
    float times[3] = {0.0f, 0.0f, 0.0f};

    // 创建 CUDA Event
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // mode=0: Naive
    // mode=1: Shared
    // mode=2: Padded
    for (int mode = 0; mode < 3; ++mode)
    {
        // ------------------------------------
        // 1. 预热当前 Kernel
        // ------------------------------------

        if (mode == 0)
        {
            transposeNaive<<<grid, block>>>(
                d_A, d_B, WIDTH, HEIGHT);
        }
        else if (mode == 1)
        {
            transposeShared<<<grid, block>>>(
                d_A, d_B, WIDTH, HEIGHT);
        }
        else
        {
            transposePadded<<<grid, block>>>(
                d_A, d_B, WIDTH, HEIGHT);
        }

        err = cudaDeviceSynchronize();

        if (err != cudaSuccess)
        {
            printf("Warmup failed: %s\n",
                   cudaGetErrorString(err));
            return 1;
        }

        // ------------------------------------
        // 2. 开始计时
        // ------------------------------------

        cudaEventRecord(start);

        // ------------------------------------
        // 3. 重复运行100次
        // ------------------------------------

        for (int i = 0; i < repeat; ++i)
        {
            if (mode == 0)
            {
                transposeNaive<<<grid, block>>>(
                    d_A, d_B, WIDTH, HEIGHT);
            }
            else if (mode == 1)
            {
                transposeShared<<<grid, block>>>(
                    d_A, d_B, WIDTH, HEIGHT);
            }
            else
            {
                transposePadded<<<grid, block>>>(
                    d_A, d_B, WIDTH, HEIGHT);
            }
        }

        // ------------------------------------
        // 4. 结束计时
        // ------------------------------------

        cudaEventRecord(stop);

        err = cudaEventSynchronize(stop);

        if (err != cudaSuccess)
        {
            printf("Benchmark failed: %s\n",
                   cudaGetErrorString(err));
            return 1;
        }

        // ------------------------------------
        // 5. 计算平均执行时间
        // ------------------------------------

        float elapsed_ms = 0.0f;

        cudaEventElapsedTime(
            &elapsed_ms,
            start,
            stop
        );

        times[mode] = elapsed_ms / repeat;

        printf("%s Average Time: %.6f ms\n",
               names[mode],
               times[mode]);
    }

    // ------------------------------------
    // 6. 计算加速比
    // ------------------------------------

    printf("\n");

    printf("Shared Speedup: %.3fx\n",
           times[0] / times[1]);

    printf("Padded Speedup: %.3fx\n",
           times[0] / times[2]);

    printf("Padding vs Shared: %.3fx\n",
           times[1] / times[2]);

    // ------------------------------------
    // 7. 释放 CUDA Event
    // ------------------------------------

    cudaEventDestroy(start);
    cudaEventDestroy(stop);



    // ------------------------------------
    // 9. 检查 Kernel 启动错误
    // ------------------------------------

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Kernel launch failed: %s\n",
               cudaGetErrorString(err));

        cudaFree(d_A);
        cudaFree(d_B);
        return 1;
    }


    // ------------------------------------
    // 10. 等待 GPU 执行完成
    // ------------------------------------

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        printf("Kernel execution failed: %s\n",
               cudaGetErrorString(err));

        cudaFree(d_A);
        cudaFree(d_B);
        return 1;
    }


    // ------------------------------------
    // 11. GPU -> CPU
    // 复制 GPU 计算结果
    // ------------------------------------

    err = cudaMemcpy(
        h_B.data(),
        d_B,
        bytes,
        cudaMemcpyDeviceToHost
    );

    if (err != cudaSuccess)
    {
        printf("Copy result failed: %s\n",
               cudaGetErrorString(err));

        cudaFree(d_A);
        cudaFree(d_B);
        return 1;
    }


    // ------------------------------------
    // 12. CPU 正确性验证
    // ------------------------------------

    bool passed = true;

    for (int row = 0; row < HEIGHT; ++row)
    {
        for (int col = 0; col < WIDTH; ++col)
        {
            // 原矩阵 A[row][col]
            float expected =
                h_A[row * WIDTH + col];

            // 转置矩阵 B[col][row]
            float actual =
                h_B[col * HEIGHT + row];

            if (expected != actual)
            {
                passed = false;

                printf(
                    "Mismatch at A[%d][%d]: expected=%f, actual=%f\n",
                    row,
                    col,
                    expected,
                    actual
                );

                break;
            }
        }

        if (!passed)
        {
            break;
        }
    }
    // ------------------------------------
    // 13. 输出验证结果
    // ------------------------------------
    if (passed)
    {
        printf("Shared Transpose PASSED\n");
    }
    else
    {
        printf("Shared Transpose FAILED\n");
    }
    // ------------------------------------
    // 14. 释放 GPU 显存
    // ------------------------------------
    cudaFree(d_A);
    cudaFree(d_B);

    return passed ? 0 : 1;
}
