#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

constexpr int DEFAULT_TILE_SIZE = 16;

// Macro for indexing
#define Index(t, row, col, stride_h, stride_w) (((t)[(row) * (stride_h) + (col) * (stride_w)]))

// CUDA error checking macro
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << " at line "        \
                      << __LINE__ << ": " << cudaGetErrorString(err)        \
                      << std::endl;                                         \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel for tiled matrix multiplication
__global__ void op_mm_kernel(float *a, float *b, float *c, int M, int N, int K, int tile_size) {
    int blockRow = blockIdx.y * blockDim.y + threadIdx.y;
    int blockCol = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float shared_mem[]; // Shared memory allocation
    float* As = shared_mem;
    float* Bs = shared_mem + tile_size * tile_size;

    float r = 0.0f;
    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int k = 0; k < (K + tile_size - 1) / tile_size; k++) {
        if (blockRow < M && k * tile_size + col < K)
            As[row * tile_size + col] = Index(a, blockRow, k * tile_size + col, K, 1);
        else
            As[row * tile_size + col] = 0.0f;

        if (k * tile_size + row < K && blockCol < N)
            Bs[row * tile_size + col] = Index(b, k * tile_size + row, blockCol, N, 1);
        else
            Bs[row * tile_size + col] = 0.0f;

        __syncthreads();

        for (int e = 0; e < tile_size; e++)
            r += As[row * tile_size + e] * Bs[e * tile_size + col];

        __syncthreads();
    }

    if (blockRow < M && blockCol < N)
        Index(c, blockRow, blockCol, N, 1) = r;
}

// Initialize matrices
__host__ void InitMatrix(int M, int N, int K, float *A, float *B, float *C) {
    for (int i = 0; i < M * K; i++) A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) B[i] = 2.0f;
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;
}

// Verify results using cuBLAS
void verify_cublas(float *A, float *B, float *C, int M, int N, int K) {
    float *c_cublas;
    float *a_d, *b_d, *c_d;
    CUDA_CHECK(cudaMalloc((void**)&a_d, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&b_d, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&c_d, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(a_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, M, K, 
                &alpha, 
                b_d, N, 
                a_d, K, 
                &beta, 
                c_d, N);

    // Copy result back to host
    c_cublas = (float*)malloc(M * N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(c_cublas, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify results
    std::cout << "cuBLAS result sample: ";
    for (int i = 0; i < 5; i++) {
        std::cout << c_cublas[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    cublasDestroy(handle);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    free(c_cublas);
}

// Host function to run tiled matrix multiplication
float run_tiling_mm(int M, int N, int K, int tile_size) {
    float *a, *a_d;
    float *b, *b_d;
    float *c, *c_d;

    // Initialize timing infra
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float eventMs = 1.0f;

    // Allocate host memory
    a = (float*)malloc(M * K * sizeof(float));
    b = (float*)malloc(K * N * sizeof(float));
    c = (float*)malloc(M * N * sizeof(float));

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&a_d, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&b_d, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&c_d, M * N * sizeof(float)));

    InitMatrix(M, N, K, a, b, c);

    CUDA_CHECK(cudaMemcpy(a_d, a, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b, K * N * sizeof(float), cudaMemcpyHostToDevice));

    int numBlocksPerSM = 0;
    int blockSize = tile_size * tile_size;
    int sharedMemSize = 2 * tile_size * tile_size * sizeof(float);

    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM, op_mm_kernel, blockSize, sharedMemSize));

    std::cout << "Max active blocks per SM: " << numBlocksPerSM << std::endl;

    dim3 dimBlock(tile_size, tile_size);
    dim3 dimGrid((N + tile_size - 1) / tile_size, (M + tile_size - 1) / tile_size);

    CUDA_CHECK(cudaEventRecord(start));
    op_mm_kernel<<<dimGrid, dimBlock, sharedMemSize>>>(a_d, b_d, c_d, M, N, K, tile_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&eventMs, start, stop));

    // Copy the result back to the host
    CUDA_CHECK(cudaMemcpy(c, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print a sample of the result
    std::cout << "Tiled MM result sample: ";
    for (int i = 0; i < 5; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // Verify with cuBLAS
    verify_cublas(a, b, c, M, N, K);

    // Cleanup
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    free(a);
    free(b);
    free(c);

    return eventMs;
}

// Main function
int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K> <TILE_SIZE>" << std::endl;
        return EXIT_FAILURE;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int tile_size = atoi(argv[4]);

    if (tile_size <= 0 || tile_size > 32) {
        std::cerr << "Invalid TILE_SIZE. Must be between 1 and 32." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Running tiled matrix multiplication with M=" << M
              << ", N=" << N << ", K=" << K << ", TILE_SIZE=" << tile_size << std::endl;

    float time_ms = run_tiling_mm(M, N, K, tile_size);

    std::cout << "Execution time: " << time_ms << " ms" << std::endl;

    return EXIT_SUCCESS;
}

