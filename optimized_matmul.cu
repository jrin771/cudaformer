#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h> 

#define TILE_SIZE 4

__global__ void optimizedMatMul(float *d_A, float *d_B, float *d_C, int M, int N, int P) {
    __shared__ float ds_A[TILE_SIZE][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles, N+TILE_SIZE usually has a -1 to account for integer division if it doesn't perfectly line up but we are using an idealized example here
    for (int i = 0; i < (N + TILE_SIZE ) / TILE_SIZE; ++i) {
        // Load elements from A and B to shared memory
        if (row < M && i * TILE_SIZE + threadIdx.x < N)
            ds_A[threadIdx.y][threadIdx.x] = d_A[row * N + i * TILE_SIZE + threadIdx.x];
        else
            ds_A[threadIdx.y][threadIdx.x] = 0.0;

        if (col < P && i * TILE_SIZE + threadIdx.y < N)
            ds_B[threadIdx.y][threadIdx.x] = d_B[(i * TILE_SIZE + threadIdx.y) * P + col];
        else
            ds_B[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        // Compute inner product for this tile. 
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; ++j) {
            sum += ds_A[threadIdx.y][j] * ds_B[j][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write back result
    if (row < M && col < P)
        d_C[row * P + col] = sum;
}
void randomInitialize(float *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = (float) rand() / RAND_MAX;
    }
}
int main() {
    // Dimension definitions and data setup
    int M = 16, N = 16, P = 16;
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;   
    cudaEvent_t start, stop; 
    float elapsedTime; 

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    h_A = (float*)malloc(M * N * sizeof(float));
    h_B = (float*)malloc(N * P * sizeof(float));
    h_C = (float*)malloc(M * P * sizeof(float));

    cudaEventRecord(start, 0);

    cudaMalloc((void**)&d_A, M * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * P * sizeof(float));
    cudaMalloc((void**)&d_C, M * P * sizeof(float)); 

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to allocate memory on device: %f ms\n", elapsedTime);

    randomInitialize(h_A, M * N); 
    randomInitialize(h_B, N * P); 

    cudaEventRecord(start, 0);

    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * P * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to copy data to device: %f ms\n", elapsedTime);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((P + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    cudaEventRecord(start, 0);

    optimizedMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, P);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to execute kernel: %f ms\n", elapsedTime);

    cudaEventRecord(start, 0);

    cudaMemcpy(h_C, d_C, M * P * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to copy data from device: %f ms\n", elapsedTime);

    // print statements 
    printf("--------\n");
    printf("Matrix A:\n--------\n");
    for(int i = 0; i < M; ++i) {
            for(int j = 0; j < N; ++j) {
                    printf("%f ", h_A[i * N + j]);
            }
            printf("\n");
    }
      
    printf("--------\n");
    printf("Matrix B:\n--------\n");
    for(int i = 0; i < N; ++i) {
             for(int j = 0; j < P; ++j) {
                    printf("%f ", h_B[i * P + j]);
               }
            printf("\n");
    }
        
    printf("--------\n");
    printf("Matrix C:\n--------\n");
    for(int i = 0; i < M; ++i) {
            for(int j = 0; j < P; ++j) {
                    printf("%f ", h_C[i * P + j]);
            }
            printf("\n");
    }
    printf("--------\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); 
    cudaFree(h_A); cudaFree(h_B); cudaFree(h_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
