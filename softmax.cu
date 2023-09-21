#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void softmax(float *d_input, float *d_output, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N) return;

    float max_val = -1e20f;
    for (int col = 0; col < K; ++col) {
        max_val = fmaxf(max_val, d_input[row * K + col]);
    }

    float sum = 0.0f;
    for (int col = 0; col < K; ++col) {
        sum += expf(d_input[row * K + col] - max_val);
    }

    for (int col = 0; col < K; ++col) {
        d_output[row * K + col] = expf(d_input[row * K + col] - max_val) / sum;
    }
}

int main() {
    int N = 2, K = 5;

    float *h_input = (float*)malloc(N * K * sizeof(float));
    float *h_output = (float*)malloc(N * K * sizeof(float));

    // Initialize h_input with random values
    for (int i = 0; i < N * K; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, N * K * sizeof(float));
    cudaMalloc((void**)&d_output, N * K * sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    
    softmax<<<gridDim, blockDim>>>(d_input, d_output, N, K);

    cudaMemcpy(h_output, d_output, N * K * sizeof(float), cudaMemcpyDeviceToHost);

    // Print both the initial randomized values and the softmax values  
    printf("Randomized Values:\n"); 
    for (int i = 0; i < N; ++i) { 
        for (int j = 0; j < K; ++j) { 
                printf("%.4f ", h_input[i * K + j]); 
        } 
        printf("\n"); 
    }
    printf("Softmax Values:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
             printf("%.4f ", h_output[i * K + j]);
        }
        printf("\n");
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}

