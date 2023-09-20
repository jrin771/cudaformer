#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s, %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
}

__global__ void layer_norm(float *d_input, float *d_output, float *d_gamma, float *d_beta, int N, int K, float epsilon) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N) return;

    float mean = 0.0f;
    float var = 0.0f;

    // Compute mean
    for (int col = 0; col < K; ++col) {
        mean += d_input[row * K + col];
    }
    mean /= K;

    // Compute variance
    for (int col = 0; col < K; ++col) {
        float temp = d_input[row * K + col] - mean;
        var += temp * temp;
    }
    var /= K;

    // Compute standard deviation
    float stddev = sqrtf(var + epsilon);

    // Normalize and scale/shift
    for (int col = 0; col < K; ++col) {
        d_output[row * K + col] = d_gamma[col] * ((d_input[row * K + col] - mean) / stddev) + d_beta[col];
    }
}

int main() {
    int N = 2, K = 5;
    float epsilon = 1e-5f;

    float *h_input = (float*)malloc(N * K * sizeof(float));
    float *h_output = (float*)malloc(N * K * sizeof(float));
    float *h_gamma = (float*)malloc(K * sizeof(float));
    float *h_beta = (float*)malloc(K * sizeof(float));

    // Initialize h_input, h_gamma, and h_beta with random values
    for (int i = 0; i < N * K; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K; ++i) {
        h_gamma[i] = static_cast<float>(rand()) / RAND_MAX;
        h_beta[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_input, *d_output, *d_gamma, *d_beta;
    CHECK_CUDA(cudaMalloc((void**)&d_input, N * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_output, N * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_gamma, K * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_beta, K * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gamma, h_gamma, K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_beta, h_beta, K * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    layer_norm<<<gridDim, blockDim>>>(d_input, d_output, d_gamma, d_beta, N, K, epsilon);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * K * sizeof(float), cudaMemcpyDeviceToHost));

    // Print initial randomized values and layer normalized values  
    printf("Initial Randomized Values:\n"); 
    for (int i = 0; i < N; ++i) { 
        for (int j = 0; j < K; ++j) { 
                printf("%.4f ", h_input[i * K + j]); 
        } 
        printf("\n"); 
    }
    printf("Layer Norm Values:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
             printf("%.4f ", h_output[i * K + j]);
        }
        printf("\n");
    }
    
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_gamma));
    CHECK_CUDA(cudaFree(d_beta));
    free(h_input);
    free(h_output);
    free(h_gamma);
    free(h_beta);

    return 0;
}






