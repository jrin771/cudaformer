#include <math.h>
#include <stdio.h>
#include <stdlib.h>

const int num_heads = 2;

__global__ void fused_multi_head_attention(
    float* q, float* k, float* v,
    float* wq, float* wk, float* wv, float* wo,
    float* output, int T, int D
) {
    extern __shared__ float smem[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int t = tid / D;
    int d = tid % D;

    int head_dim = D / num_heads;
    if (head_dim == 0 || num_heads == 0) return;

    float* dot_products = smem;
    float* softmax_weights = smem + T;

    if (tid < T * D) {
        float multi_head_sum = 0.0f;
        for (int h = 0; h < num_heads; ++h) {
            float sum = 0.0f;

            if (threadIdx.x < T) {
                float dot_product = 0.0f;
                for (int j = 0; j < head_dim; ++j) {
                    int idx = h * head_dim + j;
                    dot_product += q[t * D + j] * wq[idx] * k[threadIdx.x * D + j] * wk[idx];
                }
                dot_products[threadIdx.x] = dot_product / sqrtf((float)head_dim);
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                float max_val = dot_products[0];
                for (int i = 1; i < T; ++i) {
                    max_val = fmaxf(max_val, dot_products[i]);
                }
                float exp_sum = 0.0f;
                for (int i = 0; i < T; ++i) {
                    softmax_weights[i] = expf(dot_products[i] - max_val);
                    exp_sum += softmax_weights[i];
                }
                for (int i = 0; i < T; ++i) {
                    softmax_weights[i] /= exp_sum;
                }
            }
            __syncthreads();

            if (threadIdx.x < T) {
                sum = softmax_weights[threadIdx.x] * v[threadIdx.x * D + d] * wv[h * head_dim + d];
            }
            __syncthreads();

            atomicAdd(&multi_head_sum, sum * wo[d]);
        }
        
        output[tid] = multi_head_sum;
    }
}

int main() {
    int T = 4, D = 8;
    int num_elements = T * D;

    // Allocate host memory
    float h_q[num_elements], h_k[num_elements], h_v[num_elements], h_wq[D], h_wk[D], h_wv[D], h_wo[D], h_output[num_elements];

    // Initialize host data (you can replace this with real data)
    for (int i = 0; i < num_elements; ++i) {
        h_q[i] = h_k[i] = h_v[i] = 1.0f;
    }
    for (int i = 0; i < D; ++i) {
        h_wq[i] = h_wk[i] = h_wv[i] = h_wo[i] = 1.0f;
    }

    // Allocate device memory
    float *d_q, *d_k, *d_v, *d_wq, *d_wk, *d_wv, *d_wo, *d_output;
    cudaMalloc((void**)&d_q, num_elements * sizeof(float));
    cudaMalloc((void**)&d_k, num_elements * sizeof(float));
    cudaMalloc((void**)&d_v, num_elements * sizeof(float));
    cudaMalloc((void**)&d_wq, D * sizeof(float));
    cudaMalloc((void**)&d_wk, D * sizeof(float));
    cudaMalloc((void**)&d_wv, D * sizeof(float));
    cudaMalloc((void**)&d_wo, D * sizeof(float));
    cudaMalloc((void**)&d_output, num_elements * sizeof(float));

    // Copy host to device
    cudaMemcpy(d_q, h_q, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wq, h_wq, D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wk, h_wk, D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wv, h_wv, D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wo, h_wo, D * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    fused_multi_head_attention<<<blocksPerGrid, threadsPerBlock, 2 * T * sizeof(float)>>>(
        d_q, d_k, d_v, d_wq, d_wk, d_wv, d_wo, d_output, T, D
    );

    // Copy results back to host
    cudaMemcpy(h_output, d_output, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean-up
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_wq);
    cudaFree(d_wk);
    cudaFree(d_wv);
    cudaFree(d_wo);
    cudaFree(d_output);

    // Print all matrices for validation
    printf("Matrix q:\n");
    for (int i = 0; i < T; ++i) {
        for (int j = 0; j < D; ++j) {
            printf("%.2f ", h_q[i * D + j]);
        }
        printf("\n");
    }

    printf("Matrix k:\n");
    for (int i = 0; i < T; ++i) {
        for (int j = 0; j < D; ++j) {
            printf("%.2f ", h_k[i * D + j]);
        }
        printf("\n");
    }

    printf("Matrix v:\n");
    for (int i = 0; i < T; ++i) {
        for (int j = 0; j < D; ++j) {
            printf("%.2f ", h_v[i * D + j]);
        }
        printf("\n");
    }

    printf("Output matrix:\n");
    for (int i = 0; i < T; ++i) {
        for (int j = 0; j < D; ++j) {
            printf("%.2f ", h_output[i * D + j]);
        }
        printf("\n");
    }

    return 0;
}
