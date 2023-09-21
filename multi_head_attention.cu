#include <math.h>
#include <cstdio>

const int num_heads = 2;  // Number of attention heads

__global__ void multi_head_attention(
    float* q, float* k, float* v,
    float* wq, float* wk, float* wv, float* wo,
    float* output, int T, int D
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < T * D) {
        int t = tid / D;
        int d = tid % D;
        int head_dim = D / num_heads;

        float multi_head_sum = 0.0;

        for (int h = 0; h < num_heads; ++h) {
            float sum = 0.0;

            for (int i = 0; i < T; ++i) {
                float dot_product = 0.0;

                for (int j = 0; j < head_dim; ++j) {
                    int idx = h * head_dim + j;
                    dot_product += q[t * D + j] * wq[idx] * k[i * D + j] * wk[idx];
                }

                dot_product /= sqrtf((float)head_dim);
                float weight = expf(dot_product);
                sum += weight * v[i * D + d] * wv[h * head_dim + d];
            }

            multi_head_sum += sum * wo[d];
        }

        output[tid] = multi_head_sum;
    }
}

int main() {
    // Dimensions and constants
    const int T = 3, D = 4;
    printf("T = %d, D = %d\n", T, D);

    // Initialize input and weight matrices on host
    float h_q[T * D] = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    float h_k[T * D] = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    float h_v[T * D] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
    float h_wq[num_heads * D] = {1.1, 0.9, 0.7, 0.8, 1.0, 1.2, 0.6, 1.1};
    float h_wk[num_heads * D] = {0.9, 1.0, 0.8, 1.2, 0.7, 1.1, 0.6, 1.0};
    float h_wv[num_heads * D] = {1.0, 0.9, 1.1, 0.7, 0.8, 1.2, 1.0, 0.9};
    float h_wo[D] = {1.0, 0.9, 1.1, 0.8};

    // Print initial arrays
    printf("q = ");
    for (int i = 0; i < T * D; ++i) printf("%f ", h_q[i]);
    printf("\n");

    printf("k = ");
    for (int i = 0; i < T * D; ++i) printf("%f ", h_k[i]);
    printf("\n");

    printf("v = ");
    for (int i = 0; i < T * D; ++i) printf("%f ", h_v[i]);
    printf("\n");

    printf("wq = ");
    for (int i = 0; i < num_heads * D; ++i) printf("%f ", h_wq[i]);
    printf("\n");

    printf("wk = ");
    for (int i = 0; i < num_heads * D; ++i) printf("%f ", h_wk[i]);
    printf("\n");

    printf("wv = ");
    for (int i = 0; i < num_heads * D; ++i) printf("%f ", h_wv[i]);
    printf("\n");

    printf("wo = ");
    for (int i = 0; i < D; ++i) printf("%f ", h_wo[i]);
    printf("\n");

    // Allocate and transfer to device memory
    float *d_q, *d_k, *d_v, *d_wq, *d_wk, *d_wv, *d_wo, *d_output;
    cudaMalloc(&d_q, T * D * sizeof(float));
    cudaMalloc(&d_k, T * D * sizeof(float));
    cudaMalloc(&d_v, T * D * sizeof(float));
    cudaMalloc(&d_wq, num_heads * D * sizeof(float));
    cudaMalloc(&d_wk, num_heads * D * sizeof(float));
    cudaMalloc(&d_wv, num_heads * D * sizeof(float));
    cudaMalloc(&d_wo, D * sizeof(float));
    cudaMalloc(&d_output, T * D * sizeof(float));

    cudaMemcpy(d_q, h_q, T * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, T * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, T * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wq, h_wq, num_heads * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wk, h_wk, num_heads * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wv, h_wv, num_heads * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wo, h_wo, D * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel configuration and launch
    dim3 block(32);
    dim3 grid((T * D + block.x - 1) / block.x);
    multi_head_attention<<<grid, block>>>(d_q, d_k, d_v, d_wq, d_wk, d_wv, d_wo, d_output, T, D);

    // Transfer back to host and print output
    float h_output[T * D];
    cudaMemcpy(h_output, d_output, T * D * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Multi-head Attention Output:\n");
    for (int i = 0; i < T; ++i) {
        for (int j = 0; j < D; ++j) {
            printf("%f ", h_output[i * D + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_wq);
    cudaFree(d_wk);
    cudaFree(d_wv);
    cudaFree(d_wo);
    cudaFree(d_output);

    return 0;
}

