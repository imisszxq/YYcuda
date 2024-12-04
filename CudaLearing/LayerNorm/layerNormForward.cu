#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// layerNorm forward:
// input: batch*seq_len*hidden_dim
// 每个thread_block负责一个seq的forward

// 使用warp结合shared_memory实现block级别的reduce
__global__ void layerNorm_kernel_v0(float *__restrict__ output,
                                    float *__restrict__ mean,
                                    float *__restrict__ scale,
                                    const float *__restrict__ input,
                                    const float *__restrict__ weight,
                                    const float *__restrict__ bias,
                                    unsigned int n)
{
    unsigned int idx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    unsigned int warp_num = ceil(blockDim.x / 32);
    float sum = 0;
    float sum_x2 = 0;
    const float *in = input + idx * n;
    
    for (int i = tid; i < n; i += blockDim.x)
    {
        sum += in[i];
        sum_x2 += in[i] * in[i];
    }
    __shared__ float warp_sum[32];
    __shared__ float warp_sum_x2[32];
#pragma unroll
    for (int i = 16; i > 0; i /= 2)
    {
        sum += __shfl_down_sync(0xffffffff, sum, i);
    }
#pragma unroll
    for (int i = 16; i > 0; i /= 2)
    {
        sum_x2 += __shfl_down_sync(0xffffffff, sum_x2, i);
    }
    if (lane_id == 0)
    {
        warp_sum[warp_id] = sum;
        warp_sum_x2[warp_id] = sum_x2;
    }
    __syncthreads();

    sum = lane_id < warp_num ? warp_sum[lane_id] : 0.0f;
    sum_x2 = lane_id < warp_num ? warp_sum_x2[lane_id] : 0.0f;

#pragma unroll
    for (int i = 16; i > 0; i /= 2)
    {
        sum += __shfl_down_sync(0xffffffff, sum, i);
    }
#pragma unroll
    for (int i = 16; i > 0; i /= 2)
    {
        sum_x2 += __shfl_down_sync(0xffffffff, sum_x2, i);
    }
    __shfl_sync(0xffffffff, sum, 0);
    __shfl_sync(0xffffffff, sum_x2, 0);
    float m = sum / n;
    float var = sum_x2 / n - m * m;
    float s = sqrtf(var + 1e-5f);
    if (tid == 0)
    {
        mean[idx] = m;
        scale[idx] = s;
    }
    float *o = output + idx * n;
    for (int i = tid; i < n; i + blockDim.x)
    {
        o[i] = weight[i] * (in[i] - m) / s + bias[i];
    }
    
}

int main()
{
    return 0;
}