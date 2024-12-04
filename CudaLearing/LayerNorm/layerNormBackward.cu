#include <stdio.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

/*
layerNorm的反向传播
kernel负责计算input的gradient，weight和bias的gradient
每个thread_block中的一个warp负责一个seq的n维dim的结果计算
*/

namespace cg = cooperative_groups;

__global__ void layerNorm_backward_kernel(float *d_input,
                                          float *d_weight,
                                          float *d_bias,
                                          const float *d_output,
                                          const float *mean,
                                          const float *variance,
                                          const float *input,
                                          const float *weight,
                                          unsigned int n)
{
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % 32;
    unsigned int warp_id = tid / 32;
    unsigned int warp_num = blockDim.x / 32;

    unsigned int coord = (bid * warp_num + warp_id) * n;
    const float *d_output_warp = d_output + coord;
    float *d_input_warp = d_input + coord;
    const float *input_warp = input + coord;
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    __shared__ extern float shared[];
    float *shared_d_weights = shared;
    float *shared_d_bias = shared + n;

    for (unsigned int i = tid; i < n; i += blockDim.x)
    {
        shared_d_weights[i] = 0.0f;
        shared_d_bias[i] = 0.0f;
    }
    __syncthreads();

    float sum_d_output_w = 0.0f;
    float sum_d_output_w_meanX = 0.0f;

    float mean_warp = mean[coord];
    float variance_warp = variance[coord];

    // 计算input的梯度时，需要整个hidden_dim的求和信息
    for (unsigned int i = lane_id; i < n; i += 32)
    {
        float d_output_w = d_output_warp[i] * weight[i];
        sum_d_output_w += d_output_w;
        sum_d_output_w_meanX += d_output_w * (input_warp[i] - mean_warp) / variance_warp;
    }
    sum_d_output_w = cg::reduce(warp, sum_d_output_w, cg::plus<float>()) / n;
    sum_d_output_w_meanX = cg::reduce(warp, sum_d_output_w_meanX, cg::plus<float>()) / n;

    for (unsigned int i = lane_id; i < n; i += 32)
    {
        float input_std = (input_warp[i] - mean_warp) / variance_warp;
        float d_output_warp_i = d_output_warp[i];
        atomicAdd(&shared_d_weights[i], d_output_warp_i * input_std);
        atomicAdd(&shared_d_bias[i], d_output_warp_i);

        float input_gradient = 0.0f;
        input_gradient = d_output_warp_i * weight[i];
        input_gradient -= (sum_d_output_w + input_std * sum_d_output_w_meanX);
        input_gradient /= variance_warp;
        d_input_warp[i] = input_gradient;
    }
    for (unsigned int i = tid; i < n; i += blockDim.x)
    {
        atomicAdd(&d_weight[i],shared_d_weights[i]);
        atomicAdd(&d_bias[i],shared_d_bias[i]);
    }
}

int main()
{
    return 0;
}