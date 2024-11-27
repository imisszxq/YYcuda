#include <cuda.h>
#include <cuda_runtime.h>

#include "error.cuh"

#define BLOCK_SIZE 256
#define WARP_SIZE 32
template <unsigned int blockSize>
__device__ __forceinline__ float WarpReduceSum(float sum)
{
    if (blockSize >= 32)
        sum += __shfl_down_sync(0xffffffff, sum, 16);
    if (blockSize >= 16)
        sum += __shfl_down_sync(0xffffffff, sum, 8);
    if (blockSize >= 8)
        sum += __shfl_down_sync(0xffffffff, sum, 4);
    if (blockSize >= 4)
        sum += __shfl_down_sync(0xffffffff, sum, 2);
    if (blockSize >= 2)
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}

#pragma unroll
__device__ float WarpReduceSumV2(float sum, unsigned int SumSize)
{
    for (unsigned int i = SumSize; i >= 1; i /= 2)
    {
        sum += __shfl_down_sync(0xffffffff, sum, i);
    }
    return sum;
}

__global__ void reduce_v3(float *d_out,
                          const float *d_in,
                          unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int grid_size = gridDim.x * blockDim.x;
    float sum = 0.0f;

    while (i < n)
    {
        sum += d_in[i];
        i += grid_size;
    }
    sum = WarpReduceSum<BLOCK_SIZE>(sum);

    // 将每个warp的结果写入shared_memory
    __shared__ float d_shared[WARP_SIZE];
    const int laneId = tid % WARP_SIZE;
    const int warpId = tid / WARP_SIZE;
    if (laneId == 0)
    {
        d_shared[warpId] = sum;
    }
    __syncthreads();
    sum = (tid < BLOCK_SIZE / WARP_SIZE) ? d_shared[laneId] : 0.0f;
    if (warpId == 0)
    {
        sum = WarpReduceSum<WARP_SIZE>(sum);
    }
    if (tid == 0)
    {
        d_out[blockIdx.x] = sum;
    }
}

int main()
{
    return 0;
}