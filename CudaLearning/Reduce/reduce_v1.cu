#include <cuda.h>
#include <cuda_runtime.h>

#include "error.cuh"

#define THREAD_PER_BLOCK 256
/*  reduce_v1:每个warp中存在if-else语句，导致的warp divergent 问题
    尽量的使每一个warp中的threads走相同的if分支
*/
__global__ void reduce_V1(float *d_out,
                          const float *d_in)
{

    __shared__ float d_shared[THREAD_PER_BLOCK];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    d_shared[tid] = d_in[i];
    __syncthreads();
    unsigned int idx;
    for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2)
    {
        idx = 2 * s * tid;
        if (idx < THREAD_PER_BLOCK)
        {
            d_shared[idx] += d_shared[idx + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        d_out[blockIdx.x] = d_shared[tid];
    }
}

int main()
{
    return 0;
}