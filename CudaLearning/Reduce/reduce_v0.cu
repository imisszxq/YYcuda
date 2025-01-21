#include <cuda.h>
#include <cuda_runtime.h>

#include "error.cuh"

#define THREAD_PER_BLOCK 256

/* reduce_v0:通过shared_memory完成thread_block的求和，每个线程只读取一个数据
 */
__global__ void reduce_v0(float *d_out,
                          const float *d_in,
                          int n)
{
    __shared__ float d_shared[THREAD_PER_BLOCK];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    d_shared[tid] = d_in[i];
    __syncthreads();
    unsigned int temp;
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        temp = tid + s;
        if (temp < THREAD_PER_BLOCK && tid % (2 * s) == 0)
        {
            d_shared[tid] += d_shared[temp];
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
