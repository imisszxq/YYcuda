#include <cuda.h>
#include <cuda_runtime.h>

#include "error.cuh"

/*  reduce_V3:解决读取shared_memory时reduce_V1存在的bank conflict问题
*/
#define THREAD_PER_BLOCK 256
__global__ void reduce_V2(float *d_out,
                          const float *d_in)
{
    __shared__ float d_shared[THREAD_PER_BLOCK];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    d_shared[tid] = d_in[i];
    __syncthreads();
    for(unsigned int s=THREAD_PER_BLOCK/2;s>0;s/=2){
        if(tid<s){
            d_shared[tid]+=d_shared[tid+s];
        }
        __syncthreads();
    }
    if(tid==0){
        d_out[blockIdx.x]=d_shared[tid];
    }
}

int main()
{
    return 0;
}