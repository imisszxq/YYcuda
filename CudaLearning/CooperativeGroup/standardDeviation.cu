#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

using namespace cooperative_groups;

namespace cg = cooperative_groups;

// 求取当前tile的数据的标准差
__device__ float standard_deviation(const int *input,
                                    int length,
                                    cg::thread_block_tile<32> &tile)
{
    int thread_sum = 0;
    for (int i = tile.thread_rank(); i < length; i += tile.num_threads())
    {
        thread_sum += input[i];
    }
    float avg = cg::reduce(tile, thread_sum, cg::plus<int>()) / length;
    int thread_diff = 0;
    for (int i = tile.thread_rank(); i < length; i += tile.num_threads())
    {
        int diff = input[i] - avg;
        thread_diff += (diff * diff);
    }
    float diff_sum=cg::reduce(tile,thread_diff,cg::plus<int>())/length;
    return sqrtf(diff_sum);
}
