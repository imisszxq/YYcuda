#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

__global__ void blockReduce_kernel(int output,
                                   const int *input,
                                   int count)
{
    __shared__ cuda::atomic<int, cuda::thread_scope_block> total_sum;
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    int thread_sum = 0;
    for (int i = block.thread_rank(); i < count; i += block.size())
    {
        thread_sum += input[i];
    }
    // reduce可以使用异步模式，使得block中的每个warp异步计算warp内的和，最后将每个warp的结果atomic累加到total_sum
    // cg::plus<>()cg中定义的加法function
    cg::reduce_update_async(tile, total_sum, thread_sum, cg::plus<int>());
    // 同步确保证block的所有tile完成累加，并将结果累加到total_sum
    block.sync();
    output=total_sum;
}

int main()
{
}
