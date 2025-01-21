#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cooperative_groups/reduce.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

template <typename Group, typename Data, typename Fn>
__device__ int stream_compaction(Group &g, Data *input, int count, Fn &&test_fn, Data *output)
{
    int per_thread = count / g.num_threads();
    int thread_start = min(g.thread_rank() * per_thread, count);
    int my_count = min(per_thread, count - thread_start);

    int i = thread_start;
    // 确定group组中每个线程的input数据，符合test_fn条件的数据数目
    while (i < thread_start + my_count)
    {
        if (test_fn(input[i]))
        {
            i++;
        }
        else
        {
            my_count--; // my_count记录了最终符合test_fn的数据的数目
            input[i] = input[my_count + thread_start];
        }
    }
    // 使用exclusive_scan方法获取每个线程符合test_fn的数据在存储output的起始存储位置
    // exclusive_scan返回没个线程的前缀和(不包括自身)
    int my_index = cg::exclusive_scan(g, my_count);

    for (i = 0; i < my_count; i++)
    {
        output[my_index + i] = input[thread_start + i];
    }
    //返回符合要求的数据的总数
    //cooperative_groups最后一个线程的offset+my_count
    //通过shfl 广播到每一个线程
    return g.shfl(my_index+my_count,g.num_threads()-1);
}