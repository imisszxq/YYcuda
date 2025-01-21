#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
using namespace cooperative_groups;

__global__ void cooperative_groups_kernel()
{
    thread_block g = this_thread_block();
    thread_group tile32 = cooperative_groups::tiled_partition(g, 32);
    thread_group tile4 = tiled_partition(g, 4);
    // 0-3线程进入这个kernel后，运行到tile4.sync(),只会控制线程0-3进行同步
    // 4-7线程进入这个kernel后，运行到tile4.sync(),只会控制线程4-7进行同步
    //...
    tile4.sync();
    // 静态分组调用方法
    thread_block_tile<32> s_tile32 = tiled_partition<32>(this_thread_block());

    // 根据条件对warp中的数据进行分组
    // example：奇数线程的分为一组
    if (g.thread_rank() % 2 == 1)
    {
        coalesced_group active = coalesced_threads();
        /*奇数线程的计算
         */
        active.sync();
        // 定义active组中第一个活动线程为thread_rank=0,此例中第一个奇数线程threadIdx.x==1的线程active.thread_rank是0
        if (active.thread_rank() == 0)
        {
        }
    }
    cuda::atomic<int, cuda::thread_scope_block> a;


}

int main()
{
    return 0;
}