#include<cuda.h>
#include<cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

namespace cg=cooperative_groups;

__device__ int calculate_buffer_space_needed(cg::thread_block_tile<32>& tile) {
    return tile.thread_rank() % 2 + 1;
}

__device__ int my_thread_data(int i) {
    return i;
}


__global__ void buffer_kernel(){
    __shared__ extern int buffer[];
    //cuda:: thread_scope_block 表示buffer_used的作用范围是block
    __shared__ cuda::atomic<int,cuda::thread_scope_block> buffer_used;

    auto block=cg::this_thread_block();
    cg::thread_block_tile<32> tile= cg::tiled_partition<32>(block);
    buffer_used=0;
    block.sync();
    
    int buf_needed=calculate_buffer_space_needed(tile);
    //每个tile的所有线程needed的空间，scan计算完成后，将这个tile的结果updata到buffer_used，得到在shared_memory buffer中的存储位置
    int buf_offset=cg::exclusive_scan_update(tile,buffer_used,buf_needed);

    for(int i=0;i<buf_needed;i++){
        buffer[buf_offset+i]=my_thread_data(i);
    }
    //确保block中的所有tile完成buffer的缓存
    block.sync();

}