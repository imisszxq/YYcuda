#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

// 基于cg实现global->shared的异步数据拷贝
__global__ void memcpyAsync_kernel(const int *global_data)
{
    auto block=cg::this_thread_block();
    const size_t elementsPerThreadBlock = 16 * 1024 + 64;
    const size_t elementsInShared = 128;
    __align__(16) __shared__ int local_smem[2][elementsInShared];
    int stage = 0;
    size_t copy_count=elementsInShared;
    size_t index=copy_count;
    cg::memcpy_async(block,local_smem[stage],elementsInShared,global_data,elementsPerThreadBlock);
    
    while(index<elementsPerThreadBlock){
        cg::memcpy_async(block,local_smem[stage^1],elementsInShared,global_data+index,elementsPerThreadBlock-index);
        cg::wait_prior<1>(block);

        copy_count=min(copy_count,elementsPerThreadBlock-index);
        index+=copy_count;
        stage^=1;
    }
    cg::wait(block);


}