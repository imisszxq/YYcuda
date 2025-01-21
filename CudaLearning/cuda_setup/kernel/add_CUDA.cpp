#include <torch/extention.h>
#include "add.h"

void torch_launch_add(torch::Tensor &c,
                      const torch::Tensor &a,
                      const torch::Tensor &b,
                      int n)
{
    launch_add(reinterpret_cast<float *>(c.data_ptr()),
               reinterpret_cast<float *>(a.data_ptr()),
               reinterpret_cast<float *>(b.data_ptr()),
               n);
}

PYBIND11_MODULE(TORCH_EXTENTION_NAME,m){
    m.def("torch_launch_add",
        &torch_launch_add,
        "add kernel warpper");
}



