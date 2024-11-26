from setuptools import setup
from torch.utils.cpp_extention import BuildExtension,CUDAExtension

setup(
    name="add",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension("add",
        ["kernel/add_CUDA.cpp","kernel/add_CUDA.cu"])
    ],
    cmdclass={
        "build_ext":BuildExtension
    }

)


