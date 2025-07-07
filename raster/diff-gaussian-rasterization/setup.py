from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp"
            ],
            extra_compile_args={
                "nvcc": [
                    "-I" + os.path.join(this_dir, "third_party/glm/"),
                    "-gencode=arch=compute_86,code=sm_86",   # ← 新增
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
