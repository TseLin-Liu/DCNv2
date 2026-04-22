#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="Conv2d",
    packages=['Conv2d'],
    ext_modules=[
        CUDAExtension(
            name="Conv2d._C",
            sources=[
            "forward.cu",
            "backward.cu",
            "ext.cpp"],
            # include_dirs=[cutlass_include],  
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
