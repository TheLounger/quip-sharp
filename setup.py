from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension
import platform
import os
import subprocess
import torch

def get_cuda_version(cuda_home=os.environ.get('CUDA_PATH', os.environ.get('CUDA_HOME', ''))):
    if cuda_home == '' or not os.path.exists(os.path.join(cuda_home,"bin","nvcc.exe" if platform.system() == "Windows" else "nvcc")):
        return ''
    version_str = subprocess.check_output([os.path.join(cuda_home,"bin","nvcc"),"--version"]).decode('utf-8')
    idx = version_str.find("release")
    return version_str[idx+len("release "):idx+len("release ")+4]
    
CUDA_VERSION = "".join(get_cuda_version().split(".")) if not os.environ.get('ROCM_VERSION', False) else False
ROCM_VERSION = os.environ.get('ROCM_VERSION', False) if torch.version.hip else False

extra_compile_args = {
    'cxx': ['-lineinfo'],
    'nvcc': ['-O2', '-Xcompiler', '-rdynamic', '-lineinfo']
}

if torch.version.hip:
    extra_compile_args["nvcc"].append("-U__HIP_NO_HALF_CONVERSIONS__")

version = "0.0.1" + (f"+cu{CUDA_VERSION}" if CUDA_VERSION else f"+rocm{ROCM_VERSION}" if ROCM_VERSION else "")
setup(
    name="quipsharp",
    version=version,
    install_requires=[
        "torch",
    ],
    packages=find_packages(),
    ext_modules=[cpp_extension.CUDAExtension(
        'quiptools_cuda',
        [
            'quiptools/quiptools_wrapper.cpp',
            'quiptools/quiptools.cu',
            'quiptools/quiptools_e8p_gemv.cu'
        ],
        extra_compile_args=extra_compile_args,
        libraries=["cublas"] if platform.system() == "Windows" else [],
    )],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
