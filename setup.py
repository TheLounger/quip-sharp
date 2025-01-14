import os
import platform
import subprocess
import torch

from glob import glob
from setuptools import setup, find_packages
from torch.utils import cpp_extension


def get_cuda_version(cuda_home=os.environ.get('CUDA_PATH', os.environ.get('CUDA_HOME', ''))):
    if cuda_home == '' or not os.path.exists(os.path.join(cuda_home, "bin", "nvcc.exe" if platform.system() == "Windows" else "nvcc")):
        return ''
    version_str = subprocess.check_output([os.path.join(cuda_home, "bin", "nvcc"), "--version"]).decode('utf-8')
    idx = version_str.find("release")
    return version_str[idx+len("release "):idx+len("release ")+4]


CUDA_VERSION = "".join(get_cuda_version().split(".")) if not os.environ.get('ROCM_VERSION', False) else False
ROCM_VERSION = os.environ.get('ROCM_VERSION', False) if torch.version.hip else False
PACKAGE_VERSION = "0.0.1" + (f"+cu{CUDA_VERSION}" if CUDA_VERSION else f"+rocm{ROCM_VERSION}" if ROCM_VERSION else "")

extra_compile_args = {
    'cxx': ['-lineinfo'],
    'nvcc': ['-O2', '-Xcompiler', '-rdynamic', '-lineinfo']
}

if torch.version.hip:
    extra_compile_args["nvcc"].append("-U__HIP_NO_HALF_CONVERSIONS__")

# Copied from https://github.com/PanQiWei/AutoGPTQ/blob/main/setup.py
if os.name == "nt":
    # On Windows, fix an error LNK2001: unresolved external symbol cublasHgemm bug in the compilation
    cuda_path = os.environ.get("CUDA_PATH", None)
    if cuda_path is None:
        raise ValueError("The environment variable CUDA_PATH must be set to the path to the CUDA install when installing from source on Windows systems.")
    extra_link_args = ["-L", f"{cuda_path}/lib/x64/cublas.lib"]
else:
    extra_link_args = []

quip_dir = os.path.join('vendor', 'quip-sharp')
setup(
    name="quipsharp",
    version=PACKAGE_VERSION,
    install_requires=[
        "torch",
    ],
    packages=find_packages(quip_dir),
    package_dir={'': quip_dir},
    data_files=[('model', glob(os.path.join(quip_dir, 'model', '*.py')))],
    ext_modules=[cpp_extension.CUDAExtension(
        'quiptools_cuda',
        [
            os.path.join(quip_dir, 'quiptools', 'quiptools_wrapper.cpp'),
            os.path.join(quip_dir, 'quiptools', 'quiptools.cu'),
            os.path.join(quip_dir, 'quiptools', 'quiptools_e8p_gemv.cu'),
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    )],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
