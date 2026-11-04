"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import subprocess
from packaging.version import parse, Version
from typing import List, Set
import warnings

from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

# Monkey-patch: отключаем проверку CUDA version mismatch
torch.utils.cpp_extension._check_cuda_version = lambda *a, **kw: None

# Common compiler flags.
COMMON_CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++20"]
COMMON_NVCC_FLAGS = [
    "-O3",
    "-std=c++20",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--use_fast_math",
    "--threads=8",
    "-Xptxas=-v",
    "-diag-suppress=174",
]

if CUDA_HOME is None:
    raise RuntimeError(
        "Cannot find CUDA_HOME. CUDA must be available to build the package.")

def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    """Get the CUDA version from nvcc."""
    nvcc_path = os.path.join(cuda_dir, "bin", "nvcc")
    if not os.path.exists(nvcc_path):
         nvcc_path = os.path.join(cuda_dir, "bin", "nvcc.exe")
    if not os.path.exists(nvcc_path):
        raise RuntimeError(f"nvcc not found at {nvcc_path}")

    try:
        nvcc_output = subprocess.check_output([nvcc_path, "-V"],
                                              universal_newlines=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to run nvcc -V: {e.output}") from e

    output = nvcc_output.split()
    release_idx = -1
    for i, s in enumerate(output):
        if "release" in s:
            release_idx = i + 1
            break
    if release_idx == -1:
        raise RuntimeError("Could not parse nvcc -V output for CUDA version")

    version_str = output[release_idx].split(",")[0]
    nvcc_cuda_version = parse(version_str)
    return nvcc_cuda_version

# Get CUDA version
nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
print(f"Detected CUDA version: {nvcc_cuda_version}")

MIN_CUDA_VERSION = Version("11.8")
if nvcc_cuda_version < MIN_CUDA_VERSION:
    raise RuntimeError(f"CUDA {MIN_CUDA_VERSION} or higher is required (detected {nvcc_cuda_version}).")

# Determine target architecture (SM75 only)
compute_capabilities = set()
device_count = torch.cuda.device_count()

if device_count > 0:
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        cc = f"{major}.{minor}"
        if major < 7 or (major == 7 and minor < 5):
            warnings.warn(f"Skipping GPU {i} with compute capability {cc}. Requires SM75 (Turing) or higher.")
            continue
        compute_capabilities.add(cc)
else:
    warnings.warn("No CUDA GPUs detected. Building for SM75 (Turing T4) by default.")
    arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "7.5")
    compute_capabilities.update(arch.strip() for arch in arch_list_env.split(";") if arch.strip())

if not compute_capabilities:
     raise RuntimeError("No suitable compute capabilities found. Need SM75+ and CUDA environment.")

print(f"Building for compute capabilities: {compute_capabilities}")

# Build architecture flags (SM75 only)
arch_flags = []
for capability in sorted(list(compute_capabilities)):
    major, minor = map(int, capability.split('.'))
    if major == 7 and minor == 5:
        arch_flags.append("-gencode")
        arch_flags.append(f"arch=compute_75,code=sm_75")
    else:
        warnings.warn(f"Unsupported architecture for this fork: {capability}. Skipping.")

if not arch_flags:
    raise RuntimeError("No SM75 (Turing) architecture found. This fork only supports SM75.")

# Common flags setup
CXX_FLAGS = COMMON_CXX_FLAGS[:]
NVCC_FLAGS = COMMON_NVCC_FLAGS[:]

# SM75 (Turing) does not support BF16
print("BF16 support disabled (Turing SM75).")

# Add ABI flag
ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

# Append architecture-specific flags
NVCC_FLAGS.extend(arch_flags)

# --- Define Extensions ---
ext_modules = []

# SM75 Kernel (INT8 QK, FP16 PV, FP32 Accum)
qattn_sm75_extension = CUDAExtension(
    name="sageattention._qattn_sm75",
    sources=[
        "csrc/qattn/pybind_sm75.cpp",
        "csrc/qattn/qk_int_sv_f16_cuda_sm75.cu",
    ],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(qattn_sm75_extension)

# Fused kernels (Quantization, always built)
fused_extension = CUDAExtension(
    name="sageattention._fused",
    sources=["csrc/fused/pybind.cpp", "csrc/fused/fused.cu"],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(fused_extension)

# --- Setup Call ---
setup(
    name='sageattention',
    version='2.1.1',
    author='THEANGELAI',
    description='⚡ SageAttention-SM75 — INT8 QK + FP16 PV attention for NVIDIA T4 (Turing)',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/THE-ANGEL-AI/SageAttention-SM75-path',
    packages=find_packages(),
    python_requires='>=3.9',
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=[],
)
