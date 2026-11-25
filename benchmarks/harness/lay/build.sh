#!/bin/bash
# Builds the spmspv-adaptive SpMSpV benchmark in <workdir> without editing
# spmspv-adaptive. Needs nvcc on PATH.
#
#   benchmarks/harness/lay/build.sh <workdir>
#
# The executable is <workdir>/lay_spmspv.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mkdir -p "$1" && cd "$1" && pwd)"
LAY_COMMIT=b71de128ef68b56353e19a87104827e13c7ab895
# The ModernGPU commit that spmspv-adaptive's CMake files fetch
MODERNGPU_COMMIT=2528789dd855d44d400f27cfb5a5825942f6b563

cd "$WORK"
if [ ! -d spmspv-adaptive ]; then
    git clone https://github.com/hpc-research-2020/spmspv-adaptive.git
    git -C spmspv-adaptive checkout "$LAY_COMMIT"
fi
if [ ! -d moderngpu ]; then
    git clone https://github.com/ctcyang/moderngpu.git
    git -C moderngpu checkout "$MODERNGPU_COMMIT"
fi

# The GPU's architecture, e.g. sm_120
ARCH="sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader |
    head -1 | tr -d '.')"

# -fpermissive: utils_kernel.h shifts a negative constant in an enum
nvcc -arch="$ARCH" -O3 -w -std=c++17 -Xcompiler -fpermissive \
    -DVALUE_TYPE=float \
    -include "$HERE/shims/lay_compat.h" \
    -include "$HERE/shims/lay_malloc_fix.h" \
    -o "$WORK/lay_spmspv" "$HERE/lay_spmspv.cu" \
    -I"$WORK/spmspv-adaptive/hice-spmspv/hice/la/include" \
    -I"$WORK/moderngpu/include" -I"$HERE/shims"
echo "Built $WORK/lay_spmspv"
