#!/bin/bash
# Builds the benchmark of our SpMSpV variants in <workdir>. Needs nvcc on
# PATH.
#
#   benchmarks/harness/ours/build.sh <workdir>
#
# The executable is <workdir>/ours_spmspv.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mkdir -p "$1" && cd "$1" && pwd)"
SRC="$HERE/../../../src/cuda"
# The CCCL commit that the extension is built with
CCCL_COMMIT="$(sed -n 's/^set(CCCL_COMMIT \(.*\))$/\1/p' "$SRC/CMakeLists.txt")"
CCCL_SHA256="$(sed -n 's/.*URL_HASH SHA256=\(.*\)$/\1/p' "$SRC/CMakeLists.txt")"

cd "$WORK"
if [ ! -d "cccl-$CCCL_COMMIT" ]; then
    curl -L -o cccl.tar.gz \
        "https://github.com/NVIDIA/cccl/archive/$CCCL_COMMIT.tar.gz"
    echo "$CCCL_SHA256  cccl.tar.gz" | sha256sum -c
    tar -xzf cccl.tar.gz
fi
CCCL="$WORK/cccl-$CCCL_COMMIT"

# The GPU's architecture, e.g. sm_120
ARCH="sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader |
    head -1 | tr -d '.')"

nvcc -arch="$ARCH" -O3 -DNDEBUG -w -std=c++17 \
    -o "$WORK/ours_spmspv" "$HERE/ours_spmspv.cu" \
    "$SRC/kernels/pool.cu" "$SRC/kernels/utils.cu" \
    "$SRC/kernels/spmspv/dense.cu" "$SRC/kernels/spmspv/esc.cu" \
    "$SRC/kernels/spmspv/hash.cu" "$SRC/kernels/spmspv/partition.cu" \
    "$SRC/kernels/spmspv/cusparse.cpp" \
    -I"$CCCL/cub" -I"$CCCL/libcudacxx/include" -I"$CCCL/thrust" \
    -I"$SRC" -I"$SRC/kernels/spmspv" -lcusparse
echo "Built $WORK/ours_spmspv"
