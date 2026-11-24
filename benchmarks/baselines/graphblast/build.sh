#!/bin/bash
# Builds the GraphBLAST SpMSpV benchmark in <workdir> without editing
# GraphBLAST. Needs nvcc on PATH.
#
#   benchmarks/baselines/graphblast/build.sh <workdir>
#
# The executable is <workdir>/grb_spmspv.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mkdir -p "$1" && cd "$1" && pwd)"
GRAPHBLAST_COMMIT=f56f967
BOOST=boost_1_82_0

cd "$WORK"
if [ ! -d graphblast ]; then
    git clone --recursive https://github.com/gunrock/graphblast.git
    git -C graphblast checkout "$GRAPHBLAST_COMMIT"
    git -C graphblast submodule update --init --recursive
fi

# GraphBLAST needs Boost.Program_options
if [ ! -f "$BOOST/stage/lib/libboost_program_options.a" ]; then
    curl -L -o "$BOOST.tar.gz" \
        "https://archives.boost.io/release/1.82.0/source/$BOOST.tar.gz"
    tar -xzf "$BOOST.tar.gz"
    (cd "$BOOST" && ./bootstrap.sh --with-libraries=program_options &&
        ./b2 link=static -j 16)
fi

# The GPU's architecture, e.g. sm_120
ARCH="sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader |
    head -1 | tr -d '.')"

cd graphblast
nvcc -arch="$ARCH" -O3 -use_fast_math -w -std=c++17 \
    -include "$HERE/shims/grb_compat.h" \
    -o "$WORK/grb_spmspv" "$HERE/grb_spmspv.cu" \
    -Iext/moderngpu/include/ -I"$HERE/shims" -I"$WORK/$BOOST" -I./ \
    ext/moderngpu/src/mgpucontext.cu ext/moderngpu/src/mgpuutil.cpp \
    -L"$WORK/$BOOST/stage/lib" -lboost_program_options \
    -lcublas -lcusparse -lcurand
echo "Built $WORK/grb_spmspv"
