#!/usr/bin/env bash
# Build libfgpu.so. Run on the Ubuntu GPU server.
#
# Requires: gcc, /usr/local/cuda (driver + toolkit installed via nvidia-cuda-toolkit
# or runfile installer). Adjust CUDA_HOME if your install lives elsewhere.

set -euo pipefail

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${SRC_DIR}/build"

mkdir -p "${BUILD_DIR}"

gcc -O2 -fPIC -shared -Wall -Wextra \
    -I"${CUDA_HOME}/include" \
    -o "${BUILD_DIR}/libfgpu.so" \
    "${SRC_DIR}/hook/src/fgpu_hook.c" \
    -L"${CUDA_HOME}/lib64" -lcudart -ldl -lpthread

echo "[build] wrote ${BUILD_DIR}/libfgpu.so"
ls -lh "${BUILD_DIR}/libfgpu.so"
