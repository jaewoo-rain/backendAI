#!/usr/bin/env bash
# Build the test program with nvcc and run it under LD_PRELOAD.
# Run on the Ubuntu GPU server, AFTER scripts/build_hook.sh has succeeded.
#
# Usage:
#   ./scripts/run_test.sh           # default ratio 0.4
#   FGPU_RATIO=0.6 ./scripts/run_test.sh
#   FGPU_QUOTA_BYTES=$((512*1024*1024)) ./scripts/run_test.sh

set -euo pipefail

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${SRC_DIR}/build"
HOOK_SO="${BUILD_DIR}/libfgpu.so"
TEST_BIN="${BUILD_DIR}/test_alloc"

if [[ ! -f "${HOOK_SO}" ]]; then
    echo "ERROR: ${HOOK_SO} not found. Run scripts/build_hook.sh first." >&2
    exit 1
fi

mkdir -p "${BUILD_DIR}"
"${CUDA_HOME}/bin/nvcc" -o "${TEST_BIN}" "${SRC_DIR}/hook/tests/test_alloc.cu"

echo "[run] sanity-check WITHOUT hook (baseline):"
"${TEST_BIN}" || true
echo

echo "[run] WITH hook, FGPU_RATIO=${FGPU_RATIO:-0.4}"
LD_PRELOAD="${HOOK_SO}" \
FGPU_RATIO="${FGPU_RATIO:-0.4}" \
${FGPU_QUOTA_BYTES:+FGPU_QUOTA_BYTES=${FGPU_QUOTA_BYTES}} \
    "${TEST_BIN}"
