#!/usr/bin/env bash
# Stage 4 검증: PyTorch 컨테이너 안에서 hook quota 가 텐서 할당까지 차단하는지.
#
# 사전 조건:
#   - scripts/build_hook.sh         → build/libfgpu.so
#   - scripts/build_image.sh        → fgpu-runtime:stage2
#   - scripts/build_pytorch_image.sh → fgpu-runtime-pytorch:stage4
#
# 기본 시나리오 (256 MiB + 4 GiB on RTX 4060/8GB):
#   baseline:    256 MiB OK,  4 GiB OK
#   ratio=0.4:   256 MiB OK,  4 GiB OOM   (quota 3.2 GiB)
#   ratio=0.6:   256 MiB OK,  4 GiB OK    (quota 4.8 GiB)
#
# 사용법:
#   ./scripts/run_pytorch_in_container.sh
#   FGPU_RATIO=0.6 ./scripts/run_pytorch_in_container.sh
#   PYTEST_ALLOC2_MIB=6144 ./scripts/run_pytorch_in_container.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-fgpu-runtime-pytorch:stage4}"
HOOK_SO_HOST="${ROOT_DIR}/build/libfgpu.so"
RATIO="${FGPU_RATIO:-0.4}"
ALLOC1_MIB="${PYTEST_ALLOC1_MIB:-256}"
ALLOC2_MIB="${PYTEST_ALLOC2_MIB:-4096}"

if [[ ! -f "${HOOK_SO_HOST}" ]]; then
    echo "ERROR: ${HOOK_SO_HOST} 가 없음. scripts/build_hook.sh 먼저." >&2
    exit 1
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "ERROR: 이미지 ${IMAGE} 가 없음. scripts/build_pytorch_image.sh 먼저." >&2
    exit 1
fi

echo "============================================================"
echo "[pytorch] (1/2) baseline — hook 없이"
echo "          alloc1=${ALLOC1_MIB} MiB, alloc2=${ALLOC2_MIB} MiB"
echo "============================================================"
docker run --rm --gpus all \
    -e PYTEST_ALLOC1_MIB="${ALLOC1_MIB}" \
    -e PYTEST_ALLOC2_MIB="${ALLOC2_MIB}" \
    "${IMAGE}"

echo
echo "============================================================"
echo "[pytorch] (2/2) hooked — FGPU_RATIO=${RATIO} (caching off)"
echo "          alloc1=${ALLOC1_MIB} MiB, alloc2=${ALLOC2_MIB} MiB"
echo "============================================================"
docker run --rm --gpus all \
    -v "${HOOK_SO_HOST}:/opt/fgpu/libfgpu.so:ro" \
    -e LD_PRELOAD=/opt/fgpu/libfgpu.so \
    -e FGPU_RATIO="${RATIO}" \
    -e PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
    -e PYTEST_ALLOC1_MIB="${ALLOC1_MIB}" \
    -e PYTEST_ALLOC2_MIB="${ALLOC2_MIB}" \
    "${IMAGE}"
