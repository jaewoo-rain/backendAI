#!/usr/bin/env bash
# Stage 2 검증: 컨테이너 안에서 host-built libfgpu.so 로 hook 동작 확인.
#
# 사전 조건:
#   1) scripts/build_hook.sh 가 build/libfgpu.so 를 만들어 둠.
#   2) scripts/build_image.sh 가 fgpu-runtime:stage2 를 만들어 둠.
#   3) nvidia-container-toolkit 설치됨 (--gpus all 동작).
#
# 사용법:
#   ./scripts/run_in_container.sh                    # ratio 0.4 기본
#   FGPU_RATIO=0.6 ./scripts/run_in_container.sh
#   FGPU_QUOTA_BYTES=$((512*1024*1024)) ./scripts/run_in_container.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-fgpu-runtime}"
IMAGE_TAG="${IMAGE_TAG:-stage2}"
HOOK_SO_HOST="${ROOT_DIR}/build/libfgpu.so"
RATIO="${FGPU_RATIO:-0.4}"

if [[ ! -f "${HOOK_SO_HOST}" ]]; then
    echo "ERROR: ${HOOK_SO_HOST} 가 없음. scripts/build_hook.sh 먼저 실행." >&2
    exit 1
fi

if ! docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" >/dev/null 2>&1; then
    echo "ERROR: 이미지 ${IMAGE_NAME}:${IMAGE_TAG} 가 없음. scripts/build_image.sh 먼저 실행." >&2
    exit 1
fi

echo "============================================================"
echo "[run] (1/2) baseline — 컨테이너 안, hook 없이"
echo "============================================================"
docker run --rm --gpus all \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    /opt/fgpu/test_alloc

echo
echo "============================================================"
echo "[run] (2/2) hooked  — 컨테이너 안, FGPU_RATIO=${RATIO}"
echo "============================================================"
docker run --rm --gpus all \
    -v "${HOOK_SO_HOST}:/opt/fgpu/libfgpu.so:ro" \
    -e LD_PRELOAD=/opt/fgpu/libfgpu.so \
    -e FGPU_RATIO="${RATIO}" \
    ${FGPU_QUOTA_BYTES:+-e FGPU_QUOTA_BYTES="${FGPU_QUOTA_BYTES}"} \
    "${IMAGE_NAME}:${IMAGE_TAG}"
