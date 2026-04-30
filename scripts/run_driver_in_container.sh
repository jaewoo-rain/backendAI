#!/usr/bin/env bash
# Stage 5-C 검증: Driver API (cuMemAlloc_v2 / cuMemFree_v2) 후킹.
#
# 컨테이너 안에서 test_driver_alloc 을 두 번 실행:
#   (1) baseline       — hook 없이. 두 alloc (256 MiB, 6 GiB) 모두 GPU 가 받아주면 SUCCESS.
#                         RTX 4060/8 GB 면 256 MiB 는 통과, 6 GiB 는 시스템상
#                         OUT_OF_MEMORY — hook 없는 자연스러운 상한이라 정상.
#   (2) hooked         — LD_PRELOAD + FGPU_RATIO=0.4. 256 MiB ALLOW / 6 GiB DENY.
#                         로그 prefix 가 "[fgpu] ALLOW cuMemAlloc_v2 ..." 로
#                         나오면 driver hook 이 잡혔다는 증거.
#
# 사전 조건:
#   - scripts/build_hook.sh   → build/libfgpu.so   (Stage 5-C 변경 반영된 새 hook)
#   - scripts/build_image.sh  → fgpu-runtime:stage2 (test_driver_alloc 포함된 새 이미지)
#
# 사용법:
#   ./scripts/run_driver_in_container.sh
#   FGPU_RATIO=0.6 ./scripts/run_driver_in_container.sh
#   IMAGE=fgpu-runtime:stage2 ./scripts/run_driver_in_container.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-fgpu-runtime:stage2}"
HOOK_SO_HOST="${ROOT_DIR}/build/libfgpu.so"
RATIO="${FGPU_RATIO:-0.4}"

if [[ ! -f "${HOOK_SO_HOST}" ]]; then
    echo "ERROR: ${HOOK_SO_HOST} 가 없음. scripts/build_hook.sh 먼저." >&2
    exit 1
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "ERROR: 이미지 ${IMAGE} 없음. scripts/build_image.sh 먼저." >&2
    exit 1
fi
# test_driver_alloc 가 이미지 안에 있는지 — Dockerfile 갱신 후 재빌드 안 했으면 fail.
if ! docker run --rm --entrypoint /bin/sh "${IMAGE}" \
        -c '[ -x /opt/fgpu/test_driver_alloc ]' >/dev/null 2>&1; then
    echo "ERROR: ${IMAGE} 안에 /opt/fgpu/test_driver_alloc 가 없음." >&2
    echo "       Dockerfile 갱신 후 scripts/build_image.sh 로 재빌드 필요." >&2
    exit 1
fi

echo "============================================================"
echo "[driver-test] (1/2) baseline — hook 없이"
echo "============================================================"
docker run --rm --gpus all \
    --entrypoint /opt/fgpu/test_driver_alloc \
    "${IMAGE}"

echo
echo "============================================================"
echo "[driver-test] (2/2) hooked — FGPU_RATIO=${RATIO}"
echo "============================================================"
docker run --rm --gpus all \
    --entrypoint /opt/fgpu/test_driver_alloc \
    -v "${HOOK_SO_HOST}:/opt/fgpu/libfgpu.so:ro" \
    -e LD_PRELOAD=/opt/fgpu/libfgpu.so \
    -e FGPU_RATIO="${RATIO}" \
    "${IMAGE}"

echo
echo "[driver-test] done. 기대 결과:"
echo "  hooked 실행 stderr 에"
echo "    [fgpu] init: real cuMemAlloc_v2=0x..."
echo "    [fgpu] ALLOW cuMemAlloc_v2 ptr=0x... size=268435456 ..."
echo "    [fgpu] DENY  cuMemAlloc_v2 size=6442450944 used=... quota=..."
echo "  가 나오면 driver hook 통과."
