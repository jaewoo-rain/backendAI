#!/usr/bin/env bash
# Stage 6 검증: VMM API hook (cuMemCreate / cuMemRelease).
#
# 컨테이너 안에서 test_vmm_alloc 을 두 번 실행:
#   (1) baseline       — hook 없이. [test-vmm] 라인만, [fgpu] 라인 없음.
#                         256 MiB 는 통과, 6 GiB 는 hardware 한계로 실패할
#                         수 있음 — 어느 쪽이든 정상 (no hook = no quota).
#   (2) hooked         — LD_PRELOAD + FGPU_RATIO=0.4. 256 MiB ALLOW /
#                         6 GiB DENY. stderr 의 "[fgpu] ALLOW cuMemCreate"
#                         + "[fgpu] DENY cuMemCreate" 가 결정적 증거.
#
# 사전 조건:
#   - scripts/build_hook.sh   → libfgpu.so (Stage 6 hook 포함된 새 .so)
#   - scripts/build_image.sh  → fgpu-runtime:stage2 (test_vmm_alloc 포함된
#                                새 이미지)
#
# 사용법:
#   ./scripts/run_vmm_in_container.sh
#   FGPU_RATIO=0.6 ./scripts/run_vmm_in_container.sh

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
if ! docker run --rm --entrypoint /bin/sh "${IMAGE}" \
        -c '[ -x /opt/fgpu/test_vmm_alloc ]' >/dev/null 2>&1; then
    echo "ERROR: ${IMAGE} 안에 /opt/fgpu/test_vmm_alloc 가 없음." >&2
    echo "       Dockerfile 갱신 후 scripts/build_image.sh 로 재빌드 필요." >&2
    exit 1
fi

echo "============================================================"
echo "[vmm-test] (1/2) baseline — hook 없이"
echo "============================================================"
docker run --rm --gpus all \
    --entrypoint /opt/fgpu/test_vmm_alloc \
    "${IMAGE}"

echo
echo "============================================================"
echo "[vmm-test] (2/2) hooked — FGPU_RATIO=${RATIO}"
echo "============================================================"
docker run --rm --gpus all \
    --entrypoint /opt/fgpu/test_vmm_alloc \
    -v "${HOOK_SO_HOST}:/opt/fgpu/libfgpu.so:ro" \
    -e LD_PRELOAD=/opt/fgpu/libfgpu.so \
    -e FGPU_RATIO="${RATIO}" \
    "${IMAGE}"

echo
echo "[vmm-test] done. 기대 결과:"
echo "  hooked stderr 에"
echo "    [fgpu] init: ... cuMemCreate=0x... cuMemRelease=0x..."
echo "    [fgpu] ALLOW cuMemCreate handle=0x... size=268435456 ..."
echo "    [fgpu] DENY  cuMemCreate size=6442450944 used=... quota=..."
echo "  + stdout 의 6 GiB 시도 result 가 2 (CUDA_ERROR_OUT_OF_MEMORY)"
