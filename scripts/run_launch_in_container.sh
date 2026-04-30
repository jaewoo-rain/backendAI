#!/usr/bin/env bash
# Stage 7 검증: cudaLaunchKernel 후킹 — launch counter 만 (quota 시행 X).
#
# 컨테이너 안에서 test_launch 를 두 번 실행:
#   (1) baseline       — hook 없이. [test-launch] 라인만, [fgpu] 라인 없음.
#   (2) hooked         — LD_PRELOAD + FGPU_LAUNCH_LOG_EVERY=100. 1000 launch
#                         이므로 stderr 에 "[fgpu] LAUNCH count=100/200/.../1000"
#                         10 줄 + atexit "[fgpu] exit summary: total ... = 1000".
#
# 사전 조건:
#   - scripts/build_hook.sh   → build/libfgpu.so   (Stage 7 변경 반영된 새 hook)
#   - scripts/build_image.sh  → fgpu-runtime:stage2 (test_launch 포함된 새 이미지)
#
# 사용법:
#   ./scripts/run_launch_in_container.sh
#   PYTEST_LAUNCH_N=5000 FGPU_LAUNCH_LOG_EVERY=500 ./scripts/run_launch_in_container.sh
#   FGPU_LAUNCH_LOG_EVERY=0 ./scripts/run_launch_in_container.sh   # log off (overhead 측정)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-fgpu-runtime:stage2}"
HOOK_SO_HOST="${ROOT_DIR}/build/libfgpu.so"
LAUNCH_N="${PYTEST_LAUNCH_N:-1000}"
LAUNCH_LOG_EVERY="${FGPU_LAUNCH_LOG_EVERY:-100}"

if [[ ! -f "${HOOK_SO_HOST}" ]]; then
    echo "ERROR: ${HOOK_SO_HOST} 가 없음. scripts/build_hook.sh 먼저." >&2
    exit 1
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "ERROR: 이미지 ${IMAGE} 없음. scripts/build_image.sh 먼저." >&2
    exit 1
fi
if ! docker run --rm --entrypoint /bin/sh "${IMAGE}" \
        -c '[ -x /opt/fgpu/test_launch ]' >/dev/null 2>&1; then
    echo "ERROR: ${IMAGE} 안에 /opt/fgpu/test_launch 가 없음." >&2
    echo "       Dockerfile 갱신 후 scripts/build_image.sh 로 재빌드 필요." >&2
    exit 1
fi

echo "============================================================"
echo "[launch-test] (1/2) baseline — hook 없이  (n=${LAUNCH_N})"
echo "============================================================"
docker run --rm --gpus all \
    --entrypoint /opt/fgpu/test_launch \
    -e PYTEST_LAUNCH_N="${LAUNCH_N}" \
    "${IMAGE}"

echo
echo "============================================================"
echo "[launch-test] (2/2) hooked — log every ${LAUNCH_LOG_EVERY}, n=${LAUNCH_N}"
echo "============================================================"
docker run --rm --gpus all \
    --entrypoint /opt/fgpu/test_launch \
    -v "${HOOK_SO_HOST}:/opt/fgpu/libfgpu.so:ro" \
    -e LD_PRELOAD=/opt/fgpu/libfgpu.so \
    -e FGPU_LAUNCH_LOG_EVERY="${LAUNCH_LOG_EVERY}" \
    -e PYTEST_LAUNCH_N="${LAUNCH_N}" \
    "${IMAGE}"

echo
echo "[launch-test] done. 기대 결과:"
echo "  hooked stderr 에"
echo "    [fgpu] init: ... cudaLaunchKernel=0x... (NULL 아니어야 함)"
echo "    [fgpu] LAUNCH count=${LAUNCH_LOG_EVERY} (every ${LAUNCH_LOG_EVERY})"
echo "    ... 매 ${LAUNCH_LOG_EVERY} 회마다 한 줄 ..."
echo "    [fgpu] LAUNCH count=${LAUNCH_N} (every ${LAUNCH_LOG_EVERY})"
echo "    [fgpu] exit summary: total cudaLaunchKernel = ${LAUNCH_N}"
echo "  보이면 launch hook 통과."
