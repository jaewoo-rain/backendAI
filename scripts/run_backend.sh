#!/usr/bin/env bash
# Stage 3: FastAPI 백엔드 실행.
#
# 사전 조건:
#   - scripts/build_hook.sh 로 build/libfgpu.so 생성됨
#   - scripts/build_image.sh 로 fgpu-runtime:stage2 이미지 존재
#   - 현재 사용자가 docker 그룹 소속 (docker socket 접근 가능)
#
# 사용법:
#   ./scripts/run_backend.sh                      # default port 8000
#   FGPU_BACKEND_PORT=8080 ./scripts/run_backend.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_DIR="${ROOT_DIR}/backend"
PORT="${FGPU_BACKEND_PORT:-8000}"
HOST="${FGPU_BACKEND_HOST:-0.0.0.0}"

cd "${BACKEND_DIR}"

# venv (한 번만)
if [[ ! -d .venv ]]; then
    echo "[run_backend] venv 생성"
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 의존성 (idempotent — 이미 설치된 경우 빠르게 통과)
pip install -q --upgrade pip
pip install -q -e .

# 백엔드가 컨테이너 spawn 시 마운트할 호스트 hook 경로를 명시.
export FGPU_HOST_HOOK_PATH="${FGPU_HOST_HOOK_PATH:-${ROOT_DIR}/build/libfgpu.so}"
export FGPU_RUNTIME_IMAGE="${FGPU_RUNTIME_IMAGE:-fgpu-runtime:stage2}"

if [[ ! -f "${FGPU_HOST_HOOK_PATH}" ]]; then
    echo "[run_backend] WARN: ${FGPU_HOST_HOOK_PATH} 가 없음. scripts/build_hook.sh 먼저 실행 필요."
fi

echo "[run_backend] FGPU_HOST_HOOK_PATH = ${FGPU_HOST_HOOK_PATH}"
echo "[run_backend] FGPU_RUNTIME_IMAGE  = ${FGPU_RUNTIME_IMAGE}"
echo "[run_backend] uvicorn http://${HOST}:${PORT}"

exec uvicorn app.main:app --host "${HOST}" --port "${PORT}" --reload
