#!/usr/bin/env bash
# Stage 4: PyTorch 변형 이미지 빌드. Ubuntu GPU 서버에서 실행.
#
# 사전 조건:
#   - scripts/build_image.sh 로 fgpu-runtime:stage2 가 이미 빌드됨
#
# 첫 빌드는 PyTorch 휠 다운로드 때문에 5~10 분 + 디스크 ~5 GB 추가 사용.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BASE_IMAGE="${BASE_IMAGE:-fgpu-runtime:stage2}"
IMAGE_NAME="${IMAGE_NAME:-fgpu-runtime-pytorch}"
IMAGE_TAG="${IMAGE_TAG:-stage4}"

cd "${ROOT_DIR}"

if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
    echo "ERROR: 베이스 이미지 ${BASE_IMAGE} 가 없음." >&2
    echo "       scripts/build_image.sh 를 먼저 실행." >&2
    exit 1
fi

echo "[build] base    = ${BASE_IMAGE}"
echo "[build] image   = ${IMAGE_NAME}:${IMAGE_TAG}"
echo "[build] context = ${ROOT_DIR}"

docker build \
    -f runtime-image-pytorch/Dockerfile \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    .

echo
echo "[build] done."
docker images --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}' \
    | grep -E "^REPOSITORY|^${IMAGE_NAME}\s"
