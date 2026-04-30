#!/usr/bin/env bash
# Build the Stage 2 runtime image. Run on the Ubuntu GPU server.
#
# 환경변수 (override 가능):
#   IMAGE_NAME    기본 fgpu-runtime
#   IMAGE_TAG     기본 stage2
#   CUDA_VERSION  기본 12.4.1  (host 의 CUDA major 와 일치 권장)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-fgpu-runtime}"
IMAGE_TAG="${IMAGE_TAG:-stage2}"
CUDA_VERSION="${CUDA_VERSION:-12.4.1}"

cd "${ROOT_DIR}"

echo "[build] context = ${ROOT_DIR}"
echo "[build] image   = ${IMAGE_NAME}:${IMAGE_TAG}"
echo "[build] CUDA    = ${CUDA_VERSION}"

docker build \
    -f runtime-image/Dockerfile \
    --build-arg CUDA_VERSION="${CUDA_VERSION}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    .

echo
echo "[build] done."
docker images --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}' \
    | grep -E "^REPOSITORY|^${IMAGE_NAME}\s"
