# =====================================================================
# Stage 4: PyTorch variant of the fGPU runtime image.
#
# 기존 fgpu-runtime:stage2 위에 Python + PyTorch (CUDA 12.1 build) 설치.
# 사용자가 image="fgpu-runtime-pytorch:stage4" 로 세션 생성 요청하면
# PyTorch 텐서 할당이 hook 의 quota 검사를 거치도록 한다.
#
# 핵심 환경변수:
#   PYTORCH_NO_CUDA_MEMORY_CACHING=1   기본값 (아래 ENV).
#     PyTorch CUDACachingAllocator 비활성. 이렇게 해야 매 텐서 할당이
#     cudaMalloc 으로 직접 들어가 hook quota 가 적용됨. caching 이
#     켜져 있으면 첫 큰 chunk 한 번만 cudaMalloc 호출 → 이후 quota 효과 X.
#
# Hook 호환성 메모:
#   - Stage 1 hook 은 cudaMalloc / cudaFree 만 잡는다.
#   - PyTorch 가 cudaMallocAsync 백엔드를 쓰면 hook 우회됨. 이 때문에
#     PYTORCH_CUDA_ALLOC_CONF 는 *설정하지 않는다* (default native).
#   - Driver API (cuMemAlloc_v2 등) hook 은 Stage 6 에서 추가 예정.
# =====================================================================

ARG BASE_IMAGE=fgpu-runtime:stage2
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Python + pip (devel base 에는 없음)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CUDA 12.1 build — base 이미지의 CUDA 12.4 runtime 과 호환.
# torch 단일 설치 (torchvision/torchaudio 불필요).
# --index-url 로 PyTorch CDN 만 사용 → CPU-only 휠이 잘못 깔리는 사고 방지.
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch

# 테스트 스크립트
COPY runtime-image-pytorch/test_pytorch.py /opt/fgpu/test_pytorch.py
COPY runtime-image-pytorch/test_hold.py    /opt/fgpu/test_hold.py
COPY runtime-image-pytorch/test_compute.py /opt/fgpu/test_compute.py

# caching off 기본값 (필수). -e 로 override 가능 (논문 비교 실험용).
ENV PYTORCH_NO_CUDA_MEMORY_CACHING=1

# 베이스 이미지의 ENTRYPOINT (/usr/local/bin/fgpu-entrypoint) 는 그대로 상속.
# 기본 CMD 만 PyTorch 테스트로 변경.
CMD ["python3", "/opt/fgpu/test_pytorch.py"]
