# =====================================================================
# fGPU prototype — Stage 2 runtime image
#
# 사용 시나리오:
#   docker run --rm --gpus all \
#     -v $PWD/build/libfgpu.so:/opt/fgpu/libfgpu.so:ro \
#     -e LD_PRELOAD=/opt/fgpu/libfgpu.so \
#     -e FGPU_RATIO=0.4 \
#     fgpu-runtime:stage2
#
# 설계 결정:
#   1) devel 이미지를 쓴다 — 컨테이너 안에서 test_alloc.cu 를 nvcc 로
#      미리 컴파일해 두기 위함. runtime 이미지보다 크지만 프로토타입이라
#      디스크 크기보다 재현성이 우선.
#   2) libfgpu.so 는 이미지에 굽지 않고 *런타임 마운트* 한다. 이렇게
#      해야 hook 을 고쳤을 때 이미지 재빌드가 필요 없다 — 백엔드가
#      여러 hook 버전을 운용할 때도 유리.
#   3) LD_PRELOAD 는 default ENV 로 지정하지 않는다. .so 미마운트 상태로
#      LD_PRELOAD 가 켜져 있으면 동적 링커가 시끄럽게 실패한다 (ld.so:
#      object cannot be preloaded). docker run -e 로 명시 전달이 명확.
#   4) Host 의 libfgpu.so 가 container 의 libcudart 와 ABI 호환되어야
#      한다. CUDA major version 일치를 권장 — 양쪽 모두 12.x 이면 OK.
# =====================================================================

ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# hook 마운트 지점 + 컨테이너 내장 테스트 바이너리 위치
RUN mkdir -p /opt/fgpu

# 컨테이너 내부에서 test_alloc 을 미리 컴파일.
# 이렇게 하면 host/container 의 libcudart ABI 일치 가정에 의존하는
# 부분을 binary 단에서는 피할 수 있다 (hook .so 만 host 산출물).
COPY hook/tests/test_alloc.cu        /tmp/test_alloc.cu
COPY hook/tests/bench_alloc.cu       /tmp/bench_alloc.cu
COPY hook/tests/test_driver_alloc.cu /tmp/test_driver_alloc.cu
COPY hook/tests/test_launch.cu       /tmp/test_launch.cu
COPY hook/tests/test_vmm_alloc.cu    /tmp/test_vmm_alloc.cu
# test_driver_alloc / test_vmm_alloc 는 Driver/VMM API (libcuda) 를 직접
# 호출하므로 -lcuda 필요. nvcc 가 기본으로 -lcudart 만 link 하므로 명시 추가.
RUN nvcc -O2 -o /opt/fgpu/test_alloc        /tmp/test_alloc.cu        \
 && nvcc -O2 -o /opt/fgpu/bench_alloc       /tmp/bench_alloc.cu       \
 && nvcc -O2 -o /opt/fgpu/test_driver_alloc /tmp/test_driver_alloc.cu -lcuda \
 && nvcc -O2 -o /opt/fgpu/test_launch       /tmp/test_launch.cu       \
 && nvcc -O2 -o /opt/fgpu/test_vmm_alloc    /tmp/test_vmm_alloc.cu    -lcuda \
 && rm /tmp/test_alloc.cu /tmp/bench_alloc.cu /tmp/test_driver_alloc.cu /tmp/test_launch.cu /tmp/test_vmm_alloc.cu

# Entrypoint: env / 마운트 sanity check 후 사용자 명령 exec.
COPY runtime-image/entrypoint.sh /usr/local/bin/fgpu-entrypoint
RUN chmod +x /usr/local/bin/fgpu-entrypoint

ENTRYPOINT ["/usr/local/bin/fgpu-entrypoint"]
CMD ["/opt/fgpu/test_alloc"]
