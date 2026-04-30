#!/usr/bin/env bash
# =====================================================================
# fGPU runtime container entrypoint.
#
# 책임:
#   1) 컨테이너 시작 시점의 fGPU 관련 env / 마운트 상태를 stdout 에 찍는다
#      (디버깅 / 발표 시연 시 가시성 확보).
#   2) LD_PRELOAD 가 가리키는 .so 가 실제 존재하는지 확인하고, 없으면
#      경고만 남기고 계속 진행 (사용자가 의도적으로 unhooked 실행을
#      원할 수도 있으므로 fail-fast 하지 않는다).
#   3) "$@" 로 사용자 명령을 그대로 exec — entrypoint 가 PID 1 이 되어
#      시그널 (SIGTERM 등) 이 자식 프로세스로 전달되도록.
# =====================================================================

set -e

HOOK_PATH="/opt/fgpu/libfgpu.so"

echo "[entrypoint] container starting"
echo "[entrypoint]   FGPU_RATIO       = ${FGPU_RATIO:-<unset>}"
echo "[entrypoint]   FGPU_QUOTA_BYTES = ${FGPU_QUOTA_BYTES:-<unset>}"
echo "[entrypoint]   LD_PRELOAD       = ${LD_PRELOAD:-<unset>}"

if [[ -n "${LD_PRELOAD:-}" ]]; then
    if [[ -f "${LD_PRELOAD}" ]]; then
        echo "[entrypoint]   hook .so OK  -> $(ls -l "${LD_PRELOAD}" | awk '{print $5, $9}')"
    else
        echo "[entrypoint]   WARN: LD_PRELOAD=${LD_PRELOAD} 인데 파일이 없음."
        echo "[entrypoint]         '-v <host_libfgpu.so>:${HOOK_PATH}:ro' 빠뜨렸을 가능성."
        echo "[entrypoint]         그대로 진행 — 사용자 프로그램은 unhooked 상태로 실행될 수 있음."
    fi
fi

echo "[entrypoint] exec: $*"
exec "$@"
