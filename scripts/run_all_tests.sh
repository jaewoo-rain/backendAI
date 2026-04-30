#!/usr/bin/env bash
# 모든 stage 의 happy-path 검증을 한 번에 실행.
#
# 동작
#   1) preflight (nvidia-smi, docker --gpus)
#   2) 빌드 (이미 있으면 skip): hook .so, runtime image, pytorch image
#   3) stage 1, 2, 5-C, 6, 7, 4 — 각자 컨테이너 단독 검증
#   4) backend pytest (docker / GPU 불필요)
#   5) 백엔드 spawn → /healthz 대기 → smoke + 5-A isolation + 5-A 확장
#   6) 5-D overhead (백엔드 거치지 않음)
#   7) 백엔드 정리 + 최종 PASS/FAIL 표
#
# 모든 stdout/stderr 는 experiments/runall_<TS>/ 아래 별도 .log 로 캡처.
# 한 단계 실패해도 나머지 계속 진행해서 최대 정보 수집 (set -e 안 씀).
#
# 4070 / 12 GB 환경 기준으로 ALLOC 사이즈를 6144 MiB 로 키워둠 — 8 GB 이하
# GPU 면 OOM 시나리오가 더 빨리 발생하지만 검증 자체는 통과.
#
# 사용법:
#   ./scripts/run_all_tests.sh                 # 처음 1회 — pytorch image 빌드 포함 ~10 분
#   ./scripts/run_all_tests.sh                 # 이후 — 빌드 skip 으로 ~5 분
#
# 종료 코드: 0 = 모두 PASS, 1 = 하나 이상 FAIL.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT_DIR}/experiments/runall_${TS}"
mkdir -p "${OUT_DIR}"

BACKEND_PORT="${FGPU_BACKEND_PORT:-8000}"
RESULTS=()

# ---------- helpers ----------
record_pass() { RESULTS+=("PASS|$1|"); echo "  [PASS] $1"; }
record_fail() { RESULTS+=("FAIL|$1|$2"); echo "  [FAIL] $1 — $2"; }
record_skip() { RESULTS+=("SKIP|$1|$2"); echo "  [SKIP] $1 — $2"; }

# run_check <step_name> <command...>  → log + 종료 코드만 본다
run_step() {
    local step=$1; shift
    local log="${OUT_DIR}/${step}.log"
    echo
    echo "=== ${step} ==="
    if "$@" >"${log}" 2>&1; then
        return 0
    else
        return $?
    fi
}

# pattern_check <step_name> <command...> -- <regex>
# 로그에 regex 가 있으면 PASS, 없으면 FAIL.
pattern_check() {
    local step=$1; shift
    local cmd=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do cmd+=("$1"); shift; done
    shift  # consume "--"
    local pattern=$1
    local log="${OUT_DIR}/${step}.log"
    echo
    echo "=== ${step} ==="
    if "${cmd[@]}" >"${log}" 2>&1; then
        if grep -qE "${pattern}" "${log}"; then
            record_pass "${step}"
        else
            record_fail "${step}" "pattern '${pattern}' missing — see ${log}"
        fi
    else
        record_fail "${step}" "exit non-zero — see ${log}"
    fi
}

# ---------- preflight ----------
echo "[runall] preflight"
if ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi 동작 안 함. NVIDIA 드라이버 설치 필요. LINUX_SETUP.md §2 참조." >&2
    exit 2
fi
if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi \
        >"${OUT_DIR}/_preflight_docker_gpu.log" 2>&1; then
    echo "ERROR: docker GPU 통과 실패. nvidia-container-toolkit 설치 필요. LINUX_SETUP.md §3 참조." >&2
    exit 2
fi
echo "  OK — nvidia-smi + docker GPU 통과"

# ---------- build (idempotent) ----------
if [[ ! -f "${ROOT_DIR}/build/libfgpu.so" ]]; then
    echo "[runall] build hook (build/libfgpu.so 없음 — 빌드)"
    if ! "${ROOT_DIR}/scripts/build_hook.sh" >"${OUT_DIR}/_build_hook.log" 2>&1; then
        echo "ERROR: build_hook.sh 실패 — ${OUT_DIR}/_build_hook.log 확인"
        exit 2
    fi
fi
if ! docker image inspect fgpu-runtime:stage2 >/dev/null 2>&1; then
    echo "[runall] build base image (~3 분)"
    if ! "${ROOT_DIR}/scripts/build_image.sh" >"${OUT_DIR}/_build_image.log" 2>&1; then
        echo "ERROR: build_image.sh 실패 — ${OUT_DIR}/_build_image.log 확인"
        exit 2
    fi
fi
if ! docker image inspect fgpu-runtime-pytorch:stage4 >/dev/null 2>&1; then
    echo "[runall] build pytorch image (첫 빌드 ~5~10 분, 5 GB 휠 다운)"
    if ! "${ROOT_DIR}/scripts/build_pytorch_image.sh" \
            >"${OUT_DIR}/_build_pytorch.log" 2>&1; then
        echo "ERROR: build_pytorch_image.sh 실패 — ${OUT_DIR}/_build_pytorch.log 확인"
        exit 2
    fi
fi
echo "  OK — 모든 빌드 완료"

# ---------- backend-less stages ----------
pattern_check stage1_host \
    "${ROOT_DIR}/scripts/run_test.sh" \
    -- '\[fgpu\] DENY'

pattern_check stage2_container \
    "${ROOT_DIR}/scripts/run_in_container.sh" \
    -- '\[fgpu\] DENY'

pattern_check stage5c_driver \
    "${ROOT_DIR}/scripts/run_driver_in_container.sh" \
    -- '\[fgpu\] DENY  cuMemAlloc_v2'

pattern_check stage6_vmm \
    "${ROOT_DIR}/scripts/run_vmm_in_container.sh" \
    -- '\[fgpu\] DENY  cuMemCreate'

pattern_check stage7_launch \
    "${ROOT_DIR}/scripts/run_launch_in_container.sh" \
    -- 'exit summary: total cudaLaunchKernel = 1000'

# Stage 4 — env override 라 별도 호출
echo
echo "=== stage4_pytorch ==="
PYTEST_ALLOC2_MIB=6144 "${ROOT_DIR}/scripts/run_pytorch_in_container.sh" \
    >"${OUT_DIR}/stage4_pytorch.log" 2>&1
if grep -qE '\[pytorch-test\]   OOM' "${OUT_DIR}/stage4_pytorch.log" \
   && grep -qE '\[fgpu\] DENY' "${OUT_DIR}/stage4_pytorch.log"; then
    record_pass stage4_pytorch
else
    record_fail stage4_pytorch "OOM 또는 DENY 라인 누락 — ${OUT_DIR}/stage4_pytorch.log 확인"
fi

# ---------- backend unit tests ----------
echo
echo "=== backend_pytest ==="
(
  cd "${ROOT_DIR}/backend"
  python3 -m venv .venv 2>/dev/null
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -q --upgrade pip
  pip install -q -e ".[dev]"
  pytest -q
) >"${OUT_DIR}/backend_pytest.log" 2>&1
if grep -qE 'passed' "${OUT_DIR}/backend_pytest.log"; then
    record_pass backend_pytest
else
    record_fail backend_pytest "${OUT_DIR}/backend_pytest.log 확인"
fi

# ---------- backend lifecycle ----------
echo
echo "=== backend_start (auth off, FGPU_LAUNCH_LOG_EVERY=500) ==="
FGPU_LAUNCH_LOG_EVERY=500 "${ROOT_DIR}/scripts/run_backend.sh" \
    >"${OUT_DIR}/backend.log" 2>&1 &
BACKEND_PID=$!
trap '
  echo "[runall] stopping backend (PID '"${BACKEND_PID}"')"
  kill '"${BACKEND_PID}"' 2>/dev/null || true
  wait '"${BACKEND_PID}"' 2>/dev/null || true
' EXIT

# /healthz 응답 대기 (최대 30 초)
ready=0
for _ in $(seq 1 30); do
    if curl -sS --fail "http://localhost:${BACKEND_PORT}/healthz" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 1
done
if [[ ${ready} -eq 1 ]]; then
    record_pass backend_start
else
    record_fail backend_start "30 초 내 /healthz 응답 없음 — ${OUT_DIR}/backend.log 확인"
    # 백엔드 없으면 이후 스텝 모두 skip
    record_skip stage3_smoke      "backend 미동작"
    record_skip stage5a_isolation "backend 미동작"
    record_skip stage5a_corr      "backend 미동작"
    pattern_check stage5d_overhead \
        "${ROOT_DIR}/scripts/eval/run_overhead.sh" \
        -- 'cudaMalloc latency'   # 5-D 는 backend 안 씀
    # summary 로 점프
    BACKEND_DOWN=1
fi

if [[ "${BACKEND_DOWN:-0}" -ne 1 ]]; then
    # Stage 3 smoke
    pattern_check stage3_smoke \
        "${ROOT_DIR}/scripts/smoke_test_api.sh" \
        -- '\[smoke\] done\.'

    # Stage 5-A isolation
    echo
    echo "=== stage5a_isolation ==="
    ALLOC_MIB=6144 HOLD_SEC=8 "${ROOT_DIR}/scripts/eval/run_isolation.sh" \
        >"${OUT_DIR}/stage5a_isolation.log" 2>&1
    iso_dir=$(ls -1dt "${ROOT_DIR}"/experiments/isolation_* 2>/dev/null | head -1)
    if [[ -n "${iso_dir}" ]] && grep -qE 'VERDICT: PASS' "${iso_dir}/summary.txt" 2>/dev/null; then
        record_pass stage5a_isolation
    else
        record_fail stage5a_isolation "verdict not PASS — ${iso_dir:-experiments/isolation_*}/summary.txt 확인"
    fi

    # Stage 5-A correlation extension
    echo
    echo "=== stage5a_corr ==="
    "${ROOT_DIR}/scripts/eval/run_correlation.sh" \
        >"${OUT_DIR}/stage5a_corr.log" 2>&1
    cor_dir=$(ls -1dt "${ROOT_DIR}"/experiments/correlation_* 2>/dev/null | head -1)
    if [[ -n "${cor_dir}" ]] && [[ -s "${cor_dir}/correlation.csv" ]]; then
        # CSV 가 header 만 있고 데이터가 0줄이면 fail
        rows=$(wc -l < "${cor_dir}/correlation.csv")
        if [[ "${rows}" -gt 1 ]]; then
            record_pass stage5a_corr
        else
            record_fail stage5a_corr "correlation.csv 가 header 만 있음 — workload 가 너무 짧거나 PID join 실패"
        fi
    else
        record_fail stage5a_corr "correlation.csv 없음"
    fi

    # Stage 5-D overhead
    pattern_check stage5d_overhead \
        "${ROOT_DIR}/scripts/eval/run_overhead.sh" \
        -- 'cudaMalloc latency'
fi

# ---------- summary ----------
echo
echo "============================================================"
echo "[runall] SUMMARY  (artifacts: ${OUT_DIR})"
echo "============================================================"
n_pass=0; n_fail=0; n_skip=0
for r in "${RESULTS[@]}"; do
    case "${r%%|*}" in
        PASS) n_pass=$((n_pass+1)) ;;
        FAIL) n_fail=$((n_fail+1)) ;;
        SKIP) n_skip=$((n_skip+1)) ;;
    esac
done
for r in "${RESULTS[@]}"; do
    IFS='|' read -r status name detail <<< "${r}"
    printf "  [%-4s] %-22s %s\n" "${status}" "${name}" "${detail}"
done
echo
echo "  Pass: ${n_pass}   Fail: ${n_fail}   Skip: ${n_skip}"
echo

if [[ ${n_fail} -gt 0 ]]; then
    exit 1
fi
exit 0
