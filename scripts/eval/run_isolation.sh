#!/usr/bin/env bash
# Stage 5-A: 동시 격리 실험.
#
# 목표
#   사용자 두 명이 동시에 /sessions 를 호출했을 때 hook quota 가 process-level
#   로 격리되는지 측정. 같은 워크로드(ALLOC_MIB MiB hold)에 ratio 만 다르게
#   주고, 한 쪽은 quota 초과로 OOM, 다른 한 쪽은 quota 내 ALLOW 가 나오는지
#   확인. 동시에 nvidia-smi --query-compute-apps 로 메모리 점유 추이 캡처.
#
# 사전 조건
#   - scripts/run_backend.sh 가 별도 터미널에서 동작 중
#   - scripts/build_pytorch_image.sh 로 fgpu-runtime-pytorch:stage4 빌드됨
#   - test_hold.py 가 들어간 새 이미지 — Dockerfile 갱신 후 재빌드 필수.
#
# 기본 시나리오 (RTX 4060 / 8 GB)
#   container A: ratio=0.4, alloc=4096 MiB → quota 3.2 GiB 초과 → OOM (exit 1)
#   container B: ratio=0.6, alloc=4096 MiB → quota 4.8 GiB 내   → ALLOW (exit 0)
#
# 산출물
#   experiments/isolation_<TS>/
#     container_a.log   container_b.log
#     session_a.json    session_b.json
#     nvidia_smi.csv    summary.txt
#
# 사용법
#   ./scripts/eval/run_isolation.sh
#   ALLOC_MIB=3072 HOLD_SEC=8 ./scripts/eval/run_isolation.sh
#   RATIO_A=0.3 RATIO_B=0.7 ./scripts/eval/run_isolation.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
API="${FGPU_API:-http://localhost:8000}"
IMAGE="${FGPU_PYTORCH_IMAGE:-fgpu-runtime-pytorch:stage4}"
ALLOC_MIB="${ALLOC_MIB:-4096}"
HOLD_SEC="${HOLD_SEC:-6}"
RATIO_A="${RATIO_A:-0.4}"
RATIO_B="${RATIO_B:-0.6}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT_DIR}/experiments/isolation_${TS}"
mkdir -p "${OUT_DIR}"

echo "[eval] output dir = ${OUT_DIR}"
echo "[eval] container A: ratio=${RATIO_A}  alloc=${ALLOC_MIB} MiB  hold=${HOLD_SEC}s"
echo "[eval] container B: ratio=${RATIO_B}  alloc=${ALLOC_MIB} MiB  hold=${HOLD_SEC}s"
echo

# ---- helpers ---------------------------------------------------------------
parse_field() {  # parse_field <field> < json
    python3 -c "import sys, json; v=json.load(sys.stdin).get('$1'); print('' if v is None else v)"
}
post_session() {
    local ratio=$1
    # ALLOC_MIB / HOLD_SEC 도 컨테이너 env 로 넘겨 워크로드에 반영.
    curl -sS -X POST "${API}/sessions" \
        -H 'Content-Type: application/json' \
        -d "$(cat <<EOF
{
  "ratio": ${ratio},
  "image": "${IMAGE}",
  "command": ["python3","/opt/fgpu/test_hold.py"]
}
EOF
)"
}

# ---- preflight -------------------------------------------------------------
if ! curl -sS --fail "${API}/healthz" >/dev/null; then
    echo "ERROR: 백엔드 ${API} 응답 없음. scripts/run_backend.sh 먼저." >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null; then
    echo "ERROR: nvidia-smi 가 PATH 에 없음." >&2
    exit 1
fi

# ---- 1) nvidia-smi 백그라운드 캡처 (세션 spawn 보다 먼저) -------------------
NVSMI_LOG="${OUT_DIR}/nvidia_smi.csv"
echo "timestamp, pid, process_name, used_memory" > "${NVSMI_LOG}"
nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_memory \
           --format=csv,noheader -l 1 >> "${NVSMI_LOG}" &
NVSMI_PID=$!
trap "kill ${NVSMI_PID} 2>/dev/null || true" EXIT

# ---- 2) 두 세션 동시 spawn -------------------------------------------------
# NOTE: 현재 백엔드는 docker SDK 호출이 sync 라 *세션 생성* 자체는 직렬화됨.
#       워크로드 (test_hold.py 의 hold 구간) 는 정상적으로 시간이 겹친다.
#       — 향후 SessionManager.create 를 asyncio.to_thread 로 감싸면 진짜 동시.
RESP_A="$(post_session "${RATIO_A}")"
RESP_B="$(post_session "${RATIO_B}")"
SID_A="$(echo "${RESP_A}" | parse_field id)"
SID_B="$(echo "${RESP_B}" | parse_field id)"
if [[ -z "${SID_A}" || -z "${SID_B}" ]]; then
    echo "ERROR: 세션 생성 실패. 응답:" >&2
    echo "  A: ${RESP_A}" >&2
    echo "  B: ${RESP_B}" >&2
    exit 1
fi
echo "[eval] session A id = ${SID_A}"
echo "[eval] session B id = ${SID_B}"

# ---- 3) 둘 다 종료될 때까지 폴링 -------------------------------------------
DEADLINE=$((SECONDS + HOLD_SEC + 60))
STATUS_A=""
STATUS_B=""
while [[ ${SECONDS} -lt ${DEADLINE} ]]; do
    STATUS_A="$(curl -sS "${API}/sessions/${SID_A}" | parse_field status)"
    STATUS_B="$(curl -sS "${API}/sessions/${SID_B}" | parse_field status)"
    echo "[eval] t=${SECONDS}s  A=${STATUS_A}  B=${STATUS_B}"
    if [[ "${STATUS_A}" == "exited" && "${STATUS_B}" == "exited" ]]; then
        break
    fi
    sleep 1
done

# ---- 4) nvidia-smi 종료 ----------------------------------------------------
kill "${NVSMI_PID}" 2>/dev/null || true
wait "${NVSMI_PID}" 2>/dev/null || true
trap - EXIT

# ---- 5) 로그·세션 메타 저장 ------------------------------------------------
fetch_logs() {  # fetch_logs <sid>
    curl -sS "${API}/sessions/$1/logs?tail=2000" \
        | python3 -c "import sys, json; print(json.load(sys.stdin).get('logs',''))"
}
curl -sS "${API}/sessions/${SID_A}" | python3 -m json.tool > "${OUT_DIR}/session_a.json"
curl -sS "${API}/sessions/${SID_B}" | python3 -m json.tool > "${OUT_DIR}/session_b.json"
fetch_logs "${SID_A}" > "${OUT_DIR}/container_a.log"
fetch_logs "${SID_B}" > "${OUT_DIR}/container_b.log"

# ---- 6) 정리 — DELETE -----------------------------------------------------
curl -sS -X DELETE "${API}/sessions/${SID_A}" >/dev/null || true
curl -sS -X DELETE "${API}/sessions/${SID_B}" >/dev/null || true

# ---- 7) verdict ------------------------------------------------------------
EXIT_A="$(python3 -c "import json; print(json.load(open('${OUT_DIR}/session_a.json')).get('exit_code'))")"
EXIT_B="$(python3 -c "import json; print(json.load(open('${OUT_DIR}/session_b.json')).get('exit_code'))")"
LOG_A_OOM=$(grep -c '\[hold-test\] OOM' "${OUT_DIR}/container_a.log" || true)
LOG_B_OK=$(grep -c '\[hold-test\] OK'  "${OUT_DIR}/container_b.log" || true)
HOOK_A_DENY=$(grep -c '\[fgpu\].*DENY' "${OUT_DIR}/container_a.log" || true)
HOOK_B_ALLOW=$(grep -c '\[fgpu\].*ALLOW' "${OUT_DIR}/container_b.log" || true)

PASS_A=0
PASS_B=0
[[ "${EXIT_A}" == "1" && "${LOG_A_OOM}" -ge 1 && "${HOOK_A_DENY}" -ge 1 ]] && PASS_A=1
[[ "${EXIT_B}" == "0" && "${LOG_B_OK}"  -ge 1 && "${HOOK_B_ALLOW}" -ge 1 ]] && PASS_B=1

{
    echo "=== Stage 5-A isolation experiment ==="
    echo "timestamp:        ${TS}"
    echo "image:            ${IMAGE}"
    echo "alloc per cnt:    ${ALLOC_MIB} MiB"
    echo "hold per cnt:     ${HOLD_SEC} s"
    echo
    echo "container A (ratio=${RATIO_A})"
    echo "  exit_code     = ${EXIT_A}    (expected 1 — OOM)"
    echo "  [hold-test] OOM lines = ${LOG_A_OOM}"
    echo "  [fgpu] DENY lines     = ${HOOK_A_DENY}"
    echo "  PASS_A        = ${PASS_A}"
    echo
    echo "container B (ratio=${RATIO_B})"
    echo "  exit_code     = ${EXIT_B}    (expected 0 — ALLOW)"
    echo "  [hold-test] OK lines  = ${LOG_B_OK}"
    echo "  [fgpu] ALLOW lines    = ${HOOK_B_ALLOW}"
    echo "  PASS_B        = ${PASS_B}"
    echo
    if [[ ${PASS_A} -eq 1 && ${PASS_B} -eq 1 ]]; then
        echo "VERDICT: PASS — quota isolated, A blocked, B allowed."
    else
        echo "VERDICT: FAIL — see container_a.log / container_b.log"
    fi
} | tee "${OUT_DIR}/summary.txt"

echo
echo "[eval] artifacts:"
ls -la "${OUT_DIR}"
