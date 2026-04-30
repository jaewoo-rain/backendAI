#!/usr/bin/env bash
# 5-A 확장: launch counter ↔ nvidia-smi 메모리 시계열 상관 실험.
#
# 두 컨테이너를 동시에 띄워서 각각 (a) 메모리 hold (b) cudaLaunchKernel
# 다발 워크로드를 실행. 같은 GPU 를 어떻게 *공유* 하는지 시간 trace 로
# 관찰 — 5-A 의 격리 검증과 다르게, 둘 다 quota 내라 *공존* 시나리오.
#
# 사전 조건
#   - scripts/run_backend.sh 실행 중
#   - scripts/build_hook.sh   → libfgpu.so (Stage 7 launch hook 포함)
#   - scripts/build_pytorch_image.sh  → PyTorch 이미지 (test_compute.py 포함)
#
# 산출물
#   experiments/correlation_<TS>/
#     container_a.log / container_b.log     docker logs --timestamps
#     pids_a.txt / pids_b.txt                docker top 으로 잡은 PID 목록
#     session_a.json / session_b.json        세션 메타
#     nvidia_smi.csv                          1초 polling 메모리 trace
#     correlation.csv                         시간 정렬 long-format trace
#     correlation_summary.txt                 사람이 읽는 요약
#
# 사용법
#   ./scripts/eval/run_correlation.sh
#   RATIO_A=0.3 RATIO_B=0.7 ALLOC_MIB=1024 HOLD_SEC=15 \
#       ./scripts/eval/run_correlation.sh
#   ALLOC_MIB=4096 ./scripts/eval/run_correlation.sh   # 5-A 식 OOM 재현

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
API="${FGPU_API:-http://localhost:8000}"
IMAGE="${FGPU_PYTORCH_IMAGE:-fgpu-runtime-pytorch:stage4}"
ALLOC_MIB="${ALLOC_MIB:-2048}"
HOLD_SEC="${HOLD_SEC:-10}"
RATIO_A="${RATIO_A:-0.4}"
RATIO_B="${RATIO_B:-0.6}"
LAUNCH_LOG_EVERY="${FGPU_LAUNCH_LOG_EVERY:-500}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT_DIR}/experiments/correlation_${TS}"
mkdir -p "${OUT_DIR}"

echo "[corr] output dir = ${OUT_DIR}"
echo "[corr] container A: ratio=${RATIO_A}  alloc=${ALLOC_MIB} MiB  hold=${HOLD_SEC}s"
echo "[corr] container B: ratio=${RATIO_B}  alloc=${ALLOC_MIB} MiB  hold=${HOLD_SEC}s"
echo "[corr] launch log every = ${LAUNCH_LOG_EVERY}"

# ---- helpers --------------------------------------------------------------
parse_field() {
    python3 -c "import sys, json; v=json.load(sys.stdin).get('$1'); print('' if v is None else v)"
}
post_session() {
    local ratio=$1
    curl -sS -X POST "${API}/sessions" \
        -H 'Content-Type: application/json' \
        -d "$(cat <<EOF
{
  "ratio": ${ratio},
  "image": "${IMAGE}",
  "command": ["python3","/opt/fgpu/test_compute.py"]
}
EOF
)"
}

# ---- preflight ------------------------------------------------------------
if ! curl -sS --fail "${API}/healthz" >/dev/null; then
    echo "ERROR: backend ${API} 응답 없음 — scripts/run_backend.sh 먼저." >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null; then
    echo "ERROR: nvidia-smi 가 PATH 에 없음." >&2
    exit 1
fi

# ---- 1) nvidia-smi 백그라운드 캡처 (spawn 보다 먼저) ----------------------
NVSMI_LOG="${OUT_DIR}/nvidia_smi.csv"
echo "timestamp, pid, process_name, used_memory" > "${NVSMI_LOG}"
nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_memory \
           --format=csv,noheader -l 1 >> "${NVSMI_LOG}" &
NVSMI_PID=$!
trap "kill ${NVSMI_PID} 2>/dev/null || true" EXIT

# ---- 2) 두 세션 spawn -----------------------------------------------------
# DockerManager 가 백엔드 프로세스의 FGPU_LAUNCH_LOG_EVERY 를 컨테이너로
# 자동 forward (whitelist passthrough). trace 가 의미 있게 나오려면
# 백엔드 자체를 다음과 같이 띄워야 함:
#   FGPU_LAUNCH_LOG_EVERY=${LAUNCH_LOG_EVERY} ./scripts/run_backend.sh
# 미설정 시 hook default 1000 → 짧은 워크로드에선 dump 가 1~2 줄밖에 안 찍힘.
echo "[corr] note: 백엔드 프로세스에 FGPU_LAUNCH_LOG_EVERY 가 설정돼 있어야"
echo "             trace 의 dump 빈도가 ${LAUNCH_LOG_EVERY} 로 적용됨."

RESP_A="$(post_session "${RATIO_A}")"
RESP_B="$(post_session "${RATIO_B}")"
SID_A="$(echo "${RESP_A}" | parse_field id)"
SID_B="$(echo "${RESP_B}" | parse_field id)"
CID_A="$(echo "${RESP_A}" | parse_field container_id)"
CID_B="$(echo "${RESP_B}" | parse_field container_id)"
if [[ -z "${SID_A}" || -z "${SID_B}" ]]; then
    echo "ERROR: 세션 생성 실패." >&2
    echo "  A: ${RESP_A}" >&2
    echo "  B: ${RESP_B}" >&2
    exit 1
fi
echo "[corr] session A id=${SID_A}  container=${CID_A:0:12}"
echo "[corr] session B id=${SID_B}  container=${CID_B:0:12}"

# ---- 3) 컨테이너 PID 캡처 — python3 이 뜰 시간 잠깐 줌 --------------------
sleep 2
docker top "${CID_A}" -o pid 2>/dev/null \
    | awk 'NR>1 {print $1}' > "${OUT_DIR}/pids_a.txt" || true
docker top "${CID_B}" -o pid 2>/dev/null \
    | awk 'NR>1 {print $1}' > "${OUT_DIR}/pids_b.txt" || true
echo "[corr] container A PIDs: $(tr '\n' ' ' < "${OUT_DIR}/pids_a.txt")"
echo "[corr] container B PIDs: $(tr '\n' ' ' < "${OUT_DIR}/pids_b.txt")"

# ---- 4) 둘 다 종료될 때까지 폴링 -------------------------------------------
DEADLINE=$((SECONDS + HOLD_SEC + 60))
while [[ ${SECONDS} -lt ${DEADLINE} ]]; do
    SA="$(curl -sS "${API}/sessions/${SID_A}" | parse_field status)"
    SB="$(curl -sS "${API}/sessions/${SID_B}" | parse_field status)"
    echo "[corr] t=${SECONDS}s  A=${SA}  B=${SB}"
    if [[ "${SA}" == "exited" && "${SB}" == "exited" ]]; then
        break
    fi
    sleep 1
done

# ---- 5) nvidia-smi 종료 ---------------------------------------------------
kill "${NVSMI_PID}" 2>/dev/null || true
wait "${NVSMI_PID}" 2>/dev/null || true
trap - EXIT

# ---- 6) timestamped logs + 세션 메타 저장 ---------------------------------
docker logs --timestamps "${CID_A}" > "${OUT_DIR}/container_a.log" 2>&1 || true
docker logs --timestamps "${CID_B}" > "${OUT_DIR}/container_b.log" 2>&1 || true
curl -sS "${API}/sessions/${SID_A}" | python3 -m json.tool > "${OUT_DIR}/session_a.json"
curl -sS "${API}/sessions/${SID_B}" | python3 -m json.tool > "${OUT_DIR}/session_b.json"

# ---- 7) DELETE -----------------------------------------------------------
curl -sS -X DELETE "${API}/sessions/${SID_A}" >/dev/null || true
curl -sS -X DELETE "${API}/sessions/${SID_B}" >/dev/null || true

# ---- 8) post-process: correlate ------------------------------------------
python3 "$(dirname "$0")/_correlate.py" "${OUT_DIR}"

echo
echo "[corr] artifacts:"
ls -la "${OUT_DIR}"
echo
echo "[corr] plot 예시 (Python pandas):"
echo "  import pandas as pd; df = pd.read_csv('${OUT_DIR}/correlation.csv')"
echo "  df_l = df.dropna(subset=['launch_count']).pivot(index='t_seconds', columns='container', values='launch_count')"
echo "  df_m = df.dropna(subset=['used_memory_mib']).pivot(index='t_seconds', columns='container', values='used_memory_mib')"
