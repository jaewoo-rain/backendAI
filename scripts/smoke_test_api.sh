#!/usr/bin/env bash
# Stage 3 합격 검증: API 한 바퀴 (curl 만 사용).
#
# 사전 조건:
#   - scripts/run_backend.sh 가 별도 터미널에서 동작 중
#
# 사용법:
#   ./scripts/smoke_test_api.sh                # ratio 0.4 기본
#   FGPU_RATIO=0.6 ./scripts/smoke_test_api.sh

set -euo pipefail

API="${FGPU_API:-http://localhost:8000}"
RATIO="${FGPU_RATIO:-0.4}"

# python3 가 jq 대신 (의존성 줄이기)
parse_json_field() {
    python3 -c "import sys, json; print(json.load(sys.stdin)['$1'])"
}

echo "============================================================"
echo "[smoke] (1) GET ${API}/healthz"
echo "============================================================"
curl -sS "${API}/healthz" | python3 -m json.tool
echo

echo "============================================================"
echo "[smoke] (2) POST ${API}/sessions  ratio=${RATIO}"
echo "============================================================"
RESP=$(curl -sS -X POST "${API}/sessions" \
    -H 'Content-Type: application/json' \
    -d "{\"ratio\": ${RATIO}}")
echo "${RESP}" | python3 -m json.tool
SID=$(echo "${RESP}" | parse_json_field id)
echo "[smoke] session id = ${SID}"
echo

echo "[smoke] sleep 4 — test_alloc 실행 시간 확보"
sleep 4

echo "============================================================"
echo "[smoke] (3) GET ${API}/sessions/${SID}"
echo "============================================================"
curl -sS "${API}/sessions/${SID}" | python3 -m json.tool
echo

echo "============================================================"
echo "[smoke] (4) GET ${API}/sessions/${SID}/logs"
echo "============================================================"
curl -sS "${API}/sessions/${SID}/logs" | parse_json_field logs
echo

echo "============================================================"
echo "[smoke] (5) DELETE ${API}/sessions/${SID}"
echo "============================================================"
curl -sS -X DELETE "${API}/sessions/${SID}" | python3 -m json.tool
echo

echo "[smoke] done."
