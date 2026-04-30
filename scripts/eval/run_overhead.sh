#!/usr/bin/env bash
# Stage 5-D: cudaMalloc / cudaFree overhead 측정.
#
# 같은 bench 바이너리를 컨테이너에서 두 번 실행:
#   (1) baseline — LD_PRELOAD 없이.
#   (2) hooked   — libfgpu.so 마운트 + FGPU_RATIO=0.95
#                  (모든 사이즈가 quota 내에서 ALLOW 되도록 충분히 큼)
#
# 결과
#   experiments/overhead_<TS>/
#     baseline_raw.csv   bench_alloc 의 raw CSV (size_mib,iter,malloc_ns,free_ns)
#     hooked_raw.csv
#     summary.csv        per-(size,mode) mean / p50 / p99 (μs 단위)
#     summary.txt        markdown 표 형태 + 메타
#
# 사전 조건
#   - scripts/build_hook.sh         → build/libfgpu.so
#   - scripts/build_image.sh        → fgpu-runtime:stage2  (bench_alloc 포함된 새 이미지)
#
# 사용법
#   ./scripts/eval/run_overhead.sh
#   BENCH_SIZES_MIB=1,4,16,64,256 BENCH_N=200 ./scripts/eval/run_overhead.sh
#   FGPU_RATIO=0.5 ./scripts/eval/run_overhead.sh   # 작은 quota 에서의 overhead

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${IMAGE:-fgpu-runtime:stage2}"
HOOK_SO_HOST="${ROOT_DIR}/build/libfgpu.so"
RATIO="${FGPU_RATIO:-0.95}"
BENCH_SIZES_MIB="${BENCH_SIZES_MIB:-16,64,256,1024}"
BENCH_N="${BENCH_N:-100}"
BENCH_WARMUP="${BENCH_WARMUP:-5}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT_DIR}/experiments/overhead_${TS}"
mkdir -p "${OUT_DIR}"

# ---- preflight ------------------------------------------------------------
if [[ ! -f "${HOOK_SO_HOST}" ]]; then
    echo "ERROR: ${HOOK_SO_HOST} 가 없음. scripts/build_hook.sh 먼저." >&2
    exit 1
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "ERROR: 이미지 ${IMAGE} 없음. scripts/build_image.sh 먼저." >&2
    exit 1
fi
# bench_alloc 이 이미지 안에 있는지 확인 — Dockerfile 갱신 후 재빌드 안 했으면 여기서 실패.
if ! docker run --rm --entrypoint /bin/sh "${IMAGE}" \
        -c '[ -x /opt/fgpu/bench_alloc ]' >/dev/null 2>&1; then
    echo "ERROR: ${IMAGE} 안에 /opt/fgpu/bench_alloc 가 없음." >&2
    echo "       scripts/build_image.sh 로 이미지 재빌드 필요." >&2
    exit 1
fi

echo "[overhead] sizes_mib   = ${BENCH_SIZES_MIB}"
echo "[overhead] N           = ${BENCH_N}  warmup = ${BENCH_WARMUP}"
echo "[overhead] hooked ratio= ${RATIO}"
echo "[overhead] out dir     = ${OUT_DIR}"
echo

# ---- 1) baseline ----------------------------------------------------------
echo "============================================================"
echo "[overhead] (1/2) baseline (no hook)"
echo "============================================================"
docker run --rm --gpus all \
    --entrypoint /opt/fgpu/bench_alloc \
    -e BENCH_SIZES_MIB="${BENCH_SIZES_MIB}" \
    -e BENCH_N="${BENCH_N}" \
    -e BENCH_WARMUP="${BENCH_WARMUP}" \
    "${IMAGE}" \
    > "${OUT_DIR}/baseline_raw.csv" \
    2> "${OUT_DIR}/baseline_stderr.log"
echo "[overhead] baseline rows: $(($(wc -l < "${OUT_DIR}/baseline_raw.csv") - 1))"

# ---- 2) hooked ------------------------------------------------------------
echo "============================================================"
echo "[overhead] (2/2) hooked (LD_PRELOAD, ratio=${RATIO})"
echo "============================================================"
docker run --rm --gpus all \
    --entrypoint /opt/fgpu/bench_alloc \
    -v "${HOOK_SO_HOST}:/opt/fgpu/libfgpu.so:ro" \
    -e LD_PRELOAD=/opt/fgpu/libfgpu.so \
    -e FGPU_RATIO="${RATIO}" \
    -e BENCH_SIZES_MIB="${BENCH_SIZES_MIB}" \
    -e BENCH_N="${BENCH_N}" \
    -e BENCH_WARMUP="${BENCH_WARMUP}" \
    "${IMAGE}" \
    > "${OUT_DIR}/hooked_raw.csv" \
    2> "${OUT_DIR}/hooked_stderr.log"
echo "[overhead] hooked rows:   $(($(wc -l < "${OUT_DIR}/hooked_raw.csv") - 1))"

# ---- 3) 요약 통계 ---------------------------------------------------------
python3 - "$OUT_DIR" "$BENCH_SIZES_MIB" "$BENCH_N" "$BENCH_WARMUP" "$RATIO" "$IMAGE" "$TS" <<'PYEOF'
import csv, os, statistics, sys

out_dir, sizes_str, N, warmup, ratio, image, ts = sys.argv[1:8]

def load(path):
    """size_mib -> [(malloc_ns, free_ns), ...]"""
    data = {}
    with open(path) as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if not row or row[0].startswith('#'):
                continue
            try:
                size = int(row[0])
                m_ns = int(row[2])
                f_ns = int(row[3])
            except (ValueError, IndexError):
                continue
            data.setdefault(size, []).append((m_ns, f_ns))
    return data

def pct(xs, q):  # q in [0,100]
    if not xs: return float('nan')
    s = sorted(xs)
    if len(s) == 1: return s[0]
    k = (len(s) - 1) * (q / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)

def stats_for(samples):
    m = [s[0] / 1000.0 for s in samples]   # ns → μs
    f = [s[1] / 1000.0 for s in samples]
    return {
        'n':       len(samples),
        'm_mean':  statistics.fmean(m) if m else float('nan'),
        'm_p50':   pct(m, 50),
        'm_p99':   pct(m, 99),
        'f_mean':  statistics.fmean(f) if f else float('nan'),
        'f_p50':   pct(f, 50),
        'f_p99':   pct(f, 99),
    }

base = load(os.path.join(out_dir, 'baseline_raw.csv'))
hook = load(os.path.join(out_dir, 'hooked_raw.csv'))

sizes = sorted(set(base) | set(hook))

# summary.csv
with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['size_mib','mode','n',
                'malloc_mean_us','malloc_p50_us','malloc_p99_us',
                'free_mean_us','free_p50_us','free_p99_us'])
    for sz in sizes:
        for mode, src in [('baseline', base), ('hooked', hook)]:
            if sz not in src: continue
            s = stats_for(src[sz])
            w.writerow([sz, mode, s['n'],
                        f"{s['m_mean']:.2f}", f"{s['m_p50']:.2f}", f"{s['m_p99']:.2f}",
                        f"{s['f_mean']:.2f}", f"{s['f_p50']:.2f}", f"{s['f_p99']:.2f}"])

# summary.txt — paper-friendly markdown table + meta
lines = []
lines.append("# fGPU overhead — Stage 5-D")
lines.append("")
lines.append(f"- timestamp:       {ts}")
lines.append(f"- image:           {image}")
lines.append(f"- sizes_mib:       {sizes_str}")
lines.append(f"- N (per size):    {N}")
lines.append(f"- warmup (ignored):{warmup}")
lines.append(f"- hooked ratio:    {ratio}")
lines.append("")
lines.append("## cudaMalloc latency (μs)")
lines.append("")
lines.append("| size_mib | baseline mean | baseline p50 | baseline p99 | hooked mean | hooked p50 | hooked p99 | Δ mean | Δ mean % |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for sz in sizes:
    if sz not in base or sz not in hook: continue
    b = stats_for(base[sz]); h = stats_for(hook[sz])
    delta   = h['m_mean'] - b['m_mean']
    delta_p = (delta / b['m_mean'] * 100) if b['m_mean'] > 0 else float('nan')
    lines.append(f"| {sz} | {b['m_mean']:.2f} | {b['m_p50']:.2f} | {b['m_p99']:.2f}"
                 f" | {h['m_mean']:.2f} | {h['m_p50']:.2f} | {h['m_p99']:.2f}"
                 f" | {delta:+.2f} | {delta_p:+.1f}% |")

lines.append("")
lines.append("## cudaFree latency (μs)")
lines.append("")
lines.append("| size_mib | baseline mean | baseline p50 | baseline p99 | hooked mean | hooked p50 | hooked p99 | Δ mean | Δ mean % |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for sz in sizes:
    if sz not in base or sz not in hook: continue
    b = stats_for(base[sz]); h = stats_for(hook[sz])
    delta   = h['f_mean'] - b['f_mean']
    delta_p = (delta / b['f_mean'] * 100) if b['f_mean'] > 0 else float('nan')
    lines.append(f"| {sz} | {b['f_mean']:.2f} | {b['f_p50']:.2f} | {b['f_p99']:.2f}"
                 f" | {h['f_mean']:.2f} | {h['f_p50']:.2f} | {h['f_p99']:.2f}"
                 f" | {delta:+.2f} | {delta_p:+.1f}% |")

text = "\n".join(lines) + "\n"
with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
    f.write(text)
print(text)
PYEOF

echo
echo "[overhead] artifacts:"
ls -la "${OUT_DIR}"
