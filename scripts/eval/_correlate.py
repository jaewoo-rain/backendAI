#!/usr/bin/env python3
"""
5-A 확장 post-processing.

입력 (experiments/correlation_<TS>/ 안)
  container_a.log         : `docker logs --timestamps` 출력. 각 줄이
                            `2024-04-28T11:25:30.123Z [fgpu] LAUNCH ...`
  container_b.log         : 동일
  pids_a.txt / pids_b.txt : `docker top` 의 PID 목록 (한 줄 하나)
  nvidia_smi.csv          : `nvidia-smi --query-compute-apps=
                            timestamp,pid,process_name,used_memory --format=csv`
  session_a.json / session_b.json : 컨테이너 메타

출력
  correlation.csv : columns t_seconds, container, launch_count, used_memory_mib
                    각 (시점, 컨테이너) 쌍에서 launch counter 와 GPU 메모리.
                    plot 친화적 long-format.

설계 결정
  - launch counter 는 hook 의 stderr line "[fgpu] LAUNCH count=N (every M)"
    에서 추출. docker --timestamps 가 매 줄 앞에 ISO8601 시각을 붙여줌.
  - nvidia-smi 의 timestamp 는 "YYYY/MM/DD HH:MM:SS.fff" 포맷 — 따로 파싱.
  - 두 시계열의 t=0 은 *전체 실험 시작 시점* (=earliest timestamp 중 최소).
  - PID set 으로 nvidia-smi row 를 컨테이너로 분류. 같은 컨테이너의 여러
    PID 는 used_memory 합산.

외부 dep 0 — stdlib 만 사용. matplotlib 안 씀 (CSV 만 출력).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


LAUNCH_RE = re.compile(
    r'^(?P<ts>\S+)\s+.*\[fgpu\] LAUNCH count=(?P<count>\d+)'
)


def parse_launch_log(path: Path) -> list[tuple[datetime, int]]:
    """반환: [(timestamp, cumulative_count), ...] 시간순"""
    out: list[tuple[datetime, int]] = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LAUNCH_RE.match(line)
        if not m:
            continue
        try:
            # docker --timestamps: "2024-04-28T11:25:30.123456789Z"
            ts_str = m.group("ts").rstrip("Z")
            # fromisoformat 은 9-digit nanosecond 를 못 받음 → 6 자리로 자름
            if "." in ts_str:
                whole, frac = ts_str.split(".", 1)
                frac = frac[:6]
                ts_str = f"{whole}.{frac}"
            ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
            count = int(m.group("count"))
            out.append((ts, count))
        except ValueError:
            continue
    return out


def parse_nvsmi(path: Path) -> list[tuple[datetime, int, int]]:
    """반환: [(timestamp, pid, used_mib), ...]"""
    out: list[tuple[datetime, int, int]] = []
    if not path.is_file():
        return out
    with path.open(encoding="utf-8", errors="replace") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row or row[0].strip().lower().startswith("timestamp"):
                continue
            if len(row) < 4:
                continue
            try:
                # "2024/04/28 11:25:30.123"
                ts = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                ts = ts.replace(tzinfo=timezone.utc)
                pid = int(row[1].strip())
                # used_memory like "2048 MiB"
                mem_str = row[3].strip().split()[0]
                mib = int(mem_str)
                out.append((ts, pid, mib))
            except (ValueError, IndexError):
                continue
    return out


def parse_pids(path: Path) -> set[int]:
    out: set[int] = set()
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.add(int(line))
        except ValueError:
            continue
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: _correlate.py <out_dir>", file=sys.stderr)
        sys.exit(2)
    out_dir = Path(sys.argv[1])
    if not out_dir.is_dir():
        print(f"out_dir not a directory: {out_dir}", file=sys.stderr)
        sys.exit(2)

    launches = {
        "A": parse_launch_log(out_dir / "container_a.log"),
        "B": parse_launch_log(out_dir / "container_b.log"),
    }
    pids = {
        "A": parse_pids(out_dir / "pids_a.txt"),
        "B": parse_pids(out_dir / "pids_b.txt"),
    }
    nvsmi = parse_nvsmi(out_dir / "nvidia_smi.csv")

    # t=0 = 모든 timestamp 중 최소
    all_ts: list[datetime] = []
    for entries in launches.values():
        all_ts.extend(t for t, _ in entries)
    all_ts.extend(t for t, _, _ in nvsmi)
    if not all_ts:
        print("no timestamps found in any input — nothing to correlate",
              file=sys.stderr)
        sys.exit(1)
    t0 = min(all_ts)

    # nvsmi 를 (시점, 컨테이너) 별 used_mib 합으로 집계
    nvsmi_by_container: dict[str, list[tuple[datetime, int]]] = {"A": [], "B": []}
    for ts, pid, mib in nvsmi:
        for c in ("A", "B"):
            if pid in pids[c]:
                nvsmi_by_container[c].append((ts, mib))
                break

    # CSV 출력 (long format).
    # launch row: (t, container, launch_count, "")
    # nvsmi  row: (t, container, "", used_mib)
    # plot 시 두 측을 별도 컬럼으로 join 하기 쉽게 둘을 모두 long-format 으로
    # 둠. wide format 변환은 user 측 plot 도구 (pandas/excel) 가 함.
    rows: list[tuple[float, str, str, str]] = []
    for c, entries in launches.items():
        for ts, count in entries:
            t = (ts - t0).total_seconds()
            rows.append((t, c, str(count), ""))
    for c, entries in nvsmi_by_container.items():
        for ts, mib in entries:
            t = (ts - t0).total_seconds()
            rows.append((t, c, "", str(mib)))
    rows.sort(key=lambda r: (r[0], r[1]))

    out_csv = out_dir / "correlation.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_seconds", "container", "launch_count", "used_memory_mib"])
        for r in rows:
            w.writerow([f"{r[0]:.3f}", r[1], r[2], r[3]])

    # 짧은 텍스트 요약.
    summary_path = out_dir / "correlation_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# 5-A 확장 — launch counter ↔ nvidia-smi 상관 trace\n\n")
        f.write(f"experiment dir: {out_dir.name}\n")
        f.write(f"t0 (UTC):       {t0.isoformat()}\n\n")
        for c in ("A", "B"):
            launch_n = len(launches[c])
            mem_n    = len(nvsmi_by_container[c])
            final_count = launches[c][-1][1] if launches[c] else 0
            peak_mib    = max((m for _, m in nvsmi_by_container[c]), default=0)
            sess_path = out_dir / f"session_{c.lower()}.json"
            ratio = "?"
            if sess_path.is_file():
                try:
                    j = json.loads(sess_path.read_text(encoding="utf-8"))
                    ratio = f"{j.get('ratio', '?')}"
                except json.JSONDecodeError:
                    pass
            f.write(f"container {c} (ratio={ratio})\n")
            f.write(f"  PIDs                    = {sorted(pids[c]) or '<none>'}\n")
            f.write(f"  launch dump rows        = {launch_n}\n")
            f.write(f"  final cumulative count  = {final_count}\n")
            f.write(f"  nvidia-smi rows         = {mem_n}\n")
            f.write(f"  peak used memory (MiB)  = {peak_mib}\n\n")

    print(f"[correlate] wrote {out_csv}")
    print(f"[correlate] wrote {summary_path}")


if __name__ == "__main__":
    main()
