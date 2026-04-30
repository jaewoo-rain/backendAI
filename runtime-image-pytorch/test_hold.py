#!/usr/bin/env python3
"""
Stage 5-A 워크로드: ALLOC_MIB MiB 한 번 할당 후 HOLD_SEC 초 보유.

목적
  두 컨테이너가 *동시에* GPU 메모리를 점유하는 시간 윈도우를 만들어
  hook quota 가 process-level 로 격리되는지 nvidia-smi + 컨테이너 로그로 관측.

test_pytorch.py 와의 차이
  test_pytorch.py 는 alloc → free → 즉시 exit. 동시 격리 실험에서는
  *시간 겹침* 이 핵심이므로 한 번 잡고 hold 하는 워크로드가 따로 필요.

env
  ALLOC_MIB    기본 4096   할당할 텐서 크기 (MiB, float32)
  HOLD_SEC     기본 6      할당 후 보유 시간

종료 코드
  0  할당 성공 + hold 완료 (paper verdict 의 PASS-B 케이스)
  1  할당 실패 — hook quota DENY 가 PyTorch OOM 으로 전파됨 (PASS-A 케이스)
  2  CUDA 자체 불가 (—gpus all 빠뜨림 등 환경 오류)

가정
  - PYTORCH_NO_CUDA_MEMORY_CACHING=1 (Dockerfile 기본).
    caching 켜져 있으면 첫 chunk 가 quota 보다 크게 잡혀 의미 없는 결과.
"""
from __future__ import annotations

import os
import sys
import time
import torch


def main() -> None:
    alloc_mib = int(os.environ.get("ALLOC_MIB", "4096"))
    hold_sec = int(os.environ.get("HOLD_SEC", "6"))
    n_floats = (alloc_mib * 1024 * 1024) // 4  # float32 = 4 bytes

    if not torch.cuda.is_available():
        print("[hold-test] CUDA 사용 불가 — '--gpus all' 누락 가능성", flush=True)
        sys.exit(2)

    print(f"[hold-test] device      = {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[hold-test] FGPU_RATIO  = {os.environ.get('FGPU_RATIO', '<unset>')}", flush=True)
    print(f"[hold-test] alloc {alloc_mib} MiB ({n_floats} float32)  hold {hold_sec}s",
          flush=True)

    try:
        t = torch.empty(n_floats, dtype=torch.float32, device="cuda:0")
        torch.cuda.synchronize()
        print(f"[hold-test] OK   ptr={hex(t.data_ptr())}", flush=True)
    except torch.cuda.OutOfMemoryError as e:
        print(f"[hold-test] OOM  ← cudaErrorMemoryAllocation 이 PyTorch 까지 전파됨",
              flush=True)
        print(f"[hold-test]      {e}", flush=True)
        sys.exit(1)

    print(f"[hold-test] holding {hold_sec}s ...", flush=True)
    time.sleep(hold_sec)
    print("[hold-test] release.", flush=True)
    del t
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
