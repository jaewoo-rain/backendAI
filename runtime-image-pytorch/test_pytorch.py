#!/usr/bin/env python3
"""
Stage 4 검증 스크립트 — PyTorch 텐서 할당이 LD_PRELOAD hook 의 quota 를 거치는지.

기본 시나리오 (RTX 4060, 8 GB):
  1) 256 MiB 텐서 → 모든 합리적 quota 에서 ALLOW
  2) 4   GiB 텐서 → baseline ALLOW
                    ratio=0.4 (quota 3.2 GiB) → DENY
                    ratio=0.6 (quota 4.8 GiB) → ALLOW
  → ratio 별 동작 차이가 가장 명확히 보이는 demarcation.

env override (논문 실험 시):
  PYTEST_ALLOC1_MIB=128
  PYTEST_ALLOC2_MIB=6144

핵심 가정:
  - PYTORCH_NO_CUDA_MEMORY_CACHING=1 (Dockerfile 기본).
    caching 이 켜져 있으면 첫 텐서가 큰 chunk 를 잡고 끝 → quota 효과 X.
  - PYTORCH_CUDA_ALLOC_CONF 미설정 (default native = cudaMalloc 경로).
"""

from __future__ import annotations

import os
import sys
import torch


def banner(s: str) -> None:
    print(f"\n=== {s} ===", flush=True)


def device_info() -> None:
    print(f"[pytorch-test] torch.__version__              = {torch.__version__}", flush=True)
    print(f"[pytorch-test] CUDA available                  = {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        sys.exit("[pytorch-test] CUDA not available — '--gpus all' 빠뜨렸을 가능성.")
    print(f"[pytorch-test] device                          = {torch.cuda.get_device_name(0)}", flush=True)
    free, total = torch.cuda.mem_get_info()
    print(f"[pytorch-test] mem(bytes)                      : free={free} total={total}", flush=True)
    print(f"[pytorch-test] FGPU_RATIO                      = {os.environ.get('FGPU_RATIO', '<unset>')}", flush=True)
    print(f"[pytorch-test] PYTORCH_NO_CUDA_MEMORY_CACHING  = "
          f"{os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING', '<unset>')}", flush=True)
    print(f"[pytorch-test] PYTORCH_CUDA_ALLOC_CONF         = "
          f"{os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}", flush=True)
    print(f"[pytorch-test] LD_PRELOAD                      = {os.environ.get('LD_PRELOAD', '<unset>')}", flush=True)


def try_alloc(size_mib: int):
    n_floats = (size_mib * 1024 * 1024) // 4   # float32 = 4 bytes
    print(f"[pytorch-test] allocating {size_mib} MiB ({n_floats} float32 elements) ...", flush=True)
    try:
        t = torch.empty(n_floats, dtype=torch.float32, device="cuda:0")
        # synchronize 안 하면 lazy 할당 가능 — 강제로 commit.
        torch.cuda.synchronize()
        print(f"[pytorch-test]   OK   data_ptr={hex(t.data_ptr())}", flush=True)
        return t
    except torch.cuda.OutOfMemoryError as e:
        print(f"[pytorch-test]   OOM  ← cudaErrorMemoryAllocation 이 PyTorch 까지 전파됨", flush=True)
        print(f"[pytorch-test]        {e}", flush=True)
        return None
    except RuntimeError as e:
        # 예상치 못한 다른 에러도 죽지 않고 기록.
        print(f"[pytorch-test]   ERR  {type(e).__name__}: {e}", flush=True)
        return None


def main() -> None:
    a1 = int(os.environ.get("PYTEST_ALLOC1_MIB", "256"))
    a2 = int(os.environ.get("PYTEST_ALLOC2_MIB", "4096"))

    banner("device & env")
    device_info()

    banner(f"alloc {a1} MiB")
    t1 = try_alloc(a1)

    banner(f"alloc {a2} MiB")
    t2 = try_alloc(a2)

    banner("cleanup")
    del t1, t2
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    print(f"[pytorch-test] mem after cleanup: free={free} total={total}", flush=True)
    print("[pytorch-test] done.", flush=True)


if __name__ == "__main__":
    main()
