#!/usr/bin/env python3
"""
5-A 확장 워크로드: alloc + 실제 GPU 작업 + free.

목적
  Stage 5-A 가 메모리 점유 시간 윈도우를 만들고 (alloc + sleep),
  Stage 7 가 launch counter 를 측정한다. 두 데이터를 동시에 *상관*
  시키려면 한 워크로드가 메모리도 잡고 launch 도 많이 발생시켜야 한다.
  test_hold.py 는 sleep 만 해서 launch=0. 이 파일이 그 빈 자리를 채움.

동작
  1) ALLOC_MIB MiB 의 큰 텐서 1개 (메모리 점유 — quota 가 측정됨)
  2) 1024x1024 작은 텐서 두 개로 matmul + relu + scale 루프
     HOLD_SEC 초 동안 — 매 iter ≈ 3 cudaLaunchKernel 호출
  3) 정리

env (모두 5-A 의 test_hold.py 와 호환)
  ALLOC_MIB     기본 2048    큰 텐서 크기 (MiB, float32)
  HOLD_SEC      기본 10      compute 루프 지속 시간
  COMPUTE_SIZE  기본 1024    matmul 텐서 한 변 (정사각)

종료 코드
  0  성공
  1  alloc 실패 (hook DENY 가 OOMError 로 전파됨) — 5-A 의 PASS-A 케이스
  2  CUDA 미가용

가정
  - Dockerfile 의 PYTORCH_NO_CUDA_MEMORY_CACHING=1.
"""
from __future__ import annotations

import os
import sys
import time
import torch


def main() -> None:
    alloc_mib    = int(os.environ.get("ALLOC_MIB",    "2048"))
    hold_sec     = int(os.environ.get("HOLD_SEC",     "10"))
    compute_size = int(os.environ.get("COMPUTE_SIZE", "1024"))

    if not torch.cuda.is_available():
        print("[compute-test] CUDA 사용 불가 — '--gpus all' 누락", flush=True)
        sys.exit(2)

    print(f"[compute-test] device       = {torch.cuda.get_device_name(0)}",
          flush=True)
    print(f"[compute-test] FGPU_RATIO   = "
          f"{os.environ.get('FGPU_RATIO', '<unset>')}", flush=True)
    print(f"[compute-test] alloc        = {alloc_mib} MiB", flush=True)
    print(f"[compute-test] hold         = {hold_sec} s", flush=True)
    print(f"[compute-test] compute size = {compute_size}^2", flush=True)

    n = (alloc_mib * 1024 * 1024) // 4
    try:
        big = torch.empty(n, dtype=torch.float32, device="cuda:0")
        torch.cuda.synchronize()
        print(f"[compute-test] BIG ALLOC OK  ptr={hex(big.data_ptr())}",
              flush=True)
    except torch.cuda.OutOfMemoryError as e:
        print(f"[compute-test] BIG ALLOC OOM ← hook DENY 전파됨",
              flush=True)
        print(f"[compute-test]               {e}", flush=True)
        sys.exit(1)

    # 매 iter 가 cudaLaunchKernel 다발: matmul + relu + 스케일.
    # 1024x1024 matmul 자체가 cuBLAS 내부에서 여러 launch 를 발생시킴 —
    # Stage 7 hook 의 카운터가 빠르게 증가.
    x = torch.randn(compute_size, compute_size, device="cuda:0")
    y = torch.randn(compute_size, compute_size, device="cuda:0")
    print(f"[compute-test] entering compute loop ...", flush=True)

    start = time.time()
    iters = 0
    while time.time() - start < hold_sec:
        z = x @ y
        z = torch.relu(z)
        x = z * 0.99 + 0.01
        iters += 1
        # 너무 빨리 돌면 launch queue 가 폭주 — 100 회마다 동기화로 압력 조절.
        if iters % 100 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"[compute-test] compute done  iters={iters}  elapsed={elapsed:.2f}s",
          flush=True)

    del big, x, y, z
    torch.cuda.empty_cache()
    print("[compute-test] cleanup done.", flush=True)


if __name__ == "__main__":
    main()
