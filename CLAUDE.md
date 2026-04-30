# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

A capstone/research prototype that mimics Backend.AI's fractional GPU (fGPU): a single NVIDIA GPU is shared by multiple Docker containers, each receiving a fractional quota (e.g. 0.4 / 0.6) of GPU memory. The mechanism is **LD_PRELOAD-based CUDA API hooking** — `libfgpu.so` is injected into each container, intercepts `cudaMalloc` / `cudaFree`, and enforces a per-process memory quota.

This is **not** a production-grade GPU virtualizer. SM-level / hardware isolation is out of scope (RTX 4060 has no MIG). Treat it as "cooperative quota enforcement at the CUDA API boundary."

## Workflow rule (important)

The owner builds this in numbered stages and verifies each stage before moving on. **Do not jump ahead** — when a stage is complete, propose the next stage in prose and wait for the user to say "다음" before writing its code. See `description.md` for the full stage roadmap; the current stage is reflected by what's actually present in the tree.

Each stage's deliverable must be buildable and verifiable on its own.

## Repository layout (current, Stage 6 + Stage 9 minimal + run-all orchestrator)

- `hook/src/fgpu_hook.c` — the LD_PRELOAD hook. C, no C++ deps. Heavily commented in Korean for the owner. Hooks four things: Runtime alloc API (`cudaMalloc`/`cudaFree`), Driver alloc API (`cuMemAlloc_v2`/`cuMemFree_v2`, Stage 5-C), VMM API (`cuMemCreate`/`cuMemRelease`, Stage 6 — quota charged at physical alloc, VA reservation/mapping not hooked), and `cudaLaunchKernel` for monitoring (Stage 7). All alloc layers share `g_used`/`g_quota`/`g_lock`/`g_allocs`. A per-thread `__thread g_in_hook` flag prevents double-counting if one alloc API delegates to another and re-enters our hooks. The launch hook does NOT enforce quota — it only counts via lock-free `__atomic_fetch_add` and dumps the cumulative count every `FGPU_LAUNCH_LOG_EVERY` calls (default 1000) plus once on `atexit`.
- `hook/tests/test_alloc.cu` — tiny standalone CUDA program that triggers `cudaMalloc` twice (256 MiB then 6 GiB) so a `FGPU_RATIO=0.4` run produces one ALLOW + one DENY in the log.
- `hook/tests/bench_alloc.cu` — Stage 5-D microbenchmark. For each size in `BENCH_SIZES_MIB` (default `16,64,256,1024`), runs `BENCH_WARMUP` warmup cycles then `BENCH_N` measured `cudaMalloc`/`cudaFree` pairs, timed with `clock_gettime(CLOCK_MONOTONIC)`. Streams CSV (`size_mib,iter,malloc_ns,free_ns`) on stdout; `[bench]` meta and `[fgpu]` hook lines go to stderr.
- `hook/tests/test_driver_alloc.cu` — Stage 5-C smoke. Uses Driver API only (`cuInit`, `cuCtxCreate`, `cuMemAlloc_v2`, `cuMemFree_v2`) — Runtime API is deliberately untouched so the driver-layer hook is verified in isolation. Tries 256 MiB then 6 GiB so a `FGPU_RATIO=0.4` run yields ALLOW + DENY at the driver layer.
- `hook/tests/test_launch.cu` — Stage 7 smoke. Launches a tiny noop kernel `PYTEST_LAUNCH_N` times (default 1000). Used to verify `cudaLaunchKernel` is intercepted and counted; quota is irrelevant here (the kernel does ~no allocation).
- `hook/tests/test_vmm_alloc.cu` — Stage 6 smoke. Uses VMM API only (`cuInit`, `cuCtxCreate`, `cuMemGetAllocationGranularity`, `cuMemCreate`, `cuMemRelease`) — Runtime/Driver-classic allocs are deliberately untouched so the VMM-layer hook is verified in isolation. Tries 256 MiB then 6 GiB so a `FGPU_RATIO=0.4` run yields ALLOW + DENY at the VMM layer.
- `runtime-image/Dockerfile` — `nvidia/cuda:*-devel-ubuntu22.04` base image with `test_alloc` pre-compiled at `/opt/fgpu/test_alloc`. Hook .so is *mounted in at runtime*, not baked.
- `runtime-image/entrypoint.sh` — logs FGPU env + verifies hook .so existence, then `exec "$@"`. Default CMD runs the bundled test.
- `backend/` — FastAPI + Docker SDK session manager (Stage 3). Layout:
  - `pyproject.toml` — deps: fastapi, uvicorn, docker-py, pydantic, pydantic-settings. `[project.optional-dependencies] dev` adds pytest. `[tool.pytest.ini_options] testpaths = ["tests"]` so a bare `pytest` from `backend/` runs the suite.
  - `tests/test_session_store.py` — pytest unit tests for Stage 8 SessionStore: insert/get round-trip (datetime preservation), list ordering, update_status, delete, multi-instance same-DB visibility (backend restart simulation). Covers SQLite paths only — SessionManager + docker integration is tested via the eval scripts on actual GPU hardware.
  - `app/main.py` — app factory; wires `DockerManager` + `SessionManager` into `app.state`.
  - `app/core/config.py` — `Settings` via `FGPU_*` env vars; auto-detects `<repo>/build/libfgpu.so`.
  - `app/api/sessions.py` — REST router (`POST /sessions`, `GET`, `/logs`, `/stop`, `DELETE`). Stage 9 minimal: a `_require_auth` dependency on the router checks `Authorization: Bearer <token>` against `app.state.api_token`. Empty token = auth disabled (default for development). Constant-time compare via `hmac.compare_digest`.
  - `app/services/docker_manager.py` — assembles `--gpus all` (or `--gpus device=N` if `gpu_index` set, Stage 9 minimal multi-GPU), hook .so bind-mount, `LD_PRELOAD` + `FGPU_RATIO` env when spawning containers. Also forwards a whitelist of backend env vars (`_PASSTHROUGH_ENV` — currently just `FGPU_LAUNCH_LOG_EVERY`) so operators can control hook behavior across all spawned sessions by setting it on the backend process.
  - `app/services/session_manager.py` — Stage 8: SQLite-backed via `SessionStore`, no in-memory dict. All blocking calls (docker SDK + sqlite3) wrapped in `asyncio.to_thread()` so the event loop stays responsive — concurrent POST `/sessions` requests truly run in parallel. Reconciles status from docker daemon on every read; writes back any status change to SQLite.
  - `app/services/session_store.py` — Stage 8 persistence. stdlib `sqlite3` only (no new deps). Sync CRUD; SessionManager wraps each call in `to_thread`. New connection per call (sqlite3.Connection isn't thread-safe). `contextlib.closing` ensures connections are closed (the `with sqlite3.connect(...)` context manager handles transactions but doesn't close).
  - `app/schemas/session.py` — `Session`, `SessionCreate`, `SessionLogs` models. Stage 9 minimal added `gpu_index: Optional[int]` to both — `None` = all GPUs (default), `0/1/...` = pin to that device.
  - `app/static/index.html` — Stage 5-B minimal UI (extended for Stage 9 minimal). Vanilla JS + inline CSS, no build step. Served at `GET /` via `FileResponse` (no `StaticFiles` mount, since assets = 1 file). Toolbar has an API token input (saved to `localStorage`, auto-sent as `Authorization: Bearer <token>` on all fetches). Create form has `ratio`, `gpu_index` (optional, blank = all GPUs), `image`, `command` fields. Sessions table shows a `gpu` column (`all` or device id).
- `scripts/build_hook.sh` — builds `build/libfgpu.so` with `gcc -shared -fPIC` against `/usr/local/cuda`.
- `scripts/run_test.sh` — host-side: builds the test binary with `nvcc` and runs it twice (baseline + with LD_PRELOAD).
- `scripts/build_image.sh` — `docker build` the runtime image (default tag `fgpu-runtime:stage2`).
- `scripts/run_in_container.sh` — runs the bundled test inside the container twice (baseline + with hook mounted via `-v`).
- `scripts/run_driver_in_container.sh` — Stage 5-C verification. Same baseline+hooked pattern as `run_in_container.sh`, but the entrypoint is `/opt/fgpu/test_driver_alloc` so only the driver hook is exercised.
- `scripts/run_launch_in_container.sh` — Stage 7 verification. baseline+hooked of `test_launch`, with `FGPU_LAUNCH_LOG_EVERY=100` so the hooked run prints periodic launch counter dumps plus a final `[fgpu] exit summary` line.
- `scripts/run_vmm_in_container.sh` — Stage 6 verification. Same baseline+hooked pattern as `run_driver_in_container.sh`, but the entrypoint is `/opt/fgpu/test_vmm_alloc` so only the VMM hook is exercised.
- `scripts/run_backend.sh` — venv + `pip install -e .` + uvicorn on `:8000`. Sets `FGPU_HOST_HOOK_PATH` from `<repo>/build/libfgpu.so`.
- `scripts/smoke_test_api.sh` — curl-based one-shot test of the full API loop (create → get → logs → delete).
- `scripts/run_all_tests.sh` — orchestrator. Runs preflight → idempotent build → every per-stage smoke (1, 2, 5-C, 6, 7, 4, backend pytest) → spawn backend → 3, 5-A, 5-A correlation → 5-D → cleanup → PASS/FAIL summary table. Per-step logs under `experiments/runall_<TS>/<step>.log`. Exit 0 if all pass, 1 otherwise. Handles ALLOC sizing for ≥8 GB GPUs (default 6144 MiB → OOM at ratio 0.4 quota).
- `runtime-image-pytorch/Dockerfile` — Stage 4 variant. `FROM fgpu-runtime:stage2`, adds `python3` + PyTorch (cu121 wheel). Sets `PYTORCH_NO_CUDA_MEMORY_CACHING=1` as default ENV so caching allocator doesn't mask the hook. Inherits the base image's `ENTRYPOINT`; default `CMD` runs `test_pytorch.py`.
- `runtime-image-pytorch/test_pytorch.py` — allocates a small (256 MiB) then a large (4 GiB) `torch.empty` on `cuda:0` and synchronizes, so `OutOfMemoryError` from the hook is observable. Sizes overridable via `PYTEST_ALLOC1_MIB` / `PYTEST_ALLOC2_MIB`.
- `runtime-image-pytorch/test_compute.py` — 5-A correlation extension workload. Allocates `ALLOC_MIB` then runs a matmul + relu + scale loop for `HOLD_SEC` seconds (each iter = several `cudaLaunchKernel` calls so the Stage 7 counter advances quickly), then frees. Used to generate combined (memory + launch frequency) traces.
- `scripts/build_pytorch_image.sh` — builds `fgpu-runtime-pytorch:stage4` on top of the base. First run pulls the PyTorch wheel (~5 GB, several minutes).
- `scripts/run_pytorch_in_container.sh` — runs the PyTorch test twice: baseline (no hook) then hooked with `FGPU_RATIO`. Same `-v` bind-mount pattern as Stage 2.
- `runtime-image-pytorch/test_hold.py` — Stage 5-A workload. Allocates `ALLOC_MIB` MiB on `cuda:0`, holds `HOLD_SEC` seconds, frees. Exits 0 on ALLOW, 1 on OOM, 2 on missing CUDA. Used by the isolation experiment so two containers' memory holds overlap in time.
- `scripts/eval/run_isolation.sh` — Stage 5-A driver. Spawns two sessions via `POST /sessions` (different `FGPU_RATIO`), captures `nvidia-smi --query-compute-apps -l 1` to CSV in the background, polls until both exit, fetches logs + session JSON, prints PASS/FAIL verdict. Artifacts go under `experiments/isolation_<TS>/` (gitignore candidate — these get large).
- `scripts/eval/run_overhead.sh` — Stage 5-D driver. Runs `bench_alloc` inside the runtime container twice (baseline / hooked with `FGPU_RATIO=0.95`), captures CSVs, computes per-size mean/p50/p99 of `cudaMalloc` and `cudaFree` latency in μs, emits `summary.csv` + paper-friendly `summary.txt` markdown table under `experiments/overhead_<TS>/`. Bypasses the FastAPI session manager (uses `docker run --entrypoint /opt/fgpu/bench_alloc` directly) — measurement target is hook overhead, not API overhead.
- `scripts/eval/run_correlation.sh` — 5-A correlation extension. Spawns two PyTorch sessions running `test_compute.py` with different ratios, captures `nvidia-smi` memory + container PIDs (`docker top`) + timestamped logs (`docker logs --timestamps`). Calls `_correlate.py` to join everything into `correlation.csv` (long format: `t_seconds, container, launch_count, used_memory_mib`). Designed for two containers that *coexist* under quota — set `ALLOC_MIB=4096` to recreate 5-A's OOM scenario instead.
- `scripts/eval/_correlate.py` — post-processing helper invoked by `run_correlation.sh`. stdlib only; parses ISO8601 docker timestamps + `nvidia-smi` CSV format, joins by PID set per container, emits `correlation.csv` and `correlation_summary.txt`.
- `description.md` — long-form architecture / rationale doc, Korean. Read this for the *why*.
- `LINUX_SETUP.md` — end-to-end first-run runbook for a fresh Ubuntu / RTX 4070 machine after `git clone`. Driver / CUDA / Docker / nvidia-container-toolkit setup, build, per-stage PASS criteria, troubleshooting table. Use when guiding the user through environment migration.

The Stage 3 backend supports Stage 4 with **zero code changes** — `SessionCreate` already accepts an `image` field, and `DockerManager.create_container` honors it. So `POST /sessions` with `{"ratio":0.4,"image":"fgpu-runtime-pytorch:stage4","command":["python3","/opt/fgpu/test_pytorch.py"]}` works as-is.

`scripts/eval/` is now seeded by Stage 5-A; further benchmark scripts should be added alongside `run_isolation.sh` rather than scattered elsewhere.

## Build & run (on Ubuntu GPU server)

```bash
chmod +x scripts/*.sh runtime-image/entrypoint.sh

# Stage 1 — host-only verification
./scripts/build_hook.sh           # → build/libfgpu.so
./scripts/run_test.sh             # baseline + hooked, default FGPU_RATIO=0.4

# Stage 2 — container verification
./scripts/build_image.sh          # → fgpu-runtime:stage2
./scripts/run_in_container.sh     # baseline + hooked inside container

# Stage 3 — API
./scripts/run_backend.sh          # uvicorn on :8000 (foreground, --reload)
# in another terminal:
./scripts/smoke_test_api.sh       # full create/get/logs/delete cycle

# Stage 4 — PyTorch image
./scripts/build_pytorch_image.sh  # → fgpu-runtime-pytorch:stage4 (5+ min first time)
./scripts/run_pytorch_in_container.sh                     # baseline + ratio=0.4 (default)
FGPU_RATIO=0.6 ./scripts/run_pytorch_in_container.sh      # ratio=0.6 should ALLOW 4 GiB

# Stage 5-A — concurrent isolation experiment
# (Dockerfile gained test_hold.py — rebuild the PyTorch image first)
./scripts/build_pytorch_image.sh
./scripts/run_backend.sh          # in another terminal, leave running
./scripts/eval/run_isolation.sh   # → experiments/isolation_<TS>/{summary.txt,*.log,*.json,*.csv}

# Stage 5-B — minimal Web UI
./scripts/run_backend.sh          # uvicorn :8000 (--reload picks up the new route)
# open http://localhost:8000/  (or  http://<server-ip>:8000/  from another machine)

# Stage 5-D — overhead microbenchmark
# (runtime-image Dockerfile gained bench_alloc.cu — rebuild the base image first)
./scripts/build_image.sh
./scripts/eval/run_overhead.sh    # → experiments/overhead_<TS>/{baseline_raw.csv,hooked_raw.csv,summary.csv,summary.txt}
BENCH_N=200 BENCH_SIZES_MIB=1,4,16,64,256,1024 ./scripts/eval/run_overhead.sh

# Stage 5-C — Driver API hook
# (hook/src and runtime-image both changed — rebuild .so AND base image)
./scripts/build_hook.sh           # → build/libfgpu.so with cuMemAlloc_v2/cuMemFree_v2 hooks
./scripts/build_image.sh          # → fgpu-runtime:stage2 with /opt/fgpu/test_driver_alloc
./scripts/run_driver_in_container.sh                  # baseline + hooked (default ratio 0.4)
FGPU_RATIO=0.6 ./scripts/run_driver_in_container.sh   # 6 GiB still DENY (quota 4.8 GiB), but tighter
# After Stage 5-C the existing 5-D bench can be re-run unchanged — the same .so now also covers driver-layer.

# Stage 7 — cudaLaunchKernel monitoring
# (hook/src and runtime-image both changed — rebuild .so AND base image)
./scripts/build_hook.sh           # adds cudaLaunchKernel hook + atexit summary
./scripts/build_image.sh          # adds /opt/fgpu/test_launch
./scripts/run_launch_in_container.sh                            # default n=1000, every=100
PYTEST_LAUNCH_N=10000 FGPU_LAUNCH_LOG_EVERY=1000 \
    ./scripts/run_launch_in_container.sh                        # heavier launch volume
FGPU_LAUNCH_LOG_EVERY=0 ./scripts/run_launch_in_container.sh    # log off — for re-running 5-D overhead

# One-shot — every stage at once (recommended)
./scripts/run_all_tests.sh                     # → PASS/FAIL summary + experiments/runall_<TS>/

# Stage 8 — SQLite persistence + asyncio.to_thread wrapping
# (backend code only — no rebuild needed; uvicorn --reload picks it up)
./scripts/run_backend.sh                       # starts uvicorn; first run creates data/sessions.db
./scripts/smoke_test_api.sh                    # POST/GET/DELETE round-trip; record persists
# Restart-resilience verification:
#   1) ./scripts/smoke_test_api.sh up to the GET step (capture session id), Ctrl+C the smoke before DELETE
#   2) Ctrl+C the backend
#   3) ./scripts/run_backend.sh again
#   4) curl http://localhost:8000/sessions/<id>   → record still there, status auto-reconciled

# Backend unit tests (no docker / no GPU required)
cd backend && pip install -e ".[dev]" && pytest

# 5-A correlation extension — launch counter ↔ nvidia-smi memory time-series
# (PyTorch image gained test_compute.py — rebuild it; backend gained
#  FGPU_LAUNCH_LOG_EVERY env passthrough)
./scripts/build_pytorch_image.sh
FGPU_LAUNCH_LOG_EVERY=500 ./scripts/run_backend.sh   # backend env propagates to spawned containers
# in another terminal:
./scripts/eval/run_correlation.sh                    # → experiments/correlation_<TS>/correlation.csv
ALLOC_MIB=4096 ./scripts/eval/run_correlation.sh     # OOM scenario — A 가 OOM, B 만 trace 채워짐
RATIO_A=0.3 RATIO_B=0.7 HOLD_SEC=15 ./scripts/eval/run_correlation.sh

# Stage 6 — VMM API hook (cuMemCreate / cuMemRelease)
# (hook/src and runtime-image both changed — rebuild .so AND base image)
./scripts/build_hook.sh                # libfgpu.so with VMM hooks
./scripts/build_image.sh               # /opt/fgpu/test_vmm_alloc baked in
./scripts/run_vmm_in_container.sh                  # baseline + hooked, default ratio 0.4
FGPU_RATIO=0.6 ./scripts/run_vmm_in_container.sh   # 6 GiB still DENY (quota 4.8 GiB)

# Stage 9 minimal — bearer auth + multi-GPU device pinning
# (backend code only — uvicorn --reload picks it up)
FGPU_API_TOKEN=secret-dev-token ./scripts/run_backend.sh
# auth on:
curl -X POST http://localhost:8000/sessions \
    -H 'Authorization: Bearer secret-dev-token' \
    -H 'Content-Type: application/json' \
    -d '{"ratio": 0.4}'
# multi-GPU (only useful on a host with >1 GPU):
curl -X POST http://localhost:8000/sessions \
    -H 'Authorization: Bearer secret-dev-token' \
    -H 'Content-Type: application/json' \
    -d '{"ratio": 0.4, "gpu_index": 1}'
# auth off (default — FGPU_API_TOKEN unset):
./scripts/run_backend.sh
./scripts/smoke_test_api.sh                        # works as before, no auth header
```

`CUDA_HOME` (host build) defaults to `/usr/local/cuda`; `CUDA_VERSION` (image build) defaults to `12.4.1`. Keep host CUDA major version aligned with the image's CUDA major version — the host-built `libfgpu.so` is mounted into the container and dynamically links against the container's `libcudart`.

The backend process itself does **not** need GPU access — it only talks to the docker socket. The user spawning it must be in the `docker` group.

**Persistence (Stage 8)**: session records live in `<repo>/data/sessions.db` (SQLite). `data/` is gitignored. To wipe state (e.g., after schema-incompatible changes during development), `rm -rf data/`. Containers themselves are *not* affected by deleting the DB — they keep running under docker daemon, just orphaned from the backend's view. Use `docker ps | grep fgpu-` to find them and `docker rm -f` to clean up.

### Stage 1 success criteria

The hooked run's stderr must contain, in order: `[fgpu] init`, `[fgpu] quota lazily 계산`, an `ALLOW` line for the 256 MiB alloc, a `DENY` line for the 6 GiB alloc with `err=2` propagated to the test program, then `FREE` lines that bring `used` back to 0.

### Stage 2 success criteria

`scripts/run_in_container.sh` must show the same `[fgpu] init` / `quota lazily 계산` / `ALLOW` / `DENY` / `FREE` sequence, prefixed by `[entrypoint]` lines confirming env + hook mount. Baseline run (no `-v` mount, no `-e LD_PRELOAD`) must NOT show any `[fgpu]` lines and must succeed for both 256 MiB and 6 GiB allocs (assuming GPU has free memory).

### Stage 3 success criteria

`scripts/smoke_test_api.sh` completes end-to-end. The returned `Session` JSON has a non-empty `container_id`; after `sleep 4` the `status` is `exited` with `exit_code: 0`; the `logs` field contains the `[entrypoint]` + `[fgpu]` ALLOW/DENY/FREE lines from Stage 2; `DELETE` returns `{"deleted": "<id>"}` and a follow-up `GET` would return 404.

### Stage 4 success criteria

`scripts/run_pytorch_in_container.sh` produces three observable runs (default sizes 256 MiB + 4 GiB on RTX 4060 / 8 GB):

- **Baseline** (no hook): both `[pytorch-test] OK` lines, no `[fgpu]` lines.
- **`FGPU_RATIO=0.4`** (quota ≈ 3.2 GiB): 256 MiB → `OK`; 4 GiB → `OOM ← cudaErrorMemoryAllocation 이 PyTorch 까지 전파됨`. Hook log shows one `ALLOW` (256 MiB) and one `DENY` (4 GiB) plus matching `FREE` lines.
- **`FGPU_RATIO=0.6`** (quota ≈ 4.8 GiB): both allocations `OK`. Hook log shows two `ALLOW` lines.

The OOM path proves the `cudaErrorMemoryAllocation` returned by the hook propagates all the way through `libcudart` → PyTorch's `CUDACachingAllocator` → Python `torch.cuda.OutOfMemoryError`. If `PYTORCH_NO_CUDA_MEMORY_CACHING=1` is dropped, the caching allocator's first big slab will mask the per-call quota — this is expected behavior, documented as a known limitation, and is the motivation for the Stage 6+ Driver API hook.

### Stage 5-A success criteria

`scripts/eval/run_isolation.sh` produces an `experiments/isolation_<TS>/` directory whose `summary.txt` reports `VERDICT: PASS` for the default scenario (`RATIO_A=0.4`, `RATIO_B=0.6`, `ALLOC_MIB=4096`, `HOLD_SEC=6`). The PASS condition is:

- `session_a.json.exit_code == 1`, `container_a.log` contains `[hold-test] OOM` and `[fgpu] DENY`.
- `session_b.json.exit_code == 0`, `container_b.log` contains `[hold-test] OK` and `[fgpu] ALLOW`.
- `nvidia_smi.csv` is non-empty and contains rows during the overlap window — independent ground-truth that B held ~4 GiB while A was active.

What the experiment proves
- Two processes running concurrently each see *their own* `FGPU_QUOTA_BYTES` and counter (the hook state is per-process by design — `LD_PRELOAD` injects a fresh instance per container).
- The same 4 GiB workload succeeds or fails *only* because of the ratio difference, with everything else held constant.

What it does NOT prove
- SM-level isolation. Both containers contend for the same SMs; we measure memory quota only.
- Quota correctness when PyTorch's caching allocator is on (`PYTORCH_NO_CUDA_MEMORY_CACHING=1` is required).
- Resistance to a non-cooperative tenant (statically linked binary, direct `dlopen` of cudart). Cooperative threat model only.

### Stage 5-B success criteria

`./scripts/run_backend.sh` boots and `GET http://localhost:8000/` returns the UI HTML (200 OK, content type `text/html`). In the browser:

- The "Create" form, with default values (`ratio=0.4`, image `fgpu-runtime-pytorch:stage4`, command `python3 /opt/fgpu/test_hold.py`), produces a new row in the Sessions table within ~1 second of submit.
- The new row's `status` transitions `created` → `running` → `exited` over the auto-refresh polls (3 s interval), with `exit_code` populated on terminal state.
- Clicking the row populates the Logs pane with `[entrypoint]` + `[fgpu]` + `[hold-test]` lines from the container's stdout/stderr.
- `stop` and `delete` buttons act on `POST /sessions/{id}/stop` and `DELETE /sessions/{id}` respectively; the row disappears after delete.

What it deliberately leaves out
- Authentication, multi-user accounts.
- WebSocket-based live log streaming (just polling).
- A frontend build pipeline (no React/Vue/etc.) — capstone scope only.

### Stage 5-C success criteria

`scripts/run_driver_in_container.sh` produces two runs:

- **Baseline** (no hook): `[test-driver]` lines for both 256 MiB and 6 GiB. The 256 MiB call returns `CUDA_SUCCESS`; the 6 GiB call may succeed or fail depending on free GPU memory — either is acceptable, since "no hook" means GPU's natural ceiling. **No `[fgpu]` lines.**
- **Hooked** (`FGPU_RATIO=0.4`, quota ≈ 3.2 GiB):
  - stderr contains `[fgpu] init: real cuMemAlloc_v2=0x...` (non-NULL pointer — driver symbol resolved).
  - stderr contains `[fgpu] ALLOW cuMemAlloc_v2 ptr=0x... size=268435456 ...` (256 MiB).
  - stderr contains `[fgpu] DENY  cuMemAlloc_v2 size=6442450944 used=... quota=...` (6 GiB).
  - stdout's `[test-driver]` line for the 6 GiB attempt reports `result=2 (CUDA_ERROR_OUT_OF_MEMORY)` — the driver-layer error code propagated to the caller.

What the experiment proves
- The hook's `cuMemAlloc_v2`/`cuMemFree_v2` interception works end-to-end: a program that touches **only** Driver API (no `cudaMalloc`) is still subject to the per-process quota.
- The reentrancy guard does its job — the `[fgpu] init` line shows both Runtime and Driver symbols resolved without spurious errors, and there's no double-`ALLOW` for a single user-level allocation.

What it does NOT prove
- That every modern PyTorch path goes through the now-also-hooked driver layer. PyTorch with caching off still hits `cudaMalloc` (Runtime); the driver hook's value is mainly for *direct-driver-API* tenants (custom CUDA kernels, JAX, hand-written `cuMemAlloc_v2` callers).
- Coverage of `cuMemAllocAsync`, `cuMemAllocManaged`, or VMM API (`cuMemCreate`/`cuMemMap`). Those remain Stage 6+.

### Stage 8 success criteria

**Persistence**:
1. First boot: `./scripts/run_backend.sh` creates `data/sessions.db` (visible via `ls data/`).
2. Create a session via `./scripts/smoke_test_api.sh` (or curl). `sqlite3 data/sessions.db "SELECT id, status FROM sessions"` shows the row.
3. Stop the backend (Ctrl+C). Restart with `./scripts/run_backend.sh`.
4. `curl http://localhost:8000/sessions` returns the same session record (status auto-reconciled from docker daemon — should be `exited` if the test_alloc workload finished).
5. The Web UI (`http://localhost:8000/`) shows the session row immediately on page load (no need to recreate).

**Async wrapping (concurrency)**:
- Two simultaneous `POST /sessions` requests get processed *in parallel* — neither blocks on the other's docker SDK call. `scripts/eval/run_isolation.sh` is a natural test bed; the two `post_session` curls now overlap in time at the API layer too, not just at the workload layer.
- The healthz endpoint stays responsive while a heavy `POST /sessions` is in-flight.

What it proves
- Session records survive backend restart — backend is no longer a stateful single-point-of-failure for session metadata. Future Stage 9 (multi-host) can replace SQLite with Redis/Postgres without changing SessionManager's contract.
- Backend is now multi-tenant safe at the API layer (event loop never blocks on docker SDK).

What it does NOT prove
- True high-availability — single uvicorn process, single SQLite file. Multi-process / multi-host requires a network DB.
- Container reconciliation correctness across all edge cases. Examples not covered: container removed by `docker rm -f` outside the backend (next GET marks it `removed` but doesn't delete the row); docker daemon down (raises an exception, propagates as 500). Document as known.

### Stage 7 success criteria

`scripts/run_launch_in_container.sh` produces two runs:

- **Baseline** (no hook): `[test-launch]` lines (`launching N kernels ...`, `launches done, kernel atomics = N`). **No `[fgpu]` lines.**
- **Hooked** (`FGPU_LAUNCH_LOG_EVERY=100`, `PYTEST_LAUNCH_N=1000`):
  - stderr contains `[fgpu] init: ... cudaLaunchKernel=0x...` (non-NULL — symbol resolved).
  - stderr contains exactly 10 lines `[fgpu] LAUNCH count={100,200,...,1000} (every 100)`.
  - stderr contains `[fgpu] exit summary: total cudaLaunchKernel = 1000` after the binary exits cleanly (atexit fires).
  - stdout `[test-launch] kernel atomics = 1000` confirms the kernels actually executed (the hook didn't drop calls).

What the experiment proves
- `cudaLaunchKernel` is reliably intercepted and counted, lock-free, with sub-microsecond overhead per call.
- The counter is monotonic and accurate across thousands of launches (the kernel atomic counter equals `N`, so no launches were silently dropped).
- The hook can be turned off (`FGPU_LAUNCH_LOG_EVERY=0`) for clean overhead measurement runs without losing the count internally.

What it does NOT prove
- That launch *count* correlates one-to-one with GPU device time. A kernel running 100 ms once vs 1 µs a thousand times both look very different on the GPU but our counter only sees "1" vs "1000". Real device-time measurement requires `cudaEventRecord` injection, which is intentionally out of scope (would force per-launch synchronization or a polling thread).
- Driver API path (`cuLaunchKernel` from libcuda) — not hooked. Frameworks bypassing cudart's `cudaLaunchKernel` (rare for PyTorch, possible for hand-rolled CUDA C++ via driver API) would not be counted.

### Stage 5-D success criteria

`scripts/eval/run_overhead.sh` produces `experiments/overhead_<TS>/summary.csv` with one row per `(size_mib, mode)` pair, where:

- `baseline_raw.csv` and `hooked_raw.csv` each contain `BENCH_N × len(BENCH_SIZES_MIB)` data rows (default 100 × 4 = 400) plus a header.
- `summary.txt` renders two markdown tables (`cudaMalloc` and `cudaFree`) with mean / p50 / p99 in microseconds, and a `Δ mean %` column quantifying the hooked overhead.

What the experiment proves
- Per-call `cudaMalloc` / `cudaFree` latency overhead of the LD_PRELOAD hook is bounded and reproducible across allocation sizes (the paper's headline overhead number).
- The overhead is dominated by `pthread_mutex_lock` + `dlsym`-resolved indirect call + `fprintf(stderr)` — not by quota arithmetic. Larger sizes amortize the per-call constant.

What it does NOT prove
- End-to-end application-level overhead (PyTorch caching allocator hides per-call cost; `bench_alloc` deliberately bypasses it).
- Driver API path overhead (Stage 6+).
- Performance under contention (this is single-threaded, single-process; multi-tenant overhead is Stage 5-A's data, not this one's).

### 5-A correlation extension success criteria

`scripts/eval/run_correlation.sh` produces `experiments/correlation_<TS>/correlation.csv` with the columns `t_seconds, container, launch_count, used_memory_mib`. Default scenario (`RATIO_A=0.4`, `RATIO_B=0.6`, `ALLOC_MIB=2048`, `HOLD_SEC=10`):

- `correlation_summary.txt` reports a non-zero `final cumulative count` for both A and B (Stage 7 hook is firing under PyTorch).
- `peak used memory (MiB)` is roughly `ALLOC_MIB` for each container (PyTorch alloc went through the hook and was counted by nvidia-smi).
- Both containers have non-empty PID sets in `pids_a.txt` / `pids_b.txt`, and those PIDs match rows in `nvidia_smi.csv` (PID join is intact).
- The CSV's `t_seconds` axis spans roughly `HOLD_SEC + start-up overhead`, with launch_count rows interleaved with used_memory rows for both containers — plot-ready.

What this proves (paper graph material)
- Two containers under different ratios coexist on a single GPU and we can simultaneously observe (a) memory share via `nvidia-smi`, (b) compute activity via the hook's `cudaLaunchKernel` counter. Same workload, different ratios → comparable launch frequencies but bounded memory.
- The Stage 7 launch counter actually advances during PyTorch compute (cuBLAS matmul + element-wise ops trigger many `cudaLaunchKernel` calls per iteration).

What this does NOT prove
- Direct device-time correlation. Launch *count* ≠ device-time. Heavier kernels would register the same way as lighter ones in our counter.
- That the GPU is being SM-isolated. It isn't. Both containers contend on the same SMs; we just observe the contention via two metrics.
- Coverage of `cuLaunchKernel` (driver API) — Stage 7+ future work.

### Stage 6 success criteria

`scripts/run_vmm_in_container.sh` produces two runs:

- **Baseline** (no hook): `[test-vmm]` lines for both 256 MiB and 6 GiB. The 256 MiB call returns `CUDA_SUCCESS`; the 6 GiB call may succeed or fail depending on free GPU memory + driver constraints — either is acceptable. **No `[fgpu]` lines.**
- **Hooked** (`FGPU_RATIO=0.4`, quota ≈ 3.2 GiB):
  - stderr contains `[fgpu] init: real cuMemCreate=0x... cuMemRelease=0x...` (non-NULL — VMM symbols resolved).
  - stderr contains `[fgpu] ALLOW cuMemCreate handle=0x... size=268435456 ...` (256 MiB).
  - stderr contains `[fgpu] DENY  cuMemCreate size=6442450944 used=... quota=...` (6 GiB).
  - stdout's `[test-vmm]` line for the 6 GiB attempt reports `result=2 (CUDA_ERROR_OUT_OF_MEMORY)`.

What this proves
- The hook's `cuMemCreate` / `cuMemRelease` interception works on programs that touch *only* VMM API.
- The reentrancy guard is sound — VMM hook coexists with Runtime + Driver-classic hooks on the same `g_used` / `g_quota` state without double counts.

What it does NOT prove
- Coverage of `cuMemAllocAsync` (CUDA 11.2+ stream-ordered alloc) or `cuMemAllocManaged` (UVM). Those remain unhooked — Stage 6+ continuing work.
- That PyTorch under default caching uses VMM. Empirically PyTorch's caching allocator goes through `cudaMalloc` → cudart's classical path, so this hook's value is mainly for direct VMM users (custom CUDA, memory pool libraries built on the modern API).

### Stage 9 minimal success criteria

**Auth off (default)** — `FGPU_API_TOKEN` unset:
- `GET /healthz` returns `auth_enabled: false`.
- `scripts/smoke_test_api.sh` passes unchanged (no Authorization header needed).

**Auth on** — `FGPU_API_TOKEN=secret ./scripts/run_backend.sh`:
- `GET /healthz` returns `auth_enabled: true`. (`/healthz` itself remains public.)
- `GET /` (UI) returns 200 unauthenticated.
- `POST /sessions` without `Authorization` header → 401 `{"detail":"missing bearer token"}` with `WWW-Authenticate: Bearer`.
- `POST /sessions` with `Authorization: Bearer wrong` → 401 `{"detail":"invalid bearer token"}`.
- `POST /sessions` with `Authorization: Bearer secret` → 201 normal create.

**Multi-GPU** (only meaningful on a host with ≥2 GPUs):
- `POST /sessions` with `{"ratio": 0.4, "gpu_index": 1}` spawns the container with `--gpus device=1`. Inside the container `nvidia-smi -L` lists only GPU 1.
- `gpu_index` is preserved across backend restart (persisted in SQLite via the Stage 8 store).

What it does NOT prove
- Multi-host scheduling. Single uvicorn process, no inter-host coordination — that's full Stage 9.
- Token rotation, refresh, RBAC. Single static token only.
- Rate limiting / abuse prevention. Out of scope.

## Hard constraints — do not propose around these

- **No MIG.** Target hardware is RTX 4060 (consumer card, MIG unsupported). MIG is mentioned only as a comparison baseline in the paper, never as a code path.
- **VMM API hook = `cuMemCreate`/`cuMemRelease` only (Stage 6).** `cuMemAddressReserve`/`cuMemMap`/`cuMemUnmap`/`cuMemAddressFree` are intentionally not hooked (they don't change physical memory). `cuMemAllocAsync` and `cuMemAllocManaged` remain unhooked — Stage 6+ continuing work.
- **No SM isolation.** Time-slicing via `cudaLaunchKernel` interception is a planned stage (3+ in the hook roadmap), but quota = memory only for now.
- **Cooperative threat model.** Statically linked binaries can bypass the hook; this is documented as a limitation, not a bug to fix. (Note: with Stage 5-C the surface for `dlopen("libcuda.so")` users is closed too — those calls now hit the driver hook.)
- **PyTorch caching allocator** masks per-call quota effects. When testing PyTorch integration (Stage 4), set `PYTORCH_NO_CUDA_MEMORY_CACHING=1`. Stage 5-C does not change this — the caching allocator's *one big slab* is still seen by both layers as a single allocation.

## Coding conventions

- Hook code is C (not C++) so the symbol table stays simple and `extern "C"` issues never come up. Keep it that way.
- Functions whose name ends in `_locked` assume the caller already holds `g_lock`. Honor the convention; don't lock twice.
- All `[fgpu] ...` log lines go to **stderr** (so they don't pollute the user program's stdout). Keep the `[fgpu]` prefix — paper screenshots and grep both depend on it.
- Korean comments for pedagogical files (the hook, description.md). English is fine for code Claude writes for itself.
