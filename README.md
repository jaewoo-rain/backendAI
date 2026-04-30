# fGPU Prototype

A capstone / research prototype that mimics
[Backend.AI](https://www.backend.ai/)'s **fractional GPU (fGPU)** capability:
a single NVIDIA GPU is shared by multiple Docker containers, each receiving
a fractional quota (e.g. `0.4` / `0.6`) of GPU memory.

The mechanism is **`LD_PRELOAD`-based CUDA API hooking** — `libfgpu.so`
is injected into each container, intercepts CUDA memory allocation calls
across three layers (**Runtime** `cudaMalloc`/`cudaFree`,
**Driver-classic** `cuMemAlloc_v2`/`cuMemFree_v2`, **VMM**
`cuMemCreate`/`cuMemRelease`) sharing one per-process quota state, and
also intercepts `cudaLaunchKernel` for *monitoring* (launch count) —
not enforcement.

> **This is not a production GPU virtualizer.** SM-level / hardware
> isolation is out of scope; the target hardware (RTX 4060) does not
> support MIG. Treat this as "cooperative quota enforcement at the CUDA
> API boundary, plus a thin FastAPI session manager around it."

---

## Architecture

```
                   user (curl / minimal Web UI)
                                │ HTTP
                                ▼
                    FastAPI backend (:8000)
                       /sessions REST + SQLite persistence
                                │ Docker SDK
                                ▼
                    Docker + nvidia-container-runtime
                                │ docker run --gpus all
                                ▼
                       per-user containers
                       LD_PRELOAD=/opt/fgpu/libfgpu.so
                       FGPU_RATIO=<0..1>
                                │
                       NVIDIA driver → GPU
```

Two key design decisions:

1. **Backend and hook are decoupled.** The backend only delivers the
   quota value as a container env var; the hook enforces it inside the
   container. Swapping the backend for a multi-host scheduler (k8s, etc.)
   does not require touching the hook.
2. **Hook is per-container.** No cross-container communication. State
   (`g_used`, `g_quota`, allocation list, lock, launch counter) is
   process-local. `LD_PRELOAD` injects a fresh hook instance per
   container.

---

## Hardware / OS prerequisites

| Component | Tested on |
|---|---|
| GPU | NVIDIA RTX 4060 (8 GB) — any modern NVIDIA GPU should work |
| OS | Ubuntu 22.04 (native or WSL2) |
| NVIDIA driver | 535+ |
| CUDA toolkit | 12.4 (host); 12.1 (PyTorch wheel) |
| Docker | 24+ with `nvidia-container-toolkit` |
| Python | 3.11+ |

> **Windows users:** native Windows is **not** supported (LD_PRELOAD is
> Linux-only). Use WSL2 with Ubuntu 22.04. NVIDIA officially supports
> CUDA-on-WSL2; Docker Desktop's WSL2 backend or `docker-ce` installed
> inside WSL2 both work.

---

## Quick start

```bash
# 1) build the hook .so
./scripts/build_hook.sh                 # → build/libfgpu.so

# 2) build the runtime base image (compiles bench + smoke binaries)
./scripts/build_image.sh                # → fgpu-runtime:stage2

# 3) build the PyTorch variant (~5 GB wheel pull on first run)
./scripts/build_pytorch_image.sh        # → fgpu-runtime-pytorch:stage4

# 4) run the backend
./scripts/run_backend.sh                # uvicorn on :8000
# open http://localhost:8000/  in a browser

# 5) sanity-check via curl
./scripts/smoke_test_api.sh             # POST → GET → logs → DELETE
```

The bundled web UI at `http://localhost:8000/` lets you create / inspect /
delete sessions interactively.

---

## What's implemented

| Stage | Deliverable |
|---|---|
| 1 | Runtime API hook — `cudaMalloc`, `cudaFree`. Quota lazy-computed from `cudaMemGetInfo` × ratio |
| 2 | Containerized runtime image, hook mounted at runtime (not baked) |
| 3 | FastAPI backend — `/sessions` REST CRUD, status auto-reconcile from docker daemon |
| 4 | PyTorch integration verified (`PYTORCH_NO_CUDA_MEMORY_CACHING=1`) |
| 5-A | Concurrent isolation experiment automation (two containers, different ratios, nvidia-smi capture, PASS/FAIL verdict) |
| 5-B | Minimal vanilla-JS web UI |
| 5-C | Driver API hook — `cuMemAlloc_v2`, `cuMemFree_v2`. Reentrancy guard prevents double-counting |
| 5-D | Overhead microbenchmark — `cudaMalloc` / `cudaFree` latency table (mean / p50 / p99) |
| 7 | `cudaLaunchKernel` hook — lock-free launch counter, periodic + atexit dumps |
| 8 | SQLite session persistence + `asyncio.to_thread` wrapping of all blocking docker SDK / sqlite calls |
| 5-A ext | Launch counter ↔ `nvidia-smi` memory time-series correlation experiment |
| 6 | VMM API hook — `cuMemCreate` / `cuMemRelease` (modern allocation path) |
| 9 minimal | Bearer token auth (`FGPU_API_TOKEN`) + multi-GPU device pinning (`gpu_index`) |

---

## Limitations (also detailed in `description.md`)

- **No SM isolation.** Hooks intercept memory APIs and kernel launches
  but cannot bound *compute time*. Two containers contend on the same
  SMs; we measure but don't enforce. SM isolation requires MIG (data
  center GPUs only) or MPS (single-tenant only).
- **Cooperative threat model.** Statically linked binaries
  (`nvcc -cudart=static`) bypass `LD_PRELOAD`. Stage 5-C closes the
  `dlopen("libcuda.so")` surface but not the static-link one. This is
  documented as an out-of-scope limitation, not a bug.
- **PyTorch caching allocator masks per-call quota.** When PyTorch's
  caching is on (default), the first big slab is the only real
  `cudaMalloc`, so per-tensor quota effects vanish. Set
  `PYTORCH_NO_CUDA_MEMORY_CACHING=1` (the PyTorch image does this by
  default) to get accurate per-allocation observation.
- **Driver/VMM API coverage is partial.** `cuMemAlloc_v2`/`cuMemFree_v2`
  (Stage 5-C) and `cuMemCreate`/`cuMemRelease` (Stage 6) are hooked,
  but `cuMemAllocAsync` (stream-ordered) and `cuMemAllocManaged` (UVM)
  are not. Workloads that go exclusively through those paths bypass the
  quota.
- **Single host.** SQLite store is single-node. `SessionStore` is the
  abstraction boundary — replacing it with Redis/Postgres for
  multi-host is the natural Stage 9 (full) direction; `SessionManager`
  would not need to change. Stage 9 minimal in this prototype only
  covers single-host bearer auth + per-session GPU device pinning.

---

## Reproducing the paper data

Each experiment writes its artifacts under `experiments/<name>_<timestamp>/`.
`experiments/` is git-ignored.

```bash
# memory quota — host-only (Stage 1)
./scripts/run_test.sh                                  # baseline + hooked

# memory quota — in container (Stage 2)
./scripts/run_in_container.sh

# memory quota — Driver API only (Stage 5-C)
./scripts/run_driver_in_container.sh

# memory quota — PyTorch (Stage 4)
./scripts/run_pytorch_in_container.sh
FGPU_RATIO=0.6 ./scripts/run_pytorch_in_container.sh

# launch counter (Stage 7)
./scripts/run_launch_in_container.sh

# VMM API quota (Stage 6)
./scripts/run_vmm_in_container.sh

# overhead microbench (Stage 5-D)
./scripts/build_image.sh
./scripts/eval/run_overhead.sh
# → experiments/overhead_<TS>/summary.{csv,txt}

# concurrent isolation (Stage 5-A) — backend must be running
./scripts/run_backend.sh &
./scripts/eval/run_isolation.sh
# → experiments/isolation_<TS>/summary.txt + nvidia-smi CSV

# launch ↔ memory time-series correlation (5-A extension)
FGPU_LAUNCH_LOG_EVERY=500 ./scripts/run_backend.sh &
./scripts/eval/run_correlation.sh
# → experiments/correlation_<TS>/correlation.csv
```

The correlation CSV is long-format (`t_seconds, container, launch_count,
used_memory_mib`); pivot in pandas / Excel for plotting. A pandas pivot
example is printed at the end of `run_correlation.sh`.

---

## Repository layout

See [`CLAUDE.md`](CLAUDE.md) for an authoritative file-by-file index, and
[`description.md`](description.md) (Korean) for the full architectural
rationale, design alternatives, and limitation analysis intended as the
paper's source material.

Top-level:

```
hook/             LD_PRELOAD hook source + smoke / bench binaries (CUDA C/C++)
runtime-image/    Base Docker image (CUDA devel, smoke binaries pre-compiled)
runtime-image-pytorch/  PyTorch variant on top of the base
backend/          FastAPI session manager (Python)
scripts/          Build + run + evaluation drivers
scripts/eval/     Paper-relevant experiment automation (isolation, overhead,
                  correlation)
experiments/      (gitignored) experiment outputs
build/            (gitignored) built artifacts
data/             (gitignored) SQLite session store
```

---

## Tests

A small pytest suite covers the Stage 8 SQLite session store
(`backend/tests/test_session_store.py`). It does **not** require docker
or a GPU.

```bash
cd backend
pip install -e ".[dev]"
pytest
```

The full backend → docker → GPU integration path is exercised by the
shell scripts under `scripts/eval/` (run on actual hardware).

---

## Authentication (optional)

By default the API runs unauthenticated for ease of development. To turn
on bearer auth, set `FGPU_API_TOKEN` before starting the backend:

```bash
FGPU_API_TOKEN=secret-dev-token ./scripts/run_backend.sh

# requests must now include Authorization
curl -X POST http://localhost:8000/sessions \
    -H 'Authorization: Bearer secret-dev-token' \
    -H 'Content-Type: application/json' \
    -d '{"ratio": 0.4}'
```

`/healthz` and `/` (UI) remain public so health checks and demos still
work. Token comparison uses `hmac.compare_digest` to defeat timing
attacks. There is no token rotation, refresh, or RBAC — that's full
Stage 9 future work.

## Multi-GPU

`SessionCreate` accepts an optional `gpu_index` field. `None` (default)
exposes every GPU to the container; an integer pins it to that device:

```bash
curl -X POST http://localhost:8000/sessions \
    -H 'Content-Type: application/json' \
    -d '{"ratio": 0.4, "gpu_index": 1}'
```

The hook inside the container then sees only the pinned GPU and computes
quota against its memory total.

---

## Status

Capstone / research prototype. Stages 1 through 8, the 5-A correlation
extension, Stage 6 (VMM API hook), and Stage 9 minimal (bearer auth +
multi-GPU device pinning) are implemented and self-verified. Stage 9
full (Kubernetes scheduler, Redis store, RBAC) and additional VMM paths
(`cuMemAllocAsync`, `cuMemAllocManaged`) are out of scope for the
current write-up but the structure is intentionally extensible toward
them.

For questions about the design, start with `description.md`. For build
details and per-stage acceptance criteria, see `CLAUDE.md`.

---

## License

MIT — see [`LICENSE`](LICENSE). © 2026 양재우 (Jaewoo Yang).
