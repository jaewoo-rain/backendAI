# fGPU 프로토타입 — 기술 설명서

> 이 문서는 본 프로젝트가 *무엇을* 만들고 있고 *왜* 이렇게 만드는지를 풀어 쓴 기술 설명서입니다. 코드만 봐서는 알기 어려운 설계 의도, 대안 비교, 한계, 평가 계획까지 한 곳에 모았습니다. 캡스톤 발표 자료와 논문 본문의 1차 원천으로 사용합니다.

---

## 1. 프로젝트 한 줄 정의

**"CUDA API Hooking 기반 fractional GPU resource control prototype"** — 단일 NVIDIA GPU를 여러 Docker 컨테이너가 0.4 / 0.6 같은 비율로 *논리적으로* 나눠 쓰는 것처럼 보이게 만드는 연구 프로토타입.

목적은 (a) Backend.AI fGPU의 핵심 메커니즘을 오픈소스로 재현하고, (b) 컨슈머 GPU 환경에서 그 메커니즘이 어디까지 동작하고 어디서 깨지는지를 정량적으로 측정하는 것.

---

## 2. 왜 만드는가 — 배경

### 2.1 GPU 가상화 시장의 빈틈

- **데이터센터 GPU (A100, H100)** 는 NVIDIA MIG (Multi-Instance GPU) 로 하드웨어 수준에서 GPU를 7개까지 분할할 수 있다. 메모리, SM, L2 캐시 모두 격리된다.
- **컨슈머 GPU (RTX 30/40 시리즈)** 는 MIG 미지원. NVIDIA MPS (Multi-Process Service) 는 같은 process pool 안에서만 동작하므로 멀티테넌트 컨테이너 시나리오엔 부적합.
- 결과: "한 장의 컨슈머 GPU를 여러 사용자에게 fractional로 나눠주려면" 표준 답이 없다. → API hooking 기반의 *소프트웨어 가상화* 가 사실상 유일한 선택지.

### 2.2 Backend.AI의 fGPU

Lablup의 Backend.AI는 fGPU(fractional GPU) 기능을 통해 단일 GPU를 0.5, 0.25 같은 단위로 잘라 컨테이너에 부여한다. 핵심 메커니즘은 **컨테이너에 자체 hook 라이브러리를 주입해 CUDA API 호출을 가로채는** 것으로 알려져 있다 (정확한 구현은 비공개).

본 프로젝트는 그 아이디어를 오픈소스로, 그리고 학습/논문 목적에 맞게 *최소한의 형태로* 재현한다.

### 2.3 본 프로젝트와 Backend.AI의 차이

| 항목 | Backend.AI fGPU (상용) | 본 프로토타입 |
|---|---|---|
| 격리 | 메모리 + 연산 (시간 분할) | **메모리 quota** + **launch frequency 측정** (시행은 메모리만) |
| 후킹 계층 | Runtime + Driver + 자체 allocator | Runtime + Driver (`cuMemAlloc_v2`) + VMM (`cuMemCreate`) + `cudaLaunchKernel` 카운터 |
| 스케줄러 | Sokovan, 멀티 노드 | 단일 노드 단순 매니저 (SQLite persistence, 멀티-GPU device pinning) |
| 사용자 시스템 | RBAC, 과금, 키페어 | bearer token 1개 (Stage 9 minimal) — RBAC/과금/키페어는 후속 |
| 코드 공개 | 핵심 fGPU 로직 비공개 | 전부 공개 (MIT) |
| 목적 | 프로덕션 멀티테넌트 | 캡스톤 / 논문 PoC |

---

## 3. 전체 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│  사용자 (curl / 최소 Web UI)                                  │
└──────────────┬───────────────────────────────────────────────┘
               │ HTTP
┌──────────────▼───────────────────────────────────────────────┐
│  FastAPI Backend (Stage 3 + 5-B UI + Stage 8 persistence)    │
│   ├─ /sessions REST + GET /  (vanilla-JS UI)                 │
│   ├─ Session Manager  (asyncio.to_thread 으로 docker SDK 호출)│
│   ├─ Session Store    (SQLite — data/sessions.db)            │
│   └─ Docker Manager   (env passthrough whitelist 포함)       │
└──────────────┬───────────────────────────────────────────────┘
               │ Docker SDK
┌──────────────▼───────────────────────────────────────────────┐
│  Docker + nvidia-container-runtime                           │
│   docker run --gpus all                                      │
│     -e FGPU_RATIO=0.4                                        │
│     -e LD_PRELOAD=/opt/fgpu/libfgpu.so                       │
│     -e FGPU_LAUNCH_LOG_EVERY=...   (passthrough)             │
│     -v <host>/build/libfgpu.so:/opt/fgpu/libfgpu.so:ro       │
│     <runtime-image | runtime-image-pytorch>                  │
└──────────────┬───────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────┐
│  사용자 컨테이너 (PyTorch, cuBLAS, custom CUDA, ...)         │
│    [LD_PRELOAD] libfgpu.so                                   │
│       ├─ cudaMalloc / cudaFree           (Stage 1, Runtime)  │
│       ├─ cuMemAlloc_v2 / cuMemFree_v2    (Stage 5-C, Driver) │
│       ├─ cuMemCreate / cuMemRelease       (Stage 6, VMM)     │
│       └─ cudaLaunchKernel  (Stage 7 — counter only, no enf.) │
│   reentrancy guard (__thread g_in_hook) 으로 세 alloc layer  │
│   가 같은 g_used / g_quota / g_allocs 공유 시 이중 카운트    │
│   방지. VMM 의 cuMemAddressReserve/cuMemMap 은 *물리 변화 X*  │
│   라 의도적으로 후킹 안 함.                                  │
└──────────────┬───────────────────────────────────────────────┘
               │
        NVIDIA Driver → RTX 4060
```

설계 핵심 두 가지:

1. **백엔드와 hook은 분리.** 백엔드는 quota 값을 *환경변수로만* 전달하고, hook은 컨테이너 안에서 자체적으로 quota를 시행한다. 이렇게 해야 나중에 멀티노드/Kubernetes로 갈 때 백엔드만 갈아끼우면 된다.
2. **hook은 컨테이너별로 독립.** 컨테이너 간 통신은 0. 모든 정책은 컨테이너 시작 시점에 env로 주입. 향후 메트릭 수집이 필요하면 hook → 호스트 unix socket 으로 단방향 push만 추가.

---

## 4. LD_PRELOAD CUDA Hooking 동작 원리 (교육용)

### 4.1 일반적인 cudaMalloc 호출의 흐름

```
[사용자 코드]    → cudaMalloc(&p, N)
                       ↓ (동적 링커가 cudart의 cudaMalloc 으로 점프)
[libcudart.so] → 실제 GPU 메모리 할당
```

### 4.2 LD_PRELOAD가 끼어든 흐름

```
환경변수: LD_PRELOAD=/opt/fgpu/libfgpu.so

[사용자 코드]    → cudaMalloc(&p, N)
                       ↓ (libfgpu.so가 먼저 로드됐으므로 우리 cudaMalloc 으로 점프)
[libfgpu.so]    → quota 검사
                  ├─ 초과: cudaErrorMemoryAllocation 반환 (cudart 호출 안 함)
                  └─ 통과: dlsym(RTLD_NEXT,"cudaMalloc") 으로 진짜 함수 호출
                              ↓
                     [libcudart.so] → 실제 GPU 메모리 할당
```

핵심 트릭은 두 단계:
- **이름 가로채기**: 우리 .so 에 `cudaMalloc` 이라는 같은 이름의 함수가 있으면 동적 링커가 *그쪽* 으로 점프한다.
- **진짜 함수 호출**: `dlsym(RTLD_NEXT, "cudaMalloc")` 은 "내 다음 라이브러리에서 같은 이름을 찾아라" 를 의미하므로, cudart 의 진짜 `cudaMalloc` 주소를 돌려준다.

### 4.3 왜 LD_PRELOAD인가 — 대안 비교

| 방법 | 장점 | 단점 | 채택? |
|---|---|---|---|
| **LD_PRELOAD + dlsym** | 사용자 코드/이미지 변경 0. 컨테이너 단위로 적용. | 정적 링크 / dlopen 우회 가능. | ✓ MVP |
| Custom CUDA driver | 완전 통제 | 커널 모듈 / NVIDIA 라이선스 문제 | ✗ |
| eBPF/uprobes | 커널 레벨에서 가로챔, 우회 어려움 | 복잡, GPU 콜은 사용자 공간이라 효용 낮음 | ✗ |
| PyTorch monkey-patch | 구현 쉬움 | PyTorch 한정, cuBLAS 직접 호출 못 잡음 | ✗ (너무 좁음) |
| MIG | 하드웨어 격리 | RTX 4060 미지원 | ✗ |
| MPS | 컨텍스트 통합으로 효율 | 같은 user/tenant 안에서만, 멀티컨테이너 부적합 | ✗ |

### 4.4 LD_PRELOAD의 알려진 함정 (논문에 명시)

- **정적 링크된 cudart**: `nvcc -cudart=static` 으로 빌드된 바이너리는 cudart 가 .so 가 아니므로 LD_PRELOAD 불가.
- **`dlopen("libcudart.so", ...)` 직접 호출**: 일부 코드는 동적 링크 표준을 거치지 않고 직접 dlopen 한다. 이 경우 우리 hook의 `cudaMalloc` 심볼이 link order 에서 안 보일 수 있다.
- **nvidia-container-runtime의 마운트 순서**: container runtime 이 host의 libcudart 를 컨테이너 안 특정 경로에 bind mount 한다. LD_PRELOAD 경로와 충돌할 수 있어, hook .so 는 *별도 경로* (예: `/opt/fgpu/`) 에 두고 명시적으로 PRELOAD 한다.
- **CUDA Driver API 직접 호출**: `cuMemAlloc_v2` 등 driver API 를 부르는 라이브러리는 Runtime API hook 으로 못 잡는다. → **Stage 5-C** 에서 driver-classic hook (`cuMemAlloc_v2`/`cuMemFree_v2`) + **Stage 6** 에서 VMM hook (`cuMemCreate`/`cuMemRelease`) 추가됨. `cuMemAllocAsync` (stream-ordered) / `cuMemAllocManaged` (UVM) 은 여전히 미해결 — Stage 6+ 후속.

---

## 5. Stage 1 코드 구조

### 5.1 파일 목록

| 파일 | 역할 |
|---|---|
| `hook/src/fgpu_hook.c` | LD_PRELOAD 본체. cudaMalloc / cudaFree 후킹, quota 시행, 포인터→size 추적. |
| `hook/tests/test_alloc.cu` | 후킹 검증용 최소 CUDA 프로그램. 256 MiB + 6 GiB 두 번 할당. |
| `scripts/build_hook.sh` | `gcc -shared -fPIC` 로 `build/libfgpu.so` 생성. |
| `scripts/run_test.sh` | `nvcc` 로 테스트 빌드 후 *baseline → hooked* 두 번 실행. |

### 5.2 quota 계산 로직

두 가지 경로가 있다 (하나만 적용됨, 명시적인 절대값이 우선):

```
FGPU_QUOTA_BYTES=<bytes>   # 절대값 직접 지정 (디버깅/테스트용)
        ↓ 우선
FGPU_RATIO=<0.0~1.0>       # 비율로 지정 (운영용)
        ↓
quota = ratio × cudaMemGetInfo(total)
```

`cudaMemGetInfo` 호출은 *첫 번째 cudaMalloc 시점에* lazy 하게 한다. 이유: CUDA 컨텍스트가 생성된 뒤여야 안전하게 호출 가능하기 때문 (라이브러리 로드 시점에 부르면 "no CUDA-capable device" 발생 가능).

### 5.3 동시성

- 모든 전역 상태 (`g_used`, `g_quota`, 추적 리스트) 는 단일 `pthread_mutex_t g_lock` 로 보호.
- 함수명 끝의 `_locked` 는 "호출자가 g_lock 을 *이미* 잡고 있어야 함" 을 표시하는 관습.
- PyTorch DataLoader 처럼 워커 스레드가 많은 경우에도 정확성 보장. 다만 lock contention 으로 hook overhead 가 생기는 점은 평가 항목으로 측정.

### 5.4 환경변수 요약

| 변수 | 의미 | 기본값 |
|---|---|---|
| `FGPU_RATIO` | GPU 전체 메모리 대비 사용 비율 (0.0~1.0) | 1.0 (제한 없음) |
| `FGPU_QUOTA_BYTES` | 사용 가능 바이트 수 직접 지정 (RATIO 무시) | 미설정 |

---

## 6. 실행 방법 (Stage 1)

Ubuntu GPU 서버에서:

```bash
# 0) 사전 준비 확인
nvidia-smi                              # 드라이버 동작
which nvcc                              # CUDA toolkit 설치
ls /usr/local/cuda/include/cuda_runtime.h

# 1) 권한 부여 (한 번만)
chmod +x scripts/build_hook.sh scripts/run_test.sh

# 2) hook 빌드
./scripts/build_hook.sh
# → build/libfgpu.so 생성

# 3) 테스트 실행 (default ratio = 0.4)
./scripts/run_test.sh

# 다른 비율 / 절대값 테스트
FGPU_RATIO=0.6 ./scripts/run_test.sh
FGPU_QUOTA_BYTES=$((512*1024*1024)) ./scripts/run_test.sh   # 512 MiB
```

### 합격 기준

`with hook` 실행 stderr 에 다음 패턴이 보여야 한다:

```
[fgpu] init: ratio=0.400 quota_bytes=0 (0 = lazy 계산)
[fgpu] quota lazily 계산: ratio=0.400 * total=8589934592 = 3435973836 bytes
[fgpu] ALLOW cudaMalloc ptr=0x... size=268435456 used=268435456/3435973836
[fgpu] DENY  cudaMalloc size=6442450944 used=268435456 quota=3435973836
[test] alloc2 (6GiB)   -> err=2 ptr=(nil)
[fgpu] FREE  ptr=0x... size=268435456 used=0/3435973836
```

확인 포인트: ratio 적용 / quota 계산 / ALLOW + DENY 분기 / err=2 전파 / cudaFree 후 used=0 복귀.

---

## 7. 한계 정리 (논문/발표용)

### 7.1 본질적 한계 (해결 불가, 구조적)

- **SM/연산 자원 격리 불가능**: Hooking 으로 메모리는 막아도, 한 컨테이너가 GPU 100% 점유하면 다른 컨테이너 latency 영향. → 이 부분은 MIG/MPS 가 필요한 영역이며, 본 프로젝트는 "메모리 quota + 정책적 스케줄링" 으로 한정.
- **Cooperative threat model**: 사용자가 의도적으로 우회하려 들면 막을 수 없음 (정적 링크, dlopen 직접 호출). 위협 모델 자체가 "협조적 사용자".
- **컨슈머 GPU 한정**: A100/H100 환경에서는 MIG 가 더 강력. 본 프로토타입의 가치는 *컨슈머 GPU 환경의 빈틈* 을 메우는 데 있음.

### 7.2 구현상 한계 (해결 / 미해결 현황)

- Driver API classic (`cuMemAlloc_v2`/`cuMemFree_v2`) → **Stage 5-C 에서 해결**.
- VMM API (`cuMemCreate`/`cuMemRelease`) → **Stage 6 에서 해결**. 세 alloc layer (Runtime/Driver/VMM) 모두 같은 quota state 공유 + reentrancy guard.
- `cuMemAllocAsync` (stream-ordered), `cuMemAllocManaged` (UVM) → 미해결, Stage 6+ 후속.
- Kernel-level 제어 → **Stage 7 에서 *측정만* 해결** (`cudaLaunchKernel` counter). 시행 (time-slicing) 은 도입 안 함 — SM 격리 자체가 hook 영역 밖.
- PyTorch caching allocator 가 큰 chunk 한 번 잡고 내부 재사용하는 패턴이라 *세부* 할당 패턴은 안 보임. 평가 시 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 설정으로 회피. caching on 시 한계는 *내재적* — VMM hook 도 첫 chunk 의 `cuMemCreate` 한 번만 보므로 해결 X. user-space slab 은 본질적으로 보이지 않음.
- 단일 호스트 / 단일 GPU 가정 → **Stage 8 에서 SQLite persistence + Stage 9 minimal 에서 멀티 GPU device pinning (`gpu_index`) + bearer 인증 도입**. 본격 멀티 노드 스케줄링은 Stage 9 (full) 후속.

### 7.3 그럼에도 의의

- 컨슈머 GPU (RTX 4060) + Docker 환경에서 fractional GPU 의 *메모리 quota 부분* 을 오픈소스 프로토타입으로 제시.
- Backend.AI fGPU 의 핵심 메커니즘인 API hooking 의 동작과 한계를 정량 측정.
- Stage 별로 hook → Driver API → kernel 까지 확장 가능한 구조라 후속 연구에 재활용 가능.

---

## 8. 단계별 로드맵

| Stage | 산출물 | 합격 기준 | 상태 |
|---|---|---|---|
| **0** | Ubuntu + driver + CUDA toolkit + nvidia-container-toolkit + Docker 설치 | `nvidia-smi`, `docker run --gpus all <cuda image> nvidia-smi` 동작 | 사전 |
| **1** | `libfgpu.so` + 단독 테스트 | quota 0.4 시 6 GiB 거부 로그 | ✅ |
| **2** | base Docker image, 컨테이너 안에서 hook 동작 | 컨테이너 안에서 동일 로그 | ✅ |
| **3** | FastAPI + Docker SDK, `/sessions` CRUD | curl 로 컨테이너 spawn → 안에서 cuda 코드 실행 → stop | ✅ |
| **4** | PyTorch 통합 검증 (caching off) | quota 0.4 에서 PyTorch 텐서 차단 | ✅ |
| **5-A** | 동시 격리 실험 자동화 | ratio 0.4/0.6 컨테이너 동시 실행 → 한쪽 OOM, 다른쪽 ALLOW, nvidia-smi CSV 캡처 | ✅ |
| **5-B** | 최소 Web UI | `GET /` 가 단일 HTML 반환, 폼·세션 테이블·로그 패널 동작 | ✅ |
| **5-C** | Driver API hook (`cuMemAlloc_v2`/`cuMemFree_v2`) | Driver API 만 부르는 바이너리에서도 quota 강제 + reentrancy 가드 | ✅ |
| **5-D** | Overhead 마이크로벤치 (`cudaMalloc`/`cudaFree` latency) | baseline vs hooked mean/p50/p99 표 산출 | ✅ |
| **6** | VMM API hook (`cuMemCreate` / `cuMemRelease`) | VMM 만 쓰는 바이너리에서도 quota 강제 — Runtime/Driver hook 과 같은 state 공유 + reentrancy guard | ✅ |
| **7** | `cudaLaunchKernel` 후킹 + launch counter | hook stderr 에 주기적 누적 카운트 + atexit 최종 요약 | ✅ |
| **8** | SQLite persistence + asyncio.to_thread (백엔드 비동기화) | 백엔드 재시작 후 세션 record 유지, 동시 POST 가 진짜 병렬 처리됨 | ✅ |
| **9 (minimal)** | bearer token 인증 (`FGPU_API_TOKEN`) + `gpu_index` 로 멀티-GPU device pinning | auth on/off 토글, `--gpus device=N` 패턴 | ✅ |
| **9 (full)** | Kubernetes scheduler / Redis store / RBAC | 후속 연구 영역 | 미정 |

각 stage 는 *그 단독으로 빌드/실행/검증 가능* 해야 한다.

> **로드맵 변경 메모**: 원안에서 Stage 6 으로 잡았던 "Driver API hook" 은 실제 구현 시점에 5-C 로 당겨졌다. 이유: Stage 5 (평가 실험) 의 *이후* 가 아니라 *이전* 에 driver hook 이 있어야 5-D 의 overhead 측정값이 두 layer 모두를 반영할 수 있어서. 결과적으로 Stage 6 는 VMM API (`cuMemCreate`/`cuMemRelease`) 로 재정의되어 별도 구현됨. Stage 9 는 본격 분산이 아닌 *minimal* (bearer auth + `gpu_index` device pinning) 만 먼저 도입하고, 본격 K8s/Redis 는 후속 연구로 유지.

---

## 9. 평가 실험 항목 (Stage 5)

논문 결과 섹션에 들어갈 핵심 실험들:

1. **Quota 정확도** — `FGPU_RATIO` 0.2 / 0.4 / 0.6 / 0.8 에서 실제로 거부되는 임계점 측정. 이론값 대비 오차.
2. **동시성 격리** — 컨테이너 A (ratio 0.4) + 컨테이너 B (ratio 0.6) 동시 실행. 한쪽이 다른 쪽 quota 를 침범하지 않음을 확인. 합 > 1.0 일 때 행동.
3. **차단 정상성** — quota 초과 호출이 `cudaErrorMemoryAllocation` 으로 정확히 사용자 코드까지 전파되는가.
4. **Hook overhead** — native (LD_PRELOAD 없음) vs hook(quota 1.0) vs hook(quota 0.4) 의 cudaMalloc latency, throughput 비교.
5. **PyTorch 동작** — caching off 모드에서 quota 적용 확인. caching on 모드에서의 행동 차이 분석 (논문 discussion 거리).
6. **nvidia-smi 일치성** — `nvidia-smi --query-gpu=memory.used` 값과 hook 내부 `g_used` 비교.

---

## 10. Stage 2–5 합류 요약 (논문 본문용)

§5 가 Stage 1 에 한정된 코드 구조 설명이므로, 이후 stage 에서 추가된 부분을 짧게 모아둔다. 자세한 파일 목록·실행 절차·합격 기준은 `CLAUDE.md` 참조.

### 10.1 Stage 2 — 컨테이너화

`runtime-image/Dockerfile` 이 `nvidia/cuda:12.4.1-devel-ubuntu22.04` 위에서 `test_alloc` 을 미리 컴파일한 베이스 이미지를 만든다 (`fgpu-runtime:stage2`). hook .so 는 *이미지에 굽지 않고* docker run 시 `-v` 로 bind-mount 한다 — hook 변경 시 이미지 재빌드 불필요. 컨테이너 안 entrypoint 가 env 와 hook 마운트를 sanity check 한 뒤 사용자 명령을 `exec`.

### 10.2 Stage 3 — FastAPI 백엔드

`backend/app/` 아래에 FastAPI app factory (`main.py`), 설정 (`core/config.py`, `FGPU_*` env override), Docker SDK 래퍼 (`services/docker_manager.py`), 인메모리 세션 매니저 (`services/session_manager.py` — 매 조회 시 docker daemon 에 status reconcile), REST 라우터 (`api/sessions.py` — `POST/GET/{id}/logs/{id}/stop/DELETE`), Pydantic 스키마 (`schemas/session.py`). 백엔드 자체는 GPU 권한 불필요, docker socket 만 있으면 됨.

### 10.3 Stage 4 — PyTorch 변형 이미지

`runtime-image-pytorch/Dockerfile` 이 `fgpu-runtime:stage2` 위에 PyTorch (cu121 wheel) 를 추가하고 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 을 default ENV 로 박는다. caching 이 켜져 있으면 첫 alloc 한 번만 실제 cudaMalloc 으로 가서 quota 가 *세부 텐서* 단위로 적용 안 됨 — 이 한계는 §7.2 와 동일한 이슈로 Stage 6 (VMM hook) 이 풀어주려는 문제. 백엔드는 `SessionCreate.image` 필드를 통해 코드 변경 0으로 새 이미지를 받아들임.

### 10.4 Stage 5-A — 동시 격리 실험

`scripts/eval/run_isolation.sh` 가 `/sessions` 를 두 번 POST (ratio 0.4 / 0.6, 둘 다 4 GiB 텐서 hold) 하고 백그라운드로 `nvidia-smi --query-compute-apps -l 1` 을 캡처. 산출물 `experiments/isolation_<TS>/` 에 두 컨테이너 로그 / 세션 JSON / nvidia-smi CSV / verdict 가 모임. 새 워크로드 `runtime-image-pytorch/test_hold.py` (한 번 alloc 후 N초 보유) 가 두 컨테이너의 메모리 점유 시간 윈도우를 겹치게 만듦.

이 실험이 §9.2 "동시성 격리" 평가 항목의 실측 데이터를 그대로 만들어준다.

### 10.5 Stage 5-B — Web UI

`backend/app/static/index.html` 단일 파일. 빌드 스텝 0, 외부 의존성 0 (vanilla JS + 인라인 CSS). `GET /` 에 `FileResponse` 로 직접 반환. 폼 → 세션 테이블 (3초 폴링) → 로그 패널 → Stop·Delete. 캡스톤 시연용 최소 UI 로 충분 (논문 본문에서는 스크린샷 한 장).

Stage 9 minimal 도입 후 두 필드 추가:

- **API token toolbar** — password 입력 + save 버튼. `localStorage` 에 저장, 모든 fetch 가 자동으로 `Authorization: Bearer <token>` 첨부. `auth_enabled` 가 healthz 에서 true 인데 토큰 미저장이면 빨간 경고 메시지.
- **`gpu_index` 입력란** — Create 폼의 number input. 빈 값 = `--gpus all` (기본), 정수 N = `--gpus device=N`. 세션 테이블에도 `gpu` 컬럼 추가 (`all` 또는 device id).

### 10.6 Stage 5-C — Driver API hook

`hook/src/fgpu_hook.c` 에 `cuMemAlloc_v2` / `cuMemFree_v2` 두 hook 추가. Runtime API 와 같은 `g_used`/`g_quota`/`g_lock`/`g_allocs` 를 공유. 핵심 디자인 결정:

- **이중 카운트 위험**: `cudaMalloc` 이 내부적으로 `cuMemAlloc_v2` 를 부르면 한 user-level 호출이 hook 을 두 번 통과해 `g_used` 가 두 배로 누적될 수 있음. → **per-thread reentrancy guard** (`__thread int g_in_hook`). 진입 시 set, 빠질 때 unset. 다른 hook 안에서 들어오면 bookkeeping 없이 위임만.
- **Lazy symbol resolution**: hook .so 가 먼저 로드된 뒤 사용자 프로그램이 늦게 `dlopen("libcuda.so")` 하는 케이스 대비. `fgpu_init_locked()` 가 매 호출마다 NULL 인 함수 포인터를 dlsym 으로 재시도. 한 번 채워지면 이후는 if 통과 only.

검증은 `scripts/run_driver_in_container.sh` + `hook/tests/test_driver_alloc.cu` (Driver API 만 사용, Runtime API 안 부름).

원래 §8 로드맵에서 Stage 6 으로 잡혀 있었으나, 5-D 의 overhead 측정이 두 layer 모두 반영되도록 5-C 로 당겨짐.

### 10.6-bis Stage 7 — Launch counter (cudaLaunchKernel hook)

`hook/src/fgpu_hook.c` 에 `cudaLaunchKernel` hook 추가. 의도적으로 **quota 시행 X** — 단순 호출 카운트만. SM 격리는 hook 으로 불가능하므로 (MIG/MPS 영역), 본 프로토타입은 "memory quota + launch frequency monitor" 두 축으로 한정.

핵심 결정:

- **Lock-free 카운터** — `__atomic_fetch_add(&g_launch_count, 1, __ATOMIC_RELAXED)`. PyTorch 가 launch 를 초당 수천 회 호출하므로 mutex 가 hot path. RELAXED 면 monotonic 단순 카운터에 충분.
- **Periodic stderr dump** — `FGPU_LAUNCH_LOG_EVERY` 마다 `[fgpu] LAUNCH count=N (every K)` 한 줄. 0 이면 off (5-D overhead 재측정 시 사용).
- **atexit 최종 요약** — `[fgpu] exit summary: total cudaLaunchKernel = N` 한 번. 정상 종료 경로만 보장 (signal/`_exit` 는 best-effort).
- **Reentrancy guard** — alloc hook 과 동일한 `__thread g_in_hook` 재사용. cudaLaunchKernel 내부에서 alloc 호출은 거의 없지만 일관성·보험.
- **Driver API `cuLaunchKernel` 미커버** — Stage 7+ 후속.

검증은 `scripts/run_launch_in_container.sh` + `hook/tests/test_launch.cu` (tiny noop kernel × N).

논문에서의 의의: 같은 GPU 를 공유하는 두 컨테이너의 launch frequency 비교 → fairness scheduler 의 입력값. 컨테이너 A 가 1초에 10000 launch / B 가 1000 launch 라면 *연산 자원* 도 비례 사용 중일 가능성 높음. 정확한 device time 은 cudaEvent injection 필요 (future work — Stage 7 의 자연스러운 확장).

### 10.7 Stage 5-D — Overhead 마이크로벤치

`hook/tests/bench_alloc.cu` 가 `[16, 64, 256, 1024]` MiB × 100 회 `cudaMalloc`/`cudaFree` 사이클을 `clock_gettime(CLOCK_MONOTONIC)` 으로 측정해 CSV 로 stdout 출력. `scripts/eval/run_overhead.sh` 가 baseline / hooked 두 번 돌려서 `experiments/overhead_<TS>/summary.csv` (per-size mean / p50 / p99 μs) + `summary.txt` (markdown 표) 산출. 이 데이터가 그대로 §9.4 "Hook overhead" 표가 됨.

PyTorch caching allocator 를 거치지 않는 *raw* cudaMalloc 측정이 핵심 — caching 이 첫 chunk 한 번만 잡고 이후 sub-alloc 으로 가버리면 hook overhead 자체가 보이지 않게 됨.

### 10.7-bis Stage 8 — SQLite persistence + 백엔드 비동기화

`backend/app/services/session_store.py` 신규. stdlib `sqlite3` 만 사용 (새 dep 없음). `SessionManager` 의 in-memory dict 와 asyncio.Lock 을 제거하고 store 를 single source of truth 로 전환. 모든 docker SDK + sqlite3 호출은 `asyncio.to_thread` 로 wrap.

핵심 결정:

- **stdlib sqlite3 + connection-per-call.** `sqlite3.Connection` 이 thread-safe 보장 안 함. `asyncio.to_thread` 가 임의 worker 에서 도므로 매 메서드마다 새 connection. `contextlib.closing` 으로 명시적 close — `with sqlite3.connect(...)` 만 쓰면 transaction 만 다루고 connection 은 GC 때까지 안 닫힘.
- **isolation_level=None (autocommit).** 단일 statement 만 실행하므로 명시 트랜잭션 불필요.
- **Schema migration 없음.** 변경 시 `rm -rf data/`. 프로덕션이라면 alembic 같은 도구 도입 — 본 프로토타입 범위 밖.
- **백엔드 비동기화로 진짜 동시성 확보.** 이전엔 `mgr.create()` 안의 `docker.containers.run()` 이 sync 라 이벤트 루프 막힘 → 두 POST 가 직렬화. 이제 `to_thread` 로 thread pool 에 넘기므로 두 컨테이너 spawn 이 진짜 병렬. 5-A 격리 실험의 "워크로드만 겹친다" 단서가 풀려, API 레이어부터 동시 처리됨.
- **List endpoint 의 reconcile 병렬화.** `list_all` 이 모든 session 에 대해 docker daemon status 조회 → 이전엔 순차 await, 이제 `asyncio.gather` 로 동시.

데이터 위치:
- `<repo>/data/sessions.db` — `.gitignore` 에 등록. `FGPU_DB_PATH` env 로 override 가능.

논문 의의: 백엔드가 stateless 가 아니라 *thin stateful layer* 로 진화. SQLite 는 single-host 한정이지만, SessionStore 인터페이스가 추상화 경계라 Stage 9 에서 Redis/Postgres 로 교체 시 `SessionManager` 변경 0. 본 프로토타입의 "확장 가능 구조" 명제의 1차 검증.

### 10.7-ter 5-A 확장 — launch counter ↔ nvidia-smi 메모리 시계열 상관

5-A 의 격리 검증 (한 쪽 OOM, 한 쪽 ALLOW) 와는 다른 목적: **두 컨테이너가 quota 안에서 *공존* 할 때** 같은 GPU 를 어떻게 나눠 쓰는지 시계열로 trace. 논문에서 "memory + launch frequency 두 축 monitoring" 의 직접적 증거 그래프.

새 산출물:

- `runtime-image-pytorch/test_compute.py` — 큰 텐서 alloc + 1024² matmul·relu 루프 HOLD_SEC 초 + free. 매 iter 가 cuBLAS 내부에서 cudaLaunchKernel 다수 발생 → Stage 7 카운터 빠르게 증가.
- `scripts/eval/run_correlation.sh` — driver. POST /sessions 두 번 → docker top 으로 컨테이너 PID set 캡처 → nvidia-smi 1초 polling → 종료 후 docker logs --timestamps 로 `[fgpu] LAUNCH count=...` 추출 → `_correlate.py` 가 PID 로 join.
- `scripts/eval/_correlate.py` — stdlib 만 사용. ISO8601 docker timestamp + nvidia-smi 의 `YYYY/MM/DD HH:MM:SS.fff` 둘 다 파싱, 시간축을 t=0 (실험 시작) 기준 초 단위 float 로 정규화. long-format CSV 출력.
- `scripts/eval/_correlate.py`, `correlation_summary.txt` — 사람이 읽는 요약 (PID set, final launch count, peak memory).

핵심 디자인 결정:

- **DockerManager 의 env 화이트리스트 passthrough**. 백엔드 프로세스에 `FGPU_LAUNCH_LOG_EVERY=500` 으로 떠 있으면 모든 spawn 컨테이너가 그 값 상속. API 변경 0. 임의 env leak 없게 명시 화이트리스트 (`_PASSTHROUGH_ENV` tuple) 만 forward.
- **API 우회 부분만 selective**. 세션 spawn / status / DELETE 는 백엔드 API 통과 (멀티테넌트 플로우 검증), 하지만 timestamped logs 와 `docker top` 은 docker CLI 직접 — `nvidia-smi` 캡처가 이미 그러는 패턴이라 일관됨.
- **PNG 안 만듦**. matplotlib dep 추가 안 함. CSV long-format 이 본질 — 사용자가 pandas pivot 으로 wide format 변환 후 plot.

논문 의의: §9.6 의 "nvidia-smi 일치성" 항목을 자동화한 형태. 두 metric 의 *시간 정렬* 이 핵심 증거 — hook 의 ALLOW 시점 ↔ nvidia-smi 가 메모리 증가 잡는 시점, hook 의 LAUNCH counter 누적 속도 ↔ container 의 GPU 실 활동량.

### 10.7-quater Stage 6 — VMM API hook (`cuMemCreate` / `cuMemRelease`)

`hook/src/fgpu_hook.c` 에 VMM (Virtual Memory Management) layer hook 추가. Runtime / Driver-classic 과 같은 `g_used` / `g_quota` / `g_allocs` / `g_lock` 공유.

핵심 디자인 결정:

- **Quota 부과 시점 = `cuMemCreate`**. VMM 은 *물리 alloc* (`cuMemCreate`) 과 *VA 예약* (`cuMemAddressReserve`) 과 *매핑* (`cuMemMap`) 을 분리한다. 우리 quota 는 물리 메모리량이므로 `cuMemCreate` 한 곳에만 부과, `cuMemRelease` 에서 회수. VA 관련 4개 (Reserve/Map/Unmap/AddressFree) 는 후킹 안 함 — 물리량 변화 없으므로 quota state 변화 없음.
- **Handle ↔ size 추적**: `CUmemGenericAllocationHandle` 은 `unsigned long long` typedef. x86_64 에서 `void*` 와 같은 너비 → 기존 `g_allocs` 리스트에 캐스트로 끼워넣음 (`cuMemAlloc_v2` 와 동일 패턴). 다른 layer 의 토큰과 *값* 충돌 가능성은 무시 가능 (opaque 토큰 공간 매우 큼).
- **Reentrancy guard 통합**: 같은 `__thread g_in_hook` 이 세 alloc layer (Runtime, Driver-classic, VMM) 의 상호 재진입을 모두 차단. cudart 가 내부에서 cuMemAlloc_v2 를 부르거나 cuMemAlloc_v2 가 VMM 으로 위임하는 가상의 경로도 방어.
- **`cuMemAllocAsync` / `cuMemAllocManaged` 미커버**: stream-ordered alloc / UVM 은 별도 코드 경로. Stage 6+ 후속.

검증은 `scripts/run_vmm_in_container.sh` + `hook/tests/test_vmm_alloc.cu` (cuInit + cuCtxCreate + cuMemGetAllocationGranularity + cuMemCreate × 2 + Release; Runtime/Driver-classic 안 부름).

논문 의의: §4.4 의 "Driver API 직접 호출" 한계가 Stage 5-C 로 일부, Stage 6 으로 modern path 까지 닫혔다. caching allocator 가 켜져 있어도 *어떤 alloc 경로* 든 quota 시행 — 다만 caching 은 user-space slab 추상화이므로 *세부 텐서 단위 quota* 는 여전히 안 보임 (불가피한 한계, §7.2 참조).

### 10.7-quinque Stage 9 minimal — bearer auth + 멀티-GPU device pinning

본격 Stage 9 (Kubernetes scheduler / Redis store / RBAC) 는 별도 프로젝트 규모라 후속으로 유지하고, *최소* 두 piece 만 먼저 도입:

- **Bearer token 인증** — `FGPU_API_TOKEN` env 가 비어있으면 인증 비활성 (개발 호환), 설정돼 있으면 `/sessions/*` 라우트가 `Authorization: Bearer <token>` 요구. `_require_auth` FastAPI dependency 가 라우터 전체에 적용. `hmac.compare_digest` 로 timing attack 방어. `/healthz` 와 `/` (UI) 는 dependency 영향 밖이라 항상 public — health check / 스크린샷 용도.
- **멀티-GPU device pinning** — `SessionCreate.gpu_index: Optional[int]`. None 이면 `--gpus all` (기본), 정수면 `--gpus device=N` 으로 해당 device 만 노출. 컨테이너 안 hook 은 그대로 동작 — 노출된 단일 GPU 의 `cudaMemGetInfo(total)` × ratio 가 quota.

핵심 디자인 결정:

- **Auth 가 라우터 dependency 로** — 각 엔드포인트마다 `Depends` 반복 안 하고 `APIRouter(dependencies=[Depends(_require_auth)])` 한 번만. 새 라우트 추가 시 자동 보호.
- **`gpu_index` 는 SQLite 에 영속화** — 백엔드 재시작 후 GET 이 같은 device id 반환. SessionStore 의 schema 가 v2 로 자동 migrate (`ALTER TABLE ... ADD COLUMN gpu_index INTEGER`, idempotent).
- **다중 GPU 호스트가 없는 환경에서도 회귀 0** — `gpu_index=None` 이 default, 기존 `--gpus all` 그대로. 단일 GPU 사용자에겐 보이지 않는 변경.
- **Token 회전 / RBAC / Redis 백업 store** 는 *의도적으로 미구현* — single static token + SQLite single-host 가 prototype scope. Stage 9 (full) 에서 OAuth/JWT/Redis adapter.

논문 의의: 캡스톤 시연 시 "auth 없는 데모" 와 "토큰 보호 데모" 를 한 명령으로 토글 가능. 평가 섹션의 *시스템 안전성* 측면을 채워줌 — 본격 RBAC 까진 아니지만 "API 가 protected mode 를 지원한다" 명제는 입증.

### 10.7-sextes 자동화 도구 — `run_all_tests.sh`

캡스톤 / 논문 마무리 단계에서 *전체 stage* 가 GPU 머신에서 정상 동작함을 한 번에 증명할 필요가 생긴다. 매번 9~10 개 스크립트를 손으로 돌리는 건 실수와 누락의 원인이라, 단일 orchestrator 를 도입.

`scripts/run_all_tests.sh` 가 다음을 자동 수행:

1. **Preflight** — `nvidia-smi` + `docker run --gpus all` 응답 확인. 실패 시 LINUX_SETUP.md §2/§3 으로 안내하며 즉시 종료.
2. **Idempotent build** — `build/libfgpu.so` / `fgpu-runtime:stage2` / `fgpu-runtime-pytorch:stage4` 가 없을 때만 빌드.
3. **Backend-less stage 검증** — Stage 1, 2, 5-C, 6, 7, 4 + backend pytest. 각자 컨테이너에서 단독 실행, hook stderr 의 ALLOW/DENY/exit-summary 패턴 grep 으로 PASS 판정.
4. **Backend lifecycle** — `FGPU_LAUNCH_LOG_EVERY=500` 으로 spawn → `/healthz` 30초 polling → Stage 3 smoke + 5-A isolation + 5-A correlation + 5-D overhead → trap 으로 자동 kill.
5. **Per-step 로그 분리 캡처** — 각 단계의 stdout+stderr 가 `experiments/runall_<TS>/<step>.log` 에 분리. fail 시 그 로그 한 파일만 보면 디버깅 가능.
6. **PASS/FAIL 표 + 종료 코드** — 0 = 모두 통과, 1 = 하나 이상 실패.

핵심 디자인 결정:

- **`set -e` 안 씀**. 한 단계 실패해도 나머지 진행 — 한 번 돌릴 때 최대 정보 수집. 캡스톤 마지막에 "어디까지 망가졌는지" 빠르게 진단 가능.
- **GPU 메모리 6 GiB 가정 (4070, 12 GB)**. ALLOC=6144 MiB 로 OOM 분기를 강제. 8 GB GPU 에서도 quota 0.4 의 3.2 GiB 초과라 동일하게 PASS, 더 작은 GPU 면 사용자가 env override.
- **백엔드 직접 spawn / kill** — 사용자가 별도 셸로 띄우는 부담 제거. 단, `run_all_tests.sh` 가 끝나면 백엔드도 죽으므로 *실제 UI 시연용* 으론 별도 `run_backend.sh` 필요.

논문 의의: §9 의 6개 실험 항목이 단일 명령으로 재현 가능 (`./scripts/run_all_tests.sh`). 캡스톤 시연 직전 또는 다른 GPU 머신으로 옮겼을 때 sanity check 로 사용. 결과의 일부 (5-A summary.txt, 5-D summary.csv, correlation.csv) 가 그대로 paper figure 의 입력.

### 10.8 평가 실험 항목 ↔ 실제 산출물 매핑

§9 의 평가 항목과 실제 자동화 스크립트의 대응:

| §9 항목 | 자동화 스크립트 | 산출물 |
|---|---|---|
| 9.1 Quota 정확도 | `scripts/run_in_container.sh` (수동 ratio sweep) | stderr 로그 |
| 9.2 동시성 격리 | `scripts/eval/run_isolation.sh` | `experiments/isolation_<TS>/summary.txt` |
| 9.3 차단 정상성 | `scripts/run_pytorch_in_container.sh` | `[hold-test] OOM` 라인 + exit code 1 |
| 9.4 Hook overhead | `scripts/eval/run_overhead.sh` | `experiments/overhead_<TS>/summary.csv` |
| 9.5 PyTorch 동작 | `scripts/run_pytorch_in_container.sh` (caching off) | hook 로그 + PyTorch OOM |
| 9.6 nvidia-smi 일치성 | 5-A 가 nvidia-smi CSV 같이 캡처하므로 사후 비교 | `nvidia_smi.csv` ↔ container 로그 |
| 9.7 시계열 trace (memory + launch) | `scripts/eval/run_correlation.sh` (5-A 확장) | `experiments/correlation_<TS>/correlation.csv` |
| 9.8 VMM API quota 시행 | `scripts/run_vmm_in_container.sh` | hook stderr `[fgpu] ALLOW/DENY cuMemCreate` |
| 9.9 인증 / 멀티-GPU API 동작 | `curl` (수동) | `/healthz` 의 `auth_enabled`, 401 vs 201 응답 |

---

## 11. 작업 워크플로 규칙

이 프로젝트는 *단계별 검증* 을 원칙으로 한다.

- 한 번에 큰 코드를 만들지 않는다. Stage 단위로 끊어서 빌드/실행/검증 가능한 형태로 제공.
- 다음 stage 로 넘어가기 전에 합격 기준 충족 확인.
- Claude 작업 세션에서는 사용자가 "다음" 이라고 명시할 때까지 다음 stage 코드를 작성하지 않는다 (`CLAUDE.md` 의 Workflow rule 참고).
