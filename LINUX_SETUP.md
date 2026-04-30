# Linux 머신 첫 셋업 & 검증 — Runbook

이 문서는 **Windows 에서 GitHub 에 push 한 뒤, RTX 4070 이 달린 Ubuntu
머신에서 처음 클론해서 모든 stage 를 검증**하는 끝에서 끝까지의 순서를
적은 runbook 입니다. 각 단계마다 실행 명령 + 기대 결과 + 자주 발생하는
실패 모드를 같이 적어둡니다.

---

## 0. 전제

- Ubuntu 22.04 (다른 배포판이면 패키지 이름만 다름)
- RTX 4070 (또는 다른 모던 NVIDIA GPU)
- 사용자 계정에 sudo 권한
- 인터넷 연결

목표: 약 **30~60 분** 안에 README.md "What's implemented" 의 모든 항목을
직접 실행해서 PASS 확인.

> **빠른 길**: NVIDIA + CUDA + Docker + nvidia-container-toolkit 이
> 이미 다 깔려있는 머신이면 §1 ~ §3 을 건너뛰고 §4 부터 시작해서
> `./scripts/run_all_tests.sh` 한 줄로 §5 ~ §7 의 모든 검증을
> 자동화할 수 있습니다. 처음 보는 머신이면 §1 ~ §3 부터 차례로.

---

## 1. 사전 환경 점검 (3 분)

먼저 *이미 깔려 있는 것* 부터 확인:

```bash
nvidia-smi                       # NVIDIA 드라이버
nvcc --version                   # CUDA toolkit
docker version                   # Docker
docker info | grep -i runtime    # nvidia runtime
gcc --version                    # 빌드용
python3 --version                # 3.11+ 권장
```

위 6개가 모두 동작하면 **§4 부터 바로 진행**해도 됩니다.

`nvidia-smi` 가 RTX 4070 정보를 출력하지 않으면 §2 부터, `docker info`
의 runtime 출력에 `nvidia` 가 없으면 §3 부터 진행.

---

## 2. NVIDIA 드라이버 + CUDA toolkit (한 번만, ~15 분)

### 2.1 드라이버

```bash
# 최신 drivers 확인
ubuntu-drivers devices
# 추천 driver 자동 설치
sudo ubuntu-drivers autoinstall
sudo reboot
# 재부팅 후
nvidia-smi                       # RTX 4070, driver version, CUDA version 표시 확인
```

### 2.2 CUDA toolkit (12.4 권장 — 본 프로젝트가 12.4.1 컨테이너 베이스)

NVIDIA 공식 페이지의 .deb (network) 가 가장 깔끔:

```bash
# https://developer.nvidia.com/cuda-12-4-1-download-archive
# Linux → x86_64 → Ubuntu → 22.04 → deb (network) 의 명령 그대로:

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4

# PATH 설정 (~/.bashrc 끝에)
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 확인
nvcc --version                   # release 12.4 표시
ls /usr/local/cuda/include/cuda_runtime.h
```

---

## 3. Docker + nvidia-container-toolkit (한 번만, ~5 분)

```bash
# Docker
sudo apt-get install -y docker.io
sudo usermod -aG docker $USER
newgrp docker                    # 현재 셸에 그룹 즉시 반영

# nvidia-container-toolkit
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Docker 가 nvidia runtime 을 알도록 설정
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 확인 — 컨테이너 안에서 nvidia-smi 가 돌면 성공
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
# → RTX 4070 정보 출력되면 PASS
```

**자주 발생하는 실패**:
- `permission denied while trying to connect to the Docker daemon` → `newgrp docker` 안 했거나 새 셸에서 다시 시도.
- `could not select device driver "" with capabilities: [[gpu]]` → nvidia-container-toolkit 설치 / `nvidia-ctk runtime configure` 누락.
- `nvidia-smi` 가 컨테이너 안에서 안 보임 → 호스트 nvidia-smi 부터 안 되는지 재확인.

---

## 4. 레포 클론 + 권한 (1 분)

```bash
cd ~
git clone <your-github-url> fgpu
cd fgpu

# Windows 에서 push 한 거라 실행 권한이 빠질 수 있음
chmod +x scripts/*.sh scripts/eval/*.sh runtime-image/entrypoint.sh

# 빠른 sanity
ls -la                           # README.md, CLAUDE.md, hook/, backend/ 등 보이는지
```

> 만약 `~/projects` 같은 다른 위치에 두고 싶으면 거기서 clone. 단,
> 경로에 한글/공백 들어가면 docker bind-mount 가 가끔 깨지므로 영문
> 경로 권장.

---

## 5. 빌드 (3 단계, 합쳐서 ~10 분 — 첫 빌드 기준)

```bash
# 5.1) hook .so — 호스트 CUDA 헤더와 함께 gcc 로 빌드 (몇 초)
./scripts/build_hook.sh
# → build/libfgpu.so 생성

# 5.2) base runtime image — fgpu-runtime:stage2 (~3 분)
./scripts/build_image.sh

# 5.3) PyTorch variant — fgpu-runtime-pytorch:stage4 (~5~10 분, 5 GB 휠 다운)
./scripts/build_pytorch_image.sh
```

**자주 발생하는 실패**:
- `cuda_runtime.h: No such file or directory` → `CUDA_HOME` 미설정. `CUDA_HOME=/usr/local/cuda-12.4 ./scripts/build_hook.sh` 로 override.
- `docker build ... no such image: nvidia/cuda:12.4.1-...` → 인터넷 또는 docker hub 접근 문제. 잠시 후 재시도.

---

## 6. Stage 별 검증 (각 1~2 분, 합쳐서 ~15 분)

여기부터는 **실제 hook 동작 확인**. 각 명령 후 PASS 조건 확인.

> **모든 stage 한 번에 자동 실행**: `./scripts/run_all_tests.sh` 가 §6
> 의 모든 단계 + §7 의 평가 실험까지 순차로 돌리고 끝에 PASS/FAIL 표를
> 출력합니다. 빌드 안 된 산출물은 자동 빌드. 백엔드 spawn / 정리도
> 자동. 처음 한 번은 PyTorch 이미지 빌드 때문에 ~10 분, 이후는 ~5 분.
>
> 한 단계 실패해도 멈추지 않고 끝까지 돌려서 최대 정보 수집. 결과는
> `experiments/runall_<TS>/<step>.log` 에 단계별로 분리 캡처.
>
> 아래 §6.1~§6.10 은 *수동 단계별* 검증으로, 한 stage 만 격리해서
> 디버깅하거나 자동 실행에서 fail 난 stage 를 단독 재현할 때 사용합니다.

### 6.1 Stage 1 — host hook (가장 기본)

```bash
./scripts/run_test.sh
```

**PASS**: `with hook` 실행 stderr 에 다음 줄 모두 확인:
- `[fgpu] init: ratio=0.400 ...`
- `[fgpu] quota lazily 계산: ... total=12884901888 ...` (4070 → 12 GB)
- `[fgpu] ALLOW cudaMalloc ... size=268435456 ...` (256 MiB)
- `[fgpu] DENY  cudaMalloc size=6442450944 used=... quota=...` (6 GiB)
- `[fgpu] FREE  ptr=...` (정리)

### 6.2 Stage 2 — 컨테이너 안 hook

```bash
./scripts/run_in_container.sh
```

**PASS**: 위와 같은 [fgpu] 패턴 + `[entrypoint]` 줄이 prefix 로 보임.

### 6.3 Stage 5-C — Driver API

```bash
./scripts/run_driver_in_container.sh
```

**PASS**:
- stderr `[fgpu] init: real cuMemAlloc_v2=0x...` (NULL 아님)
- `[fgpu] ALLOW cuMemAlloc_v2 ... size=268435456 ...`
- `[fgpu] DENY  cuMemAlloc_v2 size=6442450944 ...`
- stdout 6 GiB 시도 결과 `result=2 (CUDA_ERROR_OUT_OF_MEMORY)`

### 6.4 Stage 6 — VMM API

```bash
./scripts/run_vmm_in_container.sh
```

**PASS**:
- stderr `[fgpu] init: real cuMemCreate=0x... cuMemRelease=0x...`
- `[fgpu] ALLOW cuMemCreate handle=0x... size=268435456 ...`
- `[fgpu] DENY  cuMemCreate size=6442450944 ...`

### 6.5 Stage 7 — launch counter

```bash
./scripts/run_launch_in_container.sh
```

**PASS**:
- stderr `[fgpu] LAUNCH count=100 (every 100)` ... `count=1000 (every 100)` 10줄
- 마지막 `[fgpu] exit summary: total cudaLaunchKernel = 1000`
- stdout `[test-launch] kernel atomics = 1000`

### 6.6 Stage 4 — PyTorch (4070 은 텐서 키워야 OOM 보장)

```bash
# 기본: alloc1=256 MiB, alloc2=4 GiB
./scripts/run_pytorch_in_container.sh
# → 4070 quota 0.4 = 4.8 GiB → 4 GiB 통과 (OOM 안 남)

# 4070 에서 OOM 확실히 보려면:
PYTEST_ALLOC2_MIB=6144 ./scripts/run_pytorch_in_container.sh
# → 6 GiB 가 quota 4.8 GiB 초과 → OOM, [fgpu] DENY 보임
```

**PASS**: hooked 실행 시 `[pytorch-test] OOM ← cudaErrorMemoryAllocation`
+ stderr `[fgpu] DENY cudaMalloc size=6442450944 ...`.

### 6.7 Stage 3 + 5-B — 백엔드 + Web UI

```bash
# 셸 #1
./scripts/run_backend.sh
# → uvicorn http://0.0.0.0:8000 ... 출력
```

```bash
# 셸 #2 — curl 검증
./scripts/smoke_test_api.sh
# → 5단계 (healthz / POST / GET / logs / DELETE) 모두 200, exit_code=0
```

브라우저에서 `http://<linux-ip>:8000/` 접속 (같은 머신이면 localhost):
- "fGPU Sessions" 페이지
- 폼에 `ratio=0.4`, `command=python3 /opt/fgpu/test_hold.py` 제출
- 1초 안에 행 생성 → status created→running→exited 자동 갱신
- 행 클릭 → 로그 패널에 `[fgpu]`, `[hold-test]` 줄 표시
- delete 버튼 → 행 사라짐

### 6.8 Stage 8 — persistence

```bash
# 셸 #1 백엔드 실행 중인 상태에서
curl -X POST http://localhost:8000/sessions \
    -H 'Content-Type: application/json' \
    -d '{"ratio": 0.4}'
# 받은 id 메모

# 셸 #1 의 백엔드 Ctrl+C 후 재실행
./scripts/run_backend.sh

# 다른 셸에서:
curl http://localhost:8000/sessions
# → 위 id 가 그대로 보이면 PASS, status 가 docker daemon 과 reconcile 됨
```

### 6.9 Stage 9 minimal — auth

```bash
# 셸 #1: 토큰 켜고 재실행
FGPU_API_TOKEN=devsecret ./scripts/run_backend.sh

# 셸 #2:
curl -s http://localhost:8000/healthz | grep auth_enabled
# → "auth_enabled": true

curl -i -X POST http://localhost:8000/sessions \
    -H 'Content-Type: application/json' -d '{"ratio":0.4}'
# → HTTP/1.1 401 Unauthorized + "missing bearer token"

curl -i -X POST http://localhost:8000/sessions \
    -H 'Authorization: Bearer devsecret' \
    -H 'Content-Type: application/json' -d '{"ratio":0.4}'
# → HTTP/1.1 201 Created
```

브라우저 측 검증: `http://localhost:8000/` 의 상단 toolbar 에 token
입력란이 있음. `devsecret` 입력 후 save → 모든 API 호출에 자동 첨부 →
세션 테이블 정상 로딩. 미입력 상태면 `⚠ auth ON 인데 토큰 미저장 →
401 발생` 경고.

### 6.10 백엔드 단위 테스트 (docker / GPU 불필요)

```bash
cd backend
pip install -e ".[dev]"
pytest -v
# → 8 passed
cd ..
```

---

## 7. 논문/캡스톤용 데이터 산출 (각 1~3 분)

```bash
# 백엔드 떠 있는 상태에서:

# 7.1 격리 실험 — 5-A
ALLOC_MIB=6144 HOLD_SEC=8 ./scripts/eval/run_isolation.sh
# → experiments/isolation_<TS>/summary.txt 의 "VERDICT: PASS"
#   (4070 12 GB → ratio 0.4 quota 4.8 GiB, 6 GiB alloc 으로 A 가 OOM 보장)

# 7.2 overhead 마이크로벤치 — 5-D
./scripts/eval/run_overhead.sh
# → experiments/overhead_<TS>/summary.txt 의 markdown 표
#   (cudaMalloc/cudaFree mean·p50·p99 μs, Δ % 컬럼)

# 7.3 시계열 trace — 5-A 확장
# 백엔드 재시작이 필요 (env passthrough)
# 셸 #1:
FGPU_LAUNCH_LOG_EVERY=500 ./scripts/run_backend.sh

# 셸 #2:
./scripts/eval/run_correlation.sh
# → experiments/correlation_<TS>/correlation.csv
#   pandas 로 pivot 해서 plot — 명령 예시는 스크립트 끝에 출력됨
```

---

## 8. 발표 시 권장 데모 시나리오 (3 분)

브라우저 1개 + 터미널 2개 화면 분할.

**터미널 A**: `nvidia-smi --loop-ms 500` (메모리 사용량 실시간)
**터미널 B**: `docker logs -f fgpu-<id-A>` 와 `... fgpu-<id-B>` 동시

**브라우저**:
1. 폼 1번 제출 — `ratio=0.4`, `command=python3 /opt/fgpu/test_hold.py`,
   환경변수로 `ALLOC_MIB=6144` (사이드 차원에서 docker run 으로 추가
   주입은 어려우므로, 시연용으론 image 의 env 또는 백엔드를
   `ALLOC_MIB=6144` 로 띄우기)
2. 폼 2번 제출 — `ratio=0.6`, 같은 command
3. 두 컨테이너 hold 구간 동안 nvidia-smi 가 둘 다 잡음 → 격리 시각화
4. A (ratio 0.4) 가 OOM exit, B (ratio 0.6) 가 정상 exit
5. 양쪽 다 delete

> **시연 팁**: workload 의 `ALLOC_MIB` / `HOLD_SEC` env 를 컨테이너에 직접
> 넘기는 UI 필드는 현재 없습니다. 대신 image 안 default 값을 바꾸거나
> backend 에서 `_PASSTHROUGH_ENV` 화이트리스트에 추가하면 됩니다 (현재
> 는 `FGPU_LAUNCH_LOG_EVERY` 만 통과). 시연 직전 `image` 에 `ALLOC_MIB`
> 가 박힌 별도 태그 (예: `fgpu-runtime-pytorch:stage4-demo6g`) 를
> 만들어두면 깔끔.

---

## 9. 트러블슈팅 모음

| 증상 | 원인 / 해결 |
|---|---|
| `permission denied: docker.sock` | `sudo usermod -aG docker $USER && newgrp docker` |
| `could not select device driver "" with capabilities: [[gpu]]` | nvidia-container-toolkit 설치/configure 누락 → §3 재시도 |
| `cuda_runtime.h: No such file or directory` (build_hook.sh) | `CUDA_HOME` 미설정 → `CUDA_HOME=/usr/local/cuda-12.4 ./scripts/build_hook.sh` |
| `nvcc: command not found` (build_image.sh 안에서) | base image 가 devel 인지 확인 (nvidia/cuda:*-devel-* 이어야 함). 변수 `CUDA_VERSION` 으로 override 가능 |
| `OOM` 이 예상보다 빨리 / 늦게 발생 | 4070 은 12 GB. ratio × 12 GB 가 quota. 시나리오 ALLOC 값 조정 |
| 백엔드 healthz 의 `host_hook_exists: false` | `build/libfgpu.so` 가 없음 → `./scripts/build_hook.sh` 재실행 |
| 백엔드 startup 시 `data/sessions.db` 생성 실패 | `data/` 권한 → `mkdir -p data && chmod 755 data` |
| Stage 8 schema mismatch (이전 버전 DB) | `rm -rf data/` (안 풀리면 wipe) |
| Web UI 가 401 만 반환 | auth ON 상태인데 토큰 미저장 → toolbar 에 토큰 입력 후 save |
| `docker top <id>: no such container` (run_correlation.sh) | 컨테이너가 너무 빨리 exit — `HOLD_SEC` 키우기 |
| nvidia-smi CSV 가 비어있음 | 컨테이너 PID 가 host 의 nvidia-smi 시점에 GPU 미점유 → workload 더 길게 |

---

## 10. 검증 완료 체크리스트

> **빠른 길**: `./scripts/run_all_tests.sh` 한 번 돌리면 아래 6 ~ 12,
> 16 ~ 19 가 자동 채워집니다. 출력 끝의 PASS/FAIL 표를 그대로 캡스톤
> 자료로 사용 가능. 13 (브라우저 UI), 14 (재시작 resilience), 15 (auth
> 토글) 세 항목만 *수동으로* 더 확인하면 전체 검증 끝.

```
[ ] 1. nvidia-smi 가 RTX 4070 표시
[ ] 2. docker run --gpus all ... nvidia-smi 통과
[ ] 3. ./scripts/build_hook.sh 통과
[ ] 4. ./scripts/build_image.sh 통과
[ ] 5. ./scripts/build_pytorch_image.sh 통과
[ ] 6. Stage 1: ./scripts/run_test.sh — DENY 줄 확인
[ ] 7. Stage 2: ./scripts/run_in_container.sh — DENY 줄 확인
[ ] 8. Stage 5-C: ./scripts/run_driver_in_container.sh — cuMemAlloc_v2 DENY
[ ] 9. Stage 6: ./scripts/run_vmm_in_container.sh — cuMemCreate DENY
[ ]10. Stage 7: ./scripts/run_launch_in_container.sh — exit summary count=1000
[ ]11. Stage 4: PYTEST_ALLOC2_MIB=6144 ./scripts/run_pytorch_in_container.sh — OOM
[ ]12. Stage 3: ./scripts/smoke_test_api.sh — 5단계 200
[ ]13. Stage 5-B: 브라우저 UI 에서 세션 생성/삭제 (수동)
[ ]14. Stage 8: 백엔드 재시작 후 GET /sessions 에 이전 record 존재 (수동)
[ ]15. Stage 9 minimal: 토큰 ON/OFF 토글 동작 (수동)
[ ]16. backend pytest 8 passed
[ ]17. 5-A: experiments/isolation_<TS>/summary.txt → VERDICT: PASS
[ ]18. 5-D: experiments/overhead_<TS>/summary.txt 표 산출
[ ]19. 5-A 확장: experiments/correlation_<TS>/correlation.csv 산출
```

위 19개 다 체크되면 **이 프로토타입이 RTX 4070 환경에서 모든 stage 정상
동작** 임을 증명한 셈. 캡스톤 발표/논문에서 자료로 그대로 사용.

---

## 11. GitHub 푸시 시 주의 (Windows 측에서 한 번)

리눅스 옮기기 전 Windows 에서 마지막 점검:

```powershell
cd c:\Users\양재우\Desktop\backend_ai\newback
git status                       # 현재 변경 사항 확인
git ls-files | findstr "\.so$ \.db$"
# ↑ 출력 0줄이어야 함 (build/ data/ 가 .gitignore 에 잡혀있어야 정상)

# 첫 푸시인 경우:
git init
git add .
git status                       # build/ data/ experiments/ .venv/ 안 보여야 OK
git commit -m "Initial fGPU prototype"
git branch -M main
git remote add origin <your-github-url>
git push -u origin main
```

Linux 머신에서는 §4 부터 시작.
