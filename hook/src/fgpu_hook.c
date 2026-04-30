/* =====================================================================
 * fgpu_hook.c  —  LD_PRELOAD 기반 CUDA API 후킹 (Stage 1 + Stage 5-C)
 *
 * 이 파일은 무엇을 하는가?
 * ---------------------------------------------------------------------
 * NVIDIA GPU 메모리를 할당/해제하는 함수들을 "가로채서(intercept)"
 * quota 검사 로직을 먼저 실행한 뒤 진짜 CUDA 함수에 위임한다.
 * 두 layer 의 함수를 모두 잡는다:
 *
 *   - Runtime API (libcudart):  cudaMalloc, cudaFree
 *   - Driver  API (libcuda):    cuMemAlloc_v2, cuMemFree_v2
 *   - VMM     API (libcuda):    cuMemCreate, cuMemRelease (Stage 6)
 *   - Kernel  launch monitor:   cudaLaunchKernel (Stage 7, 카운터만)
 *
 * 두 alloc layer 모두 같은 quota state (g_used, g_quota, g_allocs, g_lock)
 * 를 공유한다. 따라서 사용자가 어느 쪽 API 를 쓰든 quota 가 함께
 * 강제된다.
 *
 * Kernel launch hook 은 quota 를 *시행* 하지 않는다 — 단지 호출 횟수를
 * 누적해 process 별 GPU 활동도(temporal share) 의 proxy 를 제공.
 * 진짜 SM 격리는 MIG/MPS 가 필요하므로, 본 프로토타입은 "memory quota
 * + launch frequency monitor" 두 축으로 한정.
 *
 * 어떻게 가능한가? — LD_PRELOAD 한 줄 요약
 * ---------------------------------------------------------------------
 * Linux는 프로그램을 실행할 때 "동적 라이브러리(.so)들을 어떤 순서로
 * 로드할지" 결정한다. 환경변수 LD_PRELOAD 에 .so 경로를 넣으면, 그
 * 라이브러리가 가장 먼저 로드된다. 그 안에 cudaMalloc 이라는 같은
 * 이름의 함수가 들어 있으면, 사용자 프로그램이 cudaMalloc 을 호출할 때
 * 진짜 CUDA 라이브러리(cudart) 가 아니라 *우리* 의 cudaMalloc 이
 * 먼저 불린다. 이게 "심볼 가로채기(symbol interception)" 이다.
 *
 * 그러면 진짜 cudaMalloc 은 어떻게 부르는가? — dlsym(RTLD_NEXT)
 * ---------------------------------------------------------------------
 * dlsym() 은 "어떤 함수의 주소를 찾아줘" 라고 동적 링커에게 묻는 함수다.
 * 첫 번째 인자로 RTLD_NEXT 를 주면 "*나 다음에 로드된 라이브러리들*
 * 중에서 해당 이름을 찾아줘" 라는 뜻이다. 그래서 dlsym(RTLD_NEXT,
 * "cudaMalloc") 은 우리 .so 다음에 있는 cudart 의 진짜 cudaMalloc
 * 주소를 돌려준다. 이걸 함수 포인터에 저장해 두고 필요할 때 호출한다.
 *
 * 두 layer 동시 후킹 시의 위험: 이중 카운트 (Stage 5-C 핵심 이슈)
 * ---------------------------------------------------------------------
 * 만약 libcudart 의 cudaMalloc 이 내부적으로 libcuda 의 cuMemAlloc_v2
 * 를 호출하면, 사용자 한 번의 cudaMalloc 호출이 우리 hook 을 *두 번*
 * 통과하면서 g_used 가 두 배로 누적될 수 있다.
 *
 * 해결: per-thread reentrancy flag (__thread g_in_hook).
 *   진입 시 1 set → 진짜 함수 호출 → 빠질 때 0 reset.
 *   다른 hook 안에서 들어왔을 땐 (g_in_hook==1) 카운트·로그 skip 하고
 *   진짜 함수만 위임. 멀티스레드 안전성도 자동 — TLS 라 스레드 간
 *   간섭 X.
 *
 * 이 hook 의 한계 (논문에 미리 적어둘 것)
 * ---------------------------------------------------------------------
 *   1) 정적 링크된 바이너리는 LD_PRELOAD 로 가로챌 수 없다.
 *   2) PyTorch caching allocator 는 한 번에 큰 chunk 만 잡으므로
 *      세부 할당/해제 패턴은 보이지 않는다 (테스트 시 caching off 필요).
 *   3) cuMemAllocAsync, cuMemAllocManaged, 그리고 VMM API
 *      (cuMemCreate/cuMemMap) 는 아직 후킹 안 됨 — Stage 6+.
 *   4) SM 격리는 hooking 으로 못 한다 (MIG/MPS 가 필요).
 * ===================================================================== */

/* _GNU_SOURCE 매크로:
 *   dlsym() 의 RTLD_NEXT 는 GNU 확장 기능이다. 이 매크로를 #include 보다
 *   *먼저* 정의해야 dlfcn.h 가 RTLD_NEXT 를 노출시켜 준다. 빠뜨리면
 *   "RTLD_NEXT undeclared" 컴파일 에러가 난다.
 */
#define _GNU_SOURCE

#include <stdio.h>      /* fprintf, stderr */
#include <stdlib.h>     /* getenv, atof, strtoull, malloc, free */
#include <string.h>     /* (현재는 미사용이지만 향후 strncmp 등 대비) */
#include <stdint.h>     /* uintptr_t — CUdeviceptr ↔ void* 변환 */
#include <dlfcn.h>      /* dlsym, RTLD_NEXT, dlerror */
#include <pthread.h>    /* pthread_mutex_t — 멀티스레드 안전성 */
#include <cuda_runtime_api.h>  /* cudaError_t, cudaMemGetInfo, cudaSuccess 등 */
#include <cuda.h>              /* CUresult, CUdeviceptr, CUDA_SUCCESS 등 (Driver API) */


/* =====================================================================
 * (1) "진짜" CUDA 함수의 주소를 담아둘 함수 포인터
 *
 *  - real_cudaMalloc 은 "void** 와 size_t 를 받아 cudaError_t 를 반환
 *    하는 함수"의 주소를 가리킨다. cudaMalloc 의 시그니처와 동일하다.
 *  - 처음에는 NULL 이고, fgpu_init_locked() 에서 dlsym 으로 채운다.
 *    이렇게 "필요할 때 처음 한 번만 채운다" 를 lazy initialization
 *    (지연 초기화) 이라고 부른다.
 *  - static 키워드:  이 변수가 *이 .c 파일 안에서만* 보이게 한다
 *    (다른 .so/실행파일에서 같은 이름이 충돌해도 무관).
 * ===================================================================== */
static cudaError_t (*real_cudaMalloc)(void **, size_t)        = NULL;
static cudaError_t (*real_cudaFree)(void *)                   = NULL;
/* Driver API (libcuda) 함수 포인터 — Stage 5-C.
 * libcudart 와 동일한 dlsym(RTLD_NEXT) 패턴. libcuda 가 사용자 프로세스에
 * 로드되어 있을 때만 실제 값이 채워진다. NULL 이라도 fatal 은 아니다 —
 * 사용자가 driver API 를 안 쓰면 우리 driver hook 도 안 불린다. */
static CUresult    (*real_cuMemAlloc_v2)(CUdeviceptr *, size_t) = NULL;
static CUresult    (*real_cuMemFree_v2)(CUdeviceptr)            = NULL;
/* VMM API (libcuda) — Stage 6. cuMemCreate 에서 *물리* 메모리가 잡히고
 * cuMemRelease 에서 풀리므로 quota 는 이 두 시점에 한정. VA 예약/매핑
 * (cuMemAddressReserve/cuMemMap/cuMemUnmap/cuMemAddressFree) 은 물리량
 * 변화 없음 → 의도적으로 후킹 안 함. */
static CUresult    (*real_cuMemCreate)(CUmemGenericAllocationHandle *,
                                       size_t,
                                       const CUmemAllocationProp *,
                                       unsigned long long) = NULL;
static CUresult    (*real_cuMemRelease)(CUmemGenericAllocationHandle) = NULL;
/* Kernel launch hook (Stage 7) — quota 시행 X, 호출 횟수만 누적. */
static cudaError_t (*real_cudaLaunchKernel)(const void *, dim3, dim3,
                                            void **, size_t, cudaStream_t) = NULL;


/* =====================================================================
 * (2) 프로세스 전역 quota 상태
 *
 *  - g_lock  : 여러 스레드가 동시에 cudaMalloc 을 부를 수 있으므로
 *              g_used / g_quota / 링크드 리스트를 보호하는 자물쇠.
 *  - g_used  : 현재까지 hook 을 통해 할당된 GPU 메모리 총량(bytes).
 *  - g_quota : 이 프로세스가 쓸 수 있는 상한선(bytes). 0 이면
 *              "아직 안 정함 — 첫 cudaMalloc 시점에 lazy 계산".
 *  - g_ratio : FGPU_RATIO 환경변수에서 읽은 비율 (0.0 ~ 1.0).
 *  - g_inited: 초기화가 끝났는지 여부 (한 번만 하기 위한 플래그).
 * ===================================================================== */
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static size_t g_used   = 0;
static size_t g_quota  = 0;
static double g_ratio  = 1.0;
static int    g_inited = 0;

/* =====================================================================
 * (2-bis) Reentrancy guard — Stage 5-C
 *
 * libcudart 의 cudaMalloc 이 내부적으로 libcuda 의 cuMemAlloc_v2 를
 * 호출하면, 사용자 한 번의 호출이 우리 hook 을 두 번 통과해 g_used 가
 * 두 배로 누적될 수 있다. 이를 막는 thread-local flag.
 *
 *   __thread  : GCC/Clang 의 thread-local storage 키워드. 각 스레드가
 *               이 변수의 *자기만의* 사본을 갖는다. 그래서 lock 없이도
 *               안전하다 — 다른 스레드의 g_in_hook 와 간섭 없음.
 *   g_in_hook : 0 (외부에서 진입) 또는 1 (이미 우리 hook 안).
 *
 * 사용 패턴:
 *   if (g_in_hook) { return real_func(...); }   // nested → 위임만
 *   g_in_hook = 1;
 *   ... 본 hook 로직 (lock + quota check + bookkeeping) ...
 *   g_in_hook = 0;
 *
 * 모든 return 경로에서 g_in_hook 을 0 으로 되돌려야 한다 — 빠뜨리면
 * 그 스레드의 후속 hook 호출이 영영 skip 된다.
 * ===================================================================== */
static __thread int g_in_hook = 0;


/* =====================================================================
 * (2-ter) Kernel launch counter — Stage 7
 *
 *   g_launch_count    : 누적 cudaLaunchKernel 호출 횟수.
 *                       __atomic_fetch_add 로 lock 없이 증가.
 *                       PyTorch 등은 launch 를 초당 수천 번 호출하므로
 *                       mutex 비용을 피하기 위함. RELAXED memory order
 *                       로 충분 — 단순 monotonic counter.
 *
 *   g_launch_log_every : N 번마다 한 번씩 stderr 로 누적값 dump.
 *                        FGPU_LAUNCH_LOG_EVERY env 로 조정.
 *                        0 이면 dump off (overhead 측정 시 사용).
 *                        기본 1000.
 *
 * 누적값의 *최종* dump 는 atexit() 콜백에서 한 번 더 한다 — 정상 종료
 * 경로만 보장 (signal kill / _exit 는 best-effort).
 * ===================================================================== */
static size_t       g_launch_count     = 0;
static unsigned int g_launch_log_every = 1000;
static int          g_atexit_registered = 0;

static void fgpu_launch_atexit_dump(void) {
    /* atexit 호출 시점엔 다른 thread 가 hook 진입 중일 수 있으나,
     * 단일 size_t read 라 race 가 데이터 손상으로 이어지진 않음. */
    size_t n = __atomic_load_n(&g_launch_count, __ATOMIC_RELAXED);
    fprintf(stderr,
            "[fgpu] exit summary: total cudaLaunchKernel = %zu\n",
            n);
}


/* =====================================================================
 * (3) "포인터 -> 할당 크기" 추적용 단일 연결 리스트
 *
 * cudaFree(ptr) 가 호출될 때 우리는 "그 ptr 이 몇 바이트짜리였는지"
 * 알아야 g_used 를 정확히 줄일 수 있다. CUDA 자체는 그 정보를 우리에게
 * 알려주지 않으므로, 우리가 직접 "ptr -> size" 매핑을 들고 있어야 한다.
 *
 * 자료구조 선택 이유:
 *  - 프로토타입 단계에서는 동시에 살아있는 할당이 보통 수십~수백 개를
 *    넘지 않는다. 그래서 단순 linked list 면 충분하다.
 *  - 성능이 문제가 되면 hash map 으로 교체하면 된다 (Stage 5+).
 * ===================================================================== */
typedef struct alloc_entry {
    void               *ptr;   /* GPU 디바이스 포인터 */
    size_t              size;  /* 할당된 바이트 수 */
    struct alloc_entry *next;  /* 다음 노드 */
} alloc_entry_t;

static alloc_entry_t *g_allocs = NULL;  /* 리스트의 head */

/* track_alloc:
 *   새 할당을 리스트 맨 앞에 push 한다 (O(1)).
 *   ※ 이 함수는 g_lock 이 잡힌 상태에서만 호출되어야 한다.
 */
static void track_alloc(void *ptr, size_t size) {
    alloc_entry_t *e = (alloc_entry_t *)malloc(sizeof(*e));
    if (!e) {
        /* malloc 실패는 사실상 메모리 고갈이라 복구 불가 — 일단 무시.
         * 단, 그러면 cudaFree 시 size 를 못 찾아 g_used 가 누적된다.
         * Stage 1 에서는 알고도 받아들이는 한계로 둔다. */
        return;
    }
    e->ptr  = ptr;
    e->size = size;
    e->next = g_allocs;
    g_allocs = e;
}

/* pop_alloc:
 *   리스트에서 ptr 을 찾아 제거하고 그 size 를 돌려준다.
 *   못 찾으면 0 을 돌려준다 (예: hook 이 늦게 붙어 추적 못 한 ptr).
 *   ※ 이 함수도 g_lock 이 잡힌 상태에서만 호출되어야 한다.
 */
static size_t pop_alloc(void *ptr) {
    alloc_entry_t **cur = &g_allocs;  /* 이중 포인터로 노드 제거를 단순화 */
    while (*cur) {
        if ((*cur)->ptr == ptr) {
            size_t s = (*cur)->size;
            alloc_entry_t *dead = *cur;
            *cur = dead->next;        /* 리스트에서 dead 노드 분리 */
            free(dead);
            return s;
        }
        cur = &(*cur)->next;
    }
    return 0;
}


/* =====================================================================
 * (4) 초기화 루틴
 *
 * 두 단계로 나누어져 있다:
 *
 *   fgpu_init_locked()
 *       - 진짜 cudaMalloc/cudaFree 의 주소를 dlsym 으로 가져온다.
 *       - 환경변수(FGPU_RATIO, FGPU_QUOTA_BYTES) 를 읽는다.
 *       - 한 번만 실행된다 (g_inited 플래그로 보장).
 *
 *   compute_quota_if_needed_locked()
 *       - g_quota 가 아직 0 이면, cudaMemGetInfo 로 GPU 의 *전체*
 *         메모리 양을 알아낸 뒤 ratio 를 곱해 quota 를 정한다.
 *       - 이 호출은 *cudaMalloc 안에서* 한다. 왜? cudaMemGetInfo 는
 *         CUDA 컨텍스트가 만들어진 뒤에야 동작하는데, 그게 보장되는
 *         가장 빠른 시점이 사용자가 처음 cudaMalloc 을 부른 순간이기
 *         때문이다. 너무 일찍(예: 라이브러리 로드 시점) 부르면
 *         "no CUDA-capable device" 같은 엉뚱한 에러를 만난다.
 *
 * 함수 이름 끝의 _locked 는 "이 함수는 호출자가 g_lock 을 *이미*
 * 잡고 있어야 한다" 를 표시하는 관습이다. 안 잡고 부르면 race condition.
 * ===================================================================== */
static void fgpu_init_locked(void) {
    /* (a) 진짜 함수 주소를 lazy 하게 가져온다. NULL 인 항목만 시도하므로
     *     매 호출마다 dlsym 이 한 번씩 도는 듯 보여도, 채워지는 즉시
     *     이후 호출은 if 문 통과 only — 비용 무시 가능.
     *
     *     "한 번만 init" 으로 짜지 않는 이유: hook .so 가 먼저 로드된
     *     뒤에 사용자 프로그램이 dlopen("libcuda.so") 를 늦게 부르는
     *     케이스가 있다. 그 경우 첫 init 시점엔 driver 심볼이 NULL 이고,
     *     실제 cuMemAlloc_v2 호출은 그 이후. lazy retry 로 자연스럽게
     *     해결.
     *
     *     RTLD_NEXT = "지금 이 .so 다음에 로드된 라이브러리에서 찾아라". */
    if (!real_cudaMalloc)       real_cudaMalloc       = dlsym(RTLD_NEXT, "cudaMalloc");
    if (!real_cudaFree)         real_cudaFree         = dlsym(RTLD_NEXT, "cudaFree");
    if (!real_cuMemAlloc_v2)    real_cuMemAlloc_v2    = dlsym(RTLD_NEXT, "cuMemAlloc_v2");
    if (!real_cuMemFree_v2)     real_cuMemFree_v2     = dlsym(RTLD_NEXT, "cuMemFree_v2");
    if (!real_cuMemCreate)      real_cuMemCreate      = dlsym(RTLD_NEXT, "cuMemCreate");
    if (!real_cuMemRelease)     real_cuMemRelease     = dlsym(RTLD_NEXT, "cuMemRelease");
    if (!real_cudaLaunchKernel) real_cudaLaunchKernel = dlsym(RTLD_NEXT, "cudaLaunchKernel");

    if (g_inited) return;  /* 환경변수 읽기 + 1회 init 로그는 한 번만. */

    /* (b) FGPU_RATIO 환경변수 읽기.
     *     getenv 는 변수 미설정 시 NULL 을 돌려준다. 그러면 1.0 (= 제한 없음). */
    const char *ratio_env = getenv("FGPU_RATIO");
    g_ratio = ratio_env ? atof(ratio_env) : 1.0;
    if (g_ratio <= 0.0 || g_ratio > 1.0) {
        /* 0.0 이하나 1.0 초과는 의미 없는 값 → 안전하게 1.0 으로 무효화. */
        g_ratio = 1.0;
    }

    /* (c) FGPU_QUOTA_BYTES 가 명시적으로 주어지면 ratio 보다 우선시.
     *     절대값(bytes) 으로 quota 를 직접 지정할 수 있는 escape hatch. */
    const char *abs_env = getenv("FGPU_QUOTA_BYTES");
    if (abs_env) {
        g_quota = (size_t)strtoull(abs_env, NULL, 10);
    }

    /* (d) launch counter 의 dump 주기.
     *     0 = off (overhead 측정 시), default 1000. */
    const char *every_env = getenv("FGPU_LAUNCH_LOG_EVERY");
    if (every_env) {
        long v = strtol(every_env, NULL, 10);
        g_launch_log_every = (v >= 0) ? (unsigned int)v : 1000;
    }

    /* (e) atexit 으로 누적 launch 수의 최종 dump 등록 — 한 번만. */
    if (!g_atexit_registered) {
        atexit(fgpu_launch_atexit_dump);
        g_atexit_registered = 1;
    }

    fprintf(stderr,
            "[fgpu] init: ratio=%.3f quota_bytes=%zu (0 = lazy 계산)\n",
            g_ratio, g_quota);
    fprintf(stderr,
            "[fgpu] init: real cudaMalloc=%p cudaFree=%p "
            "cuMemAlloc_v2=%p cuMemFree_v2=%p\n"
            "[fgpu] init: real cuMemCreate=%p cuMemRelease=%p "
            "cudaLaunchKernel=%p\n",
            (void *)real_cudaMalloc,    (void *)real_cudaFree,
            (void *)real_cuMemAlloc_v2, (void *)real_cuMemFree_v2,
            (void *)real_cuMemCreate,   (void *)real_cuMemRelease,
            (void *)real_cudaLaunchKernel);
    fprintf(stderr,
            "[fgpu] init: launch_log_every=%u (0=off)\n",
            g_launch_log_every);

    g_inited = 1;
}

static void compute_quota_if_needed_locked(void) {
    if (g_quota != 0) return;  /* 이미 정해졌으면 건드리지 않는다. */

    size_t free_b  = 0;  /* 현재 사용 가능한 GPU 메모리 (다른 프로세스 쓴 것 제외) */
    size_t total_b = 0;  /* GPU 의 전체 메모리 (RTX 4060 = 8 GB) */
    cudaError_t r = cudaMemGetInfo(&free_b, &total_b);
    if (r == cudaSuccess && total_b > 0) {
        g_quota = (size_t)((double)total_b * g_ratio);
        fprintf(stderr,
                "[fgpu] quota lazily 계산: ratio=%.3f * total=%zu = %zu bytes\n",
                g_ratio, total_b, g_quota);
    } else {
        /* 컨텍스트가 아직 없거나 device 가 없을 때 — 다음 호출에서 재시도. */
        fprintf(stderr,
                "[fgpu] cudaMemGetInfo 실패 (err=%d); quota 미설정 유지\n", r);
    }
}


/* =====================================================================
 * (5) 후킹 함수: 진짜 cudaMalloc 을 우리 것으로 대체
 *
 * 사용자 프로그램이 cudaMalloc(&p, N) 을 부르면 *이 함수* 가 호출된다.
 *
 * 처리 흐름:
 *   ① g_lock 을 잡는다 (멀티스레드 안전).
 *   ② 초기화가 안 됐으면 한다 (dlsym, env 읽기).
 *   ③ quota 가 아직 0 이면 cudaMemGetInfo 로 계산한다.
 *   ④ "이번 할당을 더하면 quota 를 넘는가?" 검사.
 *      넘으면 → 진짜 cudaMalloc 을 *부르지도 않고* 에러를 돌려준다.
 *   ⑤ 안 넘으면 → 진짜 cudaMalloc 을 부른다.
 *      성공 시 g_used 증가 + 포인터 추적 리스트에 등록.
 *   ⑥ g_lock 을 푼다.
 *   ⑦ cudaError_t 를 사용자에게 돌려준다.
 *
 * 반환 값:
 *   quota 초과 시 cudaErrorMemoryAllocation (= 2) 을 돌려준다.
 *   이는 PyTorch / cuBLAS 등이 "GPU OOM" 으로 인식하는 표준 에러다.
 * ===================================================================== */
cudaError_t cudaMalloc(void **devPtr, size_t size) {
    /* Reentrancy 가드: 다른 hook (예: cuMemAlloc_v2) 에서 진짜 함수를
     * 호출하다 cudaMalloc 으로 다시 들어오면, bookkeeping skip 하고
     * 위임만. 이렇게 안 하면 g_used 가 두 번 더해지고 deadlock 도 가능. */
    if (g_in_hook) {
        return real_cudaMalloc ? real_cudaMalloc(devPtr, size)
                               : cudaErrorInitializationError;
    }
    g_in_hook = 1;
    pthread_mutex_lock(&g_lock);
    fgpu_init_locked();
    compute_quota_if_needed_locked();

    /* (a) quota 검사. g_quota == 0 이면 (lazy 계산 실패) 검사 skip. */
    if (g_quota > 0 && g_used + size > g_quota) {
        fprintf(stderr,
                "[fgpu] DENY  cudaMalloc size=%zu used=%zu quota=%zu\n",
                size, g_used, g_quota);
        pthread_mutex_unlock(&g_lock);
        g_in_hook = 0;
        return cudaErrorMemoryAllocation;
    }

    /* (b) 진짜 cudaMalloc 호출 위임. */
    cudaError_t err = real_cudaMalloc(devPtr, size);
    if (err == cudaSuccess) {
        g_used += size;
        track_alloc(*devPtr, size);
        fprintf(stderr,
                "[fgpu] ALLOW cudaMalloc ptr=%p size=%zu used=%zu/%zu\n",
                *devPtr, size, g_used, g_quota);
    } else {
        /* CUDA 자체가 실패 (예: GPU 가 정말로 메모리 부족) — 추적 안 함. */
        fprintf(stderr,
                "[fgpu] FAIL  cudaMalloc size=%zu cuda_err=%d\n", size, err);
    }
    pthread_mutex_unlock(&g_lock);
    g_in_hook = 0;
    return err;
}

/* =====================================================================
 * (6) 후킹 함수: cudaFree
 *
 * 처리 흐름:
 *   ① 진짜 cudaFree 를 부른다 (실패해도 호출 자체는 통과시키는 게 안전).
 *   ② 성공한 경우, 추적 리스트에서 ptr 을 제거하면서 size 를 회수.
 *   ③ g_used 에서 그 size 를 뺀다.
 *
 * 주의:
 *   - cudaFree(NULL) 은 CUDA 표준상 no-op 이다. 우리도 추적 갱신 안 함.
 *   - hook 이 붙기 전에 할당된 ptr 을 free 하는 경우 size = 0 으로 처리.
 * ===================================================================== */
cudaError_t cudaFree(void *devPtr) {
    if (g_in_hook) {
        return real_cudaFree ? real_cudaFree(devPtr)
                             : cudaErrorInitializationError;
    }
    g_in_hook = 1;
    pthread_mutex_lock(&g_lock);
    fgpu_init_locked();

    cudaError_t err = real_cudaFree(devPtr);
    if (err == cudaSuccess && devPtr != NULL) {
        size_t freed = pop_alloc(devPtr);
        if (freed > 0 && freed <= g_used) {
            g_used -= freed;
        }
        fprintf(stderr,
                "[fgpu] FREE  ptr=%p size=%zu used=%zu/%zu\n",
                devPtr, freed, g_used, g_quota);
    }
    pthread_mutex_unlock(&g_lock);
    g_in_hook = 0;
    return err;
}


/* =====================================================================
 * (7) Stage 5-C: Driver API hook  —  cuMemAlloc_v2 / cuMemFree_v2
 *
 * Runtime API 의 cudaMalloc/cudaFree 와 거의 동형(同形) 이지만 몇 가지
 * 다름:
 *
 *   - 반환 타입이 cudaError_t 가 아니라 CUresult (둘 다 enum 인데 별도).
 *   - 포인터 타입이 void* 가 아니라 CUdeviceptr (= unsigned long long).
 *     x86_64 Linux 에서는 device pointer 가 64-bit 정수로 표현되므로
 *     uintptr_t 캐스트로 우리 g_allocs 리스트의 void* 슬롯에 안전히
 *     끼워넣을 수 있다. (다른 ABI 였으면 위험할 수 있음.)
 *
 *   - 함수 이름 끝의 "_v2" 는 CUDA 4.0 에서 ABI 가 바뀐 뒤 유지되는
 *     공식 심볼명. cuMemAlloc (no _v2) 는 이전 호환용 macro 로 cuda.h
 *     안에서 _v2 로 redirect 된다. dlsym 으로 잡을 때는 _v2 명시 필수.
 *
 * 같은 g_used / g_quota / g_lock / g_allocs 을 Runtime hook 과 공유한다.
 * Reentrancy guard 로 양 layer 가 서로의 호출에 의해 이중 카운트되는
 * 문제를 방지한다 (사이드 케이스 — 보통은 cudart 가 PLT 우회 직접 호출
 * 이라 발생 안 하지만, 보험).
 * ===================================================================== */
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    if (g_in_hook) {
        return real_cuMemAlloc_v2 ? real_cuMemAlloc_v2(dptr, bytesize)
                                  : CUDA_ERROR_NOT_INITIALIZED;
    }
    g_in_hook = 1;
    pthread_mutex_lock(&g_lock);
    fgpu_init_locked();
    compute_quota_if_needed_locked();

    if (!real_cuMemAlloc_v2) {
        /* 매우 드문 케이스 — libcuda 가 dlsym 후에도 잡히지 않음.
         * 위임할 곳이 없으므로 표준 에러로 거절. */
        fprintf(stderr,
                "[fgpu] FAIL  cuMemAlloc_v2: real symbol not resolved\n");
        pthread_mutex_unlock(&g_lock);
        g_in_hook = 0;
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    if (g_quota > 0 && g_used + bytesize > g_quota) {
        fprintf(stderr,
                "[fgpu] DENY  cuMemAlloc_v2 size=%zu used=%zu quota=%zu\n",
                bytesize, g_used, g_quota);
        pthread_mutex_unlock(&g_lock);
        g_in_hook = 0;
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    CUresult err = real_cuMemAlloc_v2(dptr, bytesize);
    if (err == CUDA_SUCCESS) {
        g_used += bytesize;
        /* CUdeviceptr → void* 캐스트 (x86_64 Linux 가정). 같은 64-bit
         * 정수 표현이므로 단순 reinterpret 가 안전. */
        track_alloc((void *)(uintptr_t)(*dptr), bytesize);
        fprintf(stderr,
                "[fgpu] ALLOW cuMemAlloc_v2 ptr=0x%llx size=%zu used=%zu/%zu\n",
                (unsigned long long)(*dptr), bytesize, g_used, g_quota);
    } else {
        fprintf(stderr,
                "[fgpu] FAIL  cuMemAlloc_v2 size=%zu cu_err=%d\n",
                bytesize, (int)err);
    }
    pthread_mutex_unlock(&g_lock);
    g_in_hook = 0;
    return err;
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    if (g_in_hook) {
        return real_cuMemFree_v2 ? real_cuMemFree_v2(dptr)
                                 : CUDA_ERROR_NOT_INITIALIZED;
    }
    g_in_hook = 1;
    pthread_mutex_lock(&g_lock);
    fgpu_init_locked();

    if (!real_cuMemFree_v2) {
        fprintf(stderr,
                "[fgpu] FAIL  cuMemFree_v2: real symbol not resolved\n");
        pthread_mutex_unlock(&g_lock);
        g_in_hook = 0;
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    CUresult err = real_cuMemFree_v2(dptr);
    if (err == CUDA_SUCCESS && dptr != 0) {
        size_t freed = pop_alloc((void *)(uintptr_t)dptr);
        if (freed > 0 && freed <= g_used) {
            g_used -= freed;
        }
        fprintf(stderr,
                "[fgpu] FREE  cuMemFree_v2 ptr=0x%llx size=%zu used=%zu/%zu\n",
                (unsigned long long)dptr, freed, g_used, g_quota);
    }
    pthread_mutex_unlock(&g_lock);
    g_in_hook = 0;
    return err;
}


/* =====================================================================
 * (7-bis) Stage 6: VMM API hook  —  cuMemCreate / cuMemRelease
 *
 * VMM (Virtual Memory Management) API 는 CUDA 10.2 부터 도입된 modern
 * allocation 경로. Classical cuMemAlloc 과 다른 점:
 *
 *   - cuMemCreate  : *물리* 메모리만 할당. 사용자에게 handle 을 돌려줌.
 *   - cuMemAddressReserve : VA range 예약 (물리 변화 X — 후킹 안 함).
 *   - cuMemMap     : handle ↔ VA range 바인딩 (물리 변화 X — 후킹 안 함).
 *   - cuMemRelease : 물리 메모리 해제.
 *
 * 따라서 quota 부과·회수는 cuMemCreate / cuMemRelease 두 시점에만.
 *
 * 추적 키: handle 은 unsigned long long. 64-bit Linux 에서 void* 와 같은
 * 너비 → 기존 g_allocs 리스트에 캐스트로 끼워넣음 (cuMemAlloc_v2 와 동일
 * 패턴). 다른 layer 의 포인터/handle 과 *값* 이 충돌할 이론적 가능성은
 * 무시 가능 수준 — opaque 토큰 공간이 매우 큼.
 *
 * 의도적 미구현:
 *   - cuMemMap 자체는 후킹 안 함 — 같은 handle 을 여러 VA 에 매핑해도
 *     물리량은 cuMemCreate 시점 그대로. quota state 변화 없음.
 *   - cuMemAllocAsync / cuMemAllocManaged 도 별개 경로 — 추가 후킹은
 *     stage 6.x / 6.y 후속 작업.
 * ===================================================================== */
CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                     const CUmemAllocationProp *prop,
                     unsigned long long flags) {
    if (g_in_hook) {
        return real_cuMemCreate
               ? real_cuMemCreate(handle, size, prop, flags)
               : CUDA_ERROR_NOT_INITIALIZED;
    }
    g_in_hook = 1;
    pthread_mutex_lock(&g_lock);
    fgpu_init_locked();
    compute_quota_if_needed_locked();

    if (!real_cuMemCreate) {
        fprintf(stderr,
                "[fgpu] FAIL  cuMemCreate: real symbol not resolved\n");
        pthread_mutex_unlock(&g_lock);
        g_in_hook = 0;
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    if (g_quota > 0 && g_used + size > g_quota) {
        fprintf(stderr,
                "[fgpu] DENY  cuMemCreate size=%zu used=%zu quota=%zu\n",
                size, g_used, g_quota);
        pthread_mutex_unlock(&g_lock);
        g_in_hook = 0;
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    CUresult err = real_cuMemCreate(handle, size, prop, flags);
    if (err == CUDA_SUCCESS) {
        g_used += size;
        /* handle 은 ulong long. 추적 리스트 키로 cast. */
        track_alloc((void *)(uintptr_t)(*handle), size);
        fprintf(stderr,
                "[fgpu] ALLOW cuMemCreate handle=0x%llx size=%zu used=%zu/%zu\n",
                (unsigned long long)(*handle), size, g_used, g_quota);
    } else {
        fprintf(stderr,
                "[fgpu] FAIL  cuMemCreate size=%zu cu_err=%d\n",
                size, (int)err);
    }
    pthread_mutex_unlock(&g_lock);
    g_in_hook = 0;
    return err;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    if (g_in_hook) {
        return real_cuMemRelease
               ? real_cuMemRelease(handle)
               : CUDA_ERROR_NOT_INITIALIZED;
    }
    g_in_hook = 1;
    pthread_mutex_lock(&g_lock);
    fgpu_init_locked();

    if (!real_cuMemRelease) {
        fprintf(stderr,
                "[fgpu] FAIL  cuMemRelease: real symbol not resolved\n");
        pthread_mutex_unlock(&g_lock);
        g_in_hook = 0;
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    CUresult err = real_cuMemRelease(handle);
    if (err == CUDA_SUCCESS && handle != 0) {
        size_t freed = pop_alloc((void *)(uintptr_t)handle);
        if (freed > 0 && freed <= g_used) {
            g_used -= freed;
        }
        fprintf(stderr,
                "[fgpu] FREE  cuMemRelease handle=0x%llx size=%zu used=%zu/%zu\n",
                (unsigned long long)handle, freed, g_used, g_quota);
    }
    pthread_mutex_unlock(&g_lock);
    g_in_hook = 0;
    return err;
}


/* =====================================================================
 * (8) Stage 7: cudaLaunchKernel 후킹  —  단순 launch counter
 *
 * 의도적으로 quota 시행은 안 한다. 이유:
 *   1) 진짜 SM 격리는 hook 으로 못 함 (MIG/MPS 영역).
 *   2) launch 거부는 사용자 코드 입장에서 매우 거친 동작 — 일반적으로
 *      memory quota 만큼 깔끔히 모델링되지 않는다.
 *   3) 본 프로토타입의 가치 명제는 "*측정* 가능 + 정책적 스케줄러 가
 *      그 측정에 기반해 fairness 결정" 이지 "강제 차단" 이 아님.
 *
 * 따라서 이 hook 은:
 *   - 호출 횟수만 atomic 으로 카운트.
 *   - g_launch_log_every 마다 stderr 로 누적값 dump.
 *   - atexit 시점에 final summary 한 번 더 dump.
 *
 * Lock 안 잡는 이유: PyTorch 가 launch 를 초당 수천 회 호출하므로
 * mutex 가 hot path 가 됨. atomic 으로 충분.
 *
 * Reentrancy guard 는 alloc hook 과 일관성 위해 그대로 적용.
 * cudaLaunchKernel 이 내부적으로 cudaMalloc 부르는 경로는 거의 없지만
 * 보험.
 * ===================================================================== */
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream) {
    if (g_in_hook) {
        return real_cudaLaunchKernel
               ? real_cudaLaunchKernel(func, gridDim, blockDim,
                                       args, sharedMem, stream)
               : cudaErrorInitializationError;
    }

    /* dlsym 이 아직 안 된 경우 — 매우 드물지만 첫 호출 보호 */
    if (!real_cudaLaunchKernel) {
        pthread_mutex_lock(&g_lock);
        fgpu_init_locked();
        pthread_mutex_unlock(&g_lock);
        if (!real_cudaLaunchKernel) {
            fprintf(stderr,
                    "[fgpu] FAIL  cudaLaunchKernel: real symbol not resolved\n");
            return cudaErrorInitializationError;
        }
    }

    g_in_hook = 1;
    /* lock-free atomic 증가 — RELAXED 로 충분 (단순 monotonic counter). */
    size_t count = __atomic_add_fetch(&g_launch_count, 1, __ATOMIC_RELAXED);

    cudaError_t err = real_cudaLaunchKernel(func, gridDim, blockDim,
                                            args, sharedMem, stream);
    g_in_hook = 0;

    /* 주기적 dump — log_every == 0 이면 off. */
    if (g_launch_log_every > 0 && (count % g_launch_log_every) == 0) {
        fprintf(stderr,
                "[fgpu] LAUNCH count=%zu (every %u)\n",
                count, g_launch_log_every);
    }
    return err;
}
