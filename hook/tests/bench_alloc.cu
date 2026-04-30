// =====================================================================
// Stage 5-D: cudaMalloc / cudaFree 호스트측 실행 시간 마이크로벤치.
//
// 목적
//   LD_PRELOAD hook 이 cudaMalloc/cudaFree 한 호출당 부가하는
//   wall-clock latency 를 정량화 — 논문 evaluation 섹션의 raw 수치.
//
// 측정 방식
//   clock_gettime(CLOCK_MONOTONIC) ns 정밀. cudaMalloc 은 host-side
//   동기 함수이므로 호스트 단일 시계로 충분 (kernel launch 와 다름).
//
//   사이즈별로:
//     1) WARMUP 회 alloc/free — 페이지 매핑·드라이버 캐시 워밍.
//     2) N 회 본 측정. 매 회 alloc → free 사이클의 alloc_ns 와 free_ns
//        를 한 줄 CSV 로 stdout 출력.
//
// 출력 (stdout)
//   header:  size_mib,iter,malloc_ns,free_ns
//   row:     1024,0,12340,4711
//
//   hook 의 [fgpu] ... 로그는 stderr 로 가므로 stdout 만 파이프하면
//   순수 CSV 가 나온다.
//
// env 로 override
//   BENCH_SIZES_MIB   기본 "16,64,256,1024"   콤마 구분
//   BENCH_N           기본 100                 사이즈별 측정 반복
//   BENCH_WARMUP      기본 5                   사이즈별 워밍업 (미측정)
//
// 종료 코드
//   0  정상 종료
//   2  CUDA 자체 에러 또는 alloc 실패가 워밍업 단계에서 발생 (환경 문제)
//   본 측정 단계의 alloc 실패는 그 iteration 만 skip — quota 초과 등은
//   비정상이지만 "측정 어려움" 으로 처리, 종료 코드는 바꾸지 않음.
// =====================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef BENCH_DEFAULT_SIZES
#define BENCH_DEFAULT_SIZES "16,64,256,1024"
#endif

static long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + (long long)ts.tv_nsec;
}

static int parse_sizes_mib(const char *s, size_t *out, int max_n) {
    int n = 0;
    char buf[256];
    strncpy(buf, s, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    for (char *tok = strtok(buf, ","); tok && n < max_n; tok = strtok(NULL, ",")) {
        long v = strtol(tok, NULL, 10);
        if (v <= 0) {
            fprintf(stderr, "[bench] invalid size '%s'\n", tok);
            return -1;
        }
        out[n++] = (size_t)v * 1024UL * 1024UL;
    }
    return n;
}

int main(void) {
    const char *sz_env  = getenv("BENCH_SIZES_MIB");
    const char *n_env   = getenv("BENCH_N");
    const char *w_env   = getenv("BENCH_WARMUP");
    int  N      = n_env ? atoi(n_env) : 100;
    int  WARMUP = w_env ? atoi(w_env) : 5;
    if (N      <= 0) N      = 100;
    if (WARMUP <  0) WARMUP = 0;

    size_t sizes[16];
    int    n_sizes = parse_sizes_mib(sz_env ? sz_env : BENCH_DEFAULT_SIZES,
                                     sizes, (int)(sizeof(sizes)/sizeof(sizes[0])));
    if (n_sizes <= 0) {
        fprintf(stderr, "[bench] no valid sizes parsed\n");
        return 2;
    }

    fprintf(stderr, "[bench] N=%d warmup=%d  sizes_mib=", N, WARMUP);
    for (int i = 0; i < n_sizes; i++) {
        fprintf(stderr, "%s%zu", i ? "," : "", sizes[i] / (1024UL * 1024UL));
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "[bench] FGPU_RATIO=%s  LD_PRELOAD=%s\n",
            getenv("FGPU_RATIO") ? getenv("FGPU_RATIO") : "<unset>",
            getenv("LD_PRELOAD")  ? getenv("LD_PRELOAD")  : "<unset>");

    // CUDA 컨텍스트 미리 초기화 — 첫 호출 비용이 측정에 섞이지 않게.
    cudaError_t e = cudaFree(0);
    if (e != cudaSuccess) {
        fprintf(stderr, "[bench] cudaFree(0) ctx init failed: %s\n",
                cudaGetErrorString(e));
        return 2;
    }

    // CSV header → stdout
    printf("size_mib,iter,malloc_ns,free_ns\n");
    fflush(stdout);

    for (int si = 0; si < n_sizes; si++) {
        size_t bytes  = sizes[si];
        size_t sz_mib = bytes / (1024UL * 1024UL);

        // ----- warmup (미측정) -----
        for (int w = 0; w < WARMUP; w++) {
            void *p = NULL;
            if (cudaMalloc(&p, bytes) != cudaSuccess) {
                fprintf(stderr, "[bench] WARMUP failed at size_mib=%zu — "
                                "환경 (quota / 메모리 부족) 확인\n", sz_mib);
                return 2;
            }
            cudaFree(p);
        }

        // ----- 본 측정 -----
        for (int i = 0; i < N; i++) {
            void *p = NULL;
            long long t0 = now_ns();
            cudaError_t er = cudaMalloc(&p, bytes);
            long long t1 = now_ns();
            if (er != cudaSuccess) {
                fprintf(stderr, "[bench] iter %d size_mib=%zu malloc FAIL: %s\n",
                        i, sz_mib, cudaGetErrorString(er));
                continue;  // skip 이 iteration
            }
            long long t2 = now_ns();
            cudaError_t fr = cudaFree(p);
            long long t3 = now_ns();
            if (fr != cudaSuccess) {
                fprintf(stderr, "[bench] iter %d size_mib=%zu free FAIL: %s\n",
                        i, sz_mib, cudaGetErrorString(fr));
                continue;
            }
            // alloc_ns = t1 - t0,  free_ns = t3 - t2  (사이 간섭 없음)
            printf("%zu,%d,%lld,%lld\n", sz_mib, i, t1 - t0, t3 - t2);
        }
        fflush(stdout);
    }

    fprintf(stderr, "[bench] done.\n");
    return 0;
}
