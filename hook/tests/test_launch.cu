// =====================================================================
// Stage 7 검증 바이너리: cudaLaunchKernel 후킹.
//
// 작은 noop kernel 을 N 번 launch. quota 와 무관 — launch counter 만
// 검증. FGPU_LAUNCH_LOG_EVERY=100 + N=1000 이면 hook stderr 에
// "[fgpu] LAUNCH count=100/200/.../1000" 라인 10개 + atexit 의
// "[fgpu] exit summary: total cudaLaunchKernel = 1000" 한 줄.
//
// env 로 override:
//   PYTEST_LAUNCH_N    기본 1000  (총 launch 횟수)
//
// 종료 코드:
//   0  성공
//   1  CUDA error
// =====================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void noop_kernel(int *p) {
    /* 의도적으로 일하는 척만 — counter 검증이 목적이라
     * GPU 작업 자체는 최소화. */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        atomicAdd(p, 1);
    }
}

#define CHECK(call)                                                       \
    do {                                                                  \
        cudaError_t _e = (call);                                          \
        if (_e != cudaSuccess) {                                          \
            fprintf(stderr, "[test-launch] %s -> %s\n",                   \
                    #call, cudaGetErrorString(_e));                       \
            return 1;                                                     \
        }                                                                 \
    } while (0)

int main(void) {
    int n = 1000;
    const char *n_env = getenv("PYTEST_LAUNCH_N");
    if (n_env) {
        long v = strtol(n_env, NULL, 10);
        if (v > 0) n = (int)v;
    }

    int *d_counter = NULL;
    CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    fprintf(stdout, "[test-launch] launching %d kernels ...\n", n);
    fflush(stdout);

    /* 작은 grid/block — kernel 자체 비용 무시 가능. */
    for (int i = 0; i < n; i++) {
        noop_kernel<<<32, 64>>>(d_counter);
    }
    CHECK(cudaDeviceSynchronize());

    int host_counter = 0;
    CHECK(cudaMemcpy(&host_counter, d_counter, sizeof(int),
                     cudaMemcpyDeviceToHost));
    fprintf(stdout, "[test-launch] launches done, kernel atomics = %d "
                    "(expected = n = %d)\n",
            host_counter, n);

    cudaFree(d_counter);
    return 0;
}
