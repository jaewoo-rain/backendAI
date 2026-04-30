// =====================================================================
// Stage 5-C 검증 바이너리: Driver API (cuMemAlloc_v2 / cuMemFree_v2)
// 만 사용해서 fGPU hook 의 driver-layer 후킹이 동작하는지 확인.
//
// Runtime API (cudaMalloc 등) 는 *일부러* 안 부른다 — 그래야 "오직
// driver hook 만으로도 quota 가 강제되는가" 가 깨끗하게 검증됨.
//
// FGPU_RATIO=0.4 (RTX 4060/8 GB → quota ≈ 3.2 GiB) 시나리오:
//   - 256 MiB cuMemAlloc_v2  → ALLOW (small)
//   - 6   GiB cuMemAlloc_v2  → DENY  (over quota)
//   호출 결과 코드는 stdout 으로, 우리 [fgpu] 로그는 stderr 로 분리.
//
// 빌드 메모:
//   nvcc 로 빌드 + -lcuda (driver). cudart 는 안 쓰지만, nvcc 가 default
//   로 -lcudart 를 붙이는 점은 무시 가능 (호출 안 하므로 동적 링크만).
// =====================================================================

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

static const char *cu_strerror(CUresult r) {
    const char *s = NULL;
    cuGetErrorName(r, &s);
    return s ? s : "<unknown>";
}

static void try_alloc(size_t bytes, const char *label) {
    CUdeviceptr dp = 0;
    fprintf(stdout, "[test-driver] %s: cuMemAlloc_v2(%zu bytes)\n",
            label, bytes);
    CUresult r = cuMemAlloc_v2(&dp, bytes);
    fprintf(stdout, "[test-driver] %s: result=%d (%s)\n",
            label, (int)r, cu_strerror(r));
    if (r == CUDA_SUCCESS) {
        cuMemFree_v2(dp);
        fprintf(stdout, "[test-driver] %s: freed.\n", label);
    }
}

int main(void) {
    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "[test-driver] cuInit failed: %d (%s)\n",
                (int)r, cu_strerror(r));
        return 1;
    }

    CUdevice dev;
    r = cuDeviceGet(&dev, 0);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "[test-driver] cuDeviceGet failed: %d\n", (int)r);
        return 1;
    }
    char name[128] = {0};
    cuDeviceGetName(name, sizeof(name) - 1, dev);
    fprintf(stdout, "[test-driver] device 0 = %s\n", name);

    CUcontext ctx = NULL;
    r = cuCtxCreate(&ctx, 0, dev);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "[test-driver] cuCtxCreate failed: %d\n", (int)r);
        return 1;
    }

    // FGPU_RATIO=0.4 시나리오의 demarcation:
    //   1) 256 MiB → ALLOW  (모든 합리적 ratio 에서 통과)
    //   2) 6   GiB → DENY   (quota 3.2 GiB 초과 → CUDA_ERROR_OUT_OF_MEMORY)
    try_alloc((size_t)256 * 1024 * 1024,                "alloc#1 (256 MiB)");
    try_alloc((size_t)6   * 1024 * 1024 * 1024,         "alloc#2 (6 GiB)");

    cuCtxDestroy(ctx);
    fprintf(stdout, "[test-driver] done.\n");
    return 0;
}
