// =====================================================================
// Stage 6 검증: VMM API hook (cuMemCreate / cuMemRelease).
//
// 의도적으로 cuMemAddressReserve/cuMemMap 은 안 부른다 — 본 hook 의
// quota 모델이 *물리 alloc* (cuMemCreate) 만 잡기 때문에, VA reservation
// 없는 상태로도 ALLOW/DENY 가 검증되어야 한다.
//
// FGPU_RATIO=0.4 (RTX 4060/8 GB → quota ≈ 3.2 GiB):
//   - 256 MiB cuMemCreate → ALLOW
//   - 6   GiB cuMemCreate → DENY (CUDA_ERROR_OUT_OF_MEMORY)
//
// VMM API 는 size 가 granularity 의 배수여야 한다. cuMemGetAllocationGranularity
// 로 조회 후 round up — 일반적으로 device-pinned 의 minimum granularity 는
// 2 MiB. 우리 입력 사이즈 (256 MiB, 6 GiB) 는 이미 정렬됨이라 round up 이
// no-op 이지만 코드는 일반화.
// =====================================================================

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

static const char *cu_strerror(CUresult r) {
    const char *s = NULL;
    cuGetErrorName(r, &s);
    return s ? s : "<unknown>";
}

static size_t round_up(size_t n, size_t align) {
    return ((n + align - 1) / align) * align;
}

static void try_alloc(size_t bytes, const char *label,
                      const CUmemAllocationProp *prop,
                      size_t granularity) {
    size_t aligned = round_up(bytes, granularity);
    fprintf(stdout, "[test-vmm] %s: cuMemCreate(%zu → %zu bytes after align)\n",
            label, bytes, aligned);
    CUmemGenericAllocationHandle h = 0;
    CUresult r = cuMemCreate(&h, aligned, prop, 0);
    fprintf(stdout, "[test-vmm] %s: result=%d (%s)\n",
            label, (int)r, cu_strerror(r));
    if (r == CUDA_SUCCESS) {
        cuMemRelease(h);
        fprintf(stdout, "[test-vmm] %s: released.\n", label);
    }
}

int main(void) {
    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "[test-vmm] cuInit failed: %d (%s)\n",
                (int)r, cu_strerror(r));
        return 1;
    }

    CUdevice dev;
    if ((r = cuDeviceGet(&dev, 0)) != CUDA_SUCCESS) {
        fprintf(stderr, "[test-vmm] cuDeviceGet failed: %d\n", (int)r);
        return 1;
    }
    char name[128] = {0};
    cuDeviceGetName(name, sizeof(name) - 1, dev);
    fprintf(stdout, "[test-vmm] device 0 = %s\n", name);

    CUcontext ctx = NULL;
    if ((r = cuCtxCreate(&ctx, 0, dev)) != CUDA_SUCCESS) {
        fprintf(stderr, "[test-vmm] cuCtxCreate failed: %d\n", (int)r);
        return 1;
    }

    // VMM allocation property: device-pinned on device 0.
    CUmemAllocationProp prop = {};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = (int)dev;

    size_t granularity = 0;
    r = cuMemGetAllocationGranularity(&granularity, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "[test-vmm] cuMemGetAllocationGranularity failed: %d\n",
                (int)r);
        return 1;
    }
    fprintf(stdout, "[test-vmm] minimum granularity = %zu bytes\n", granularity);

    // 1) 256 MiB → ALLOW (모든 합리적 ratio 에서 통과)
    // 2) 6   GiB → DENY  (FGPU_RATIO=0.4 일 때 quota 3.2 GiB 초과)
    try_alloc((size_t)256 * 1024 * 1024,             "alloc#1 (256 MiB)",
              &prop, granularity);
    try_alloc((size_t)6   * 1024 * 1024 * 1024,      "alloc#2 (6 GiB)",
              &prop, granularity);

    cuCtxDestroy(ctx);
    fprintf(stdout, "[test-vmm] done.\n");
    return 0;
}
