/*
 * test_alloc.cu — minimal driver for the Stage-1 hook.
 *
 * Allocates two buffers of well-known size so the hook log lines clearly
 * show ALLOW / DENY based on the FGPU_RATIO you pass in.
 *
 *   alloc1 = 256 MiB
 *   alloc2 = 6   GiB   (intentionally > 4060's 8GB * 0.4 ratio)
 */

#include <cstdio>
#include <cuda_runtime.h>

static void show_mem(const char *tag) {
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    printf("[test] %s : free=%zu total=%zu\n", tag, free_b, total_b);
}

int main() {
    show_mem("before");

    void *p1 = nullptr;
    cudaError_t e1 = cudaMalloc(&p1, 256ULL * 1024 * 1024);
    printf("[test] alloc1 (256MiB) -> err=%d ptr=%p\n", e1, p1);

    void *p2 = nullptr;
    cudaError_t e2 = cudaMalloc(&p2, 6ULL * 1024 * 1024 * 1024);
    printf("[test] alloc2 (6GiB)   -> err=%d ptr=%p\n", e2, p2);

    show_mem("after-allocs");

    if (p1) {
        cudaError_t f = cudaFree(p1);
        printf("[test] free p1 -> err=%d\n", f);
    }
    if (p2) {
        cudaError_t f = cudaFree(p2);
        printf("[test] free p2 -> err=%d\n", f);
    }

    show_mem("after-frees");
    return 0;
}
