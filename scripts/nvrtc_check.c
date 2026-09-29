/* Compile emitted CUDA the way the runtime does: NVRTC, no system headers.
 *
 * `nvcc` is not the compiler that runs. It has the full toolchain's
 * headers, so a kernel that leans on <math.h> -- `INFINITY`, say --
 * compiles here and then fails inside `Gpu::build_lowered` on a real
 * machine, which is NVRTC in-process. That went unnoticed until a rented
 * L4 said so, with cuda_check.sh green.
 *
 * Built and run by scripts/cuda_check.sh inside the CUDA container; needs
 * no GPU, only libnvrtc.
 */
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    *len = fread(buf, 1, (size_t)n, f);
    buf[*len] = 0;
    fclose(f);
    return buf;
}

int main(int argc, char **argv) {
    const char *arch = getenv("NVRTC_ARCH");
    if (!arch) arch = "--gpu-architecture=compute_89";
    /* The same include the runtime probes for, so cuda_fp16.h resolves. */
    const char *opts[] = {arch, "-I/usr/local/cuda/include"};
    int failed = 0;
    for (int i = 1; i < argc; i++) {
        size_t n = 0;
        char *src = slurp(argv[i], &n);
        if (!src) { fprintf(stderr, "cannot read %s\n", argv[i]); failed = 1; continue; }
        nvrtcProgram prog;
        if (nvrtcCreateProgram(&prog, src, argv[i], 0, NULL, NULL) != NVRTC_SUCCESS) {
            fprintf(stderr, "nvrtcCreateProgram failed for %s\n", argv[i]);
            failed = 1; free(src); continue;
        }
        nvrtcResult r = nvrtcCompileProgram(prog, 2, opts);
        if (r != NVRTC_SUCCESS) {
            size_t ln = 0;
            nvrtcGetProgramLogSize(prog, &ln);
            char *log = malloc(ln + 1);
            if (log) { nvrtcGetProgramLog(prog, log); log[ln] = 0;
                       fprintf(stderr, "=== %s\n%s\n", argv[i], log); free(log); }
            failed = 1;
        }
        nvrtcDestroyProgram(&prog);
        free(src);
    }
    if (!failed) printf("nvrtc: %d kernels compiled\n", argc - 1);
    return failed;
}
