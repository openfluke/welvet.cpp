#pragma once

/* Stable C ABI for welvet.cpp — used by w2a go_v_cpp and future NAC. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(WELVET_BUILD_SHARED)
#  ifdef WELVET_EXPORTS
#    define WELVET_API __declspec(dllexport)
#  else
#    define WELVET_API __declspec(dllimport)
#  endif
#else
#  define WELVET_API __attribute__((visibility("default")))
#endif

typedef struct welvet_status {
  int32_t code; /* 0=ok, 1=invalid, 2=unsupported, 3=gap, 4=internal */
  char message[256];
} welvet_status;

/* backend: 0 = CPU tiled (exact Go CPU path), 1 = imported SIMD DotTile path */
WELVET_API welvet_status welvet_dense_forward_f32(
    int32_t batch, int32_t in, int32_t out, int32_t act, int32_t dtype, int32_t backend,
    const float* W_init, const double* bias, size_t bias_len,
    const float* x, float* pre, float* post);

/* Back-compat wrappers (backend=CPU, dtype=Float32 for weights helper). */
WELVET_API welvet_status welvet_dense_forward_f32_cpu(
    int32_t batch, int32_t in, int32_t out, int32_t act, int32_t dtype,
    const float* W_init, const double* bias, size_t bias_len,
    const float* x, float* pre, float* post);

WELVET_API welvet_status welvet_dense_forward_f32_weights(
    int32_t batch, int32_t in, int32_t out, int32_t act,
    const float* W, const double* bias, size_t bias_len,
    const float* x, float* pre, float* post);

WELVET_API const char* welvet_version(void);
WELVET_API int32_t welvet_gap_count(void);
WELVET_API void welvet_print_gaps(void);

#ifdef __cplusplus
}
#endif
