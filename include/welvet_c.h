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

/* Architecture baseline: Depth×Rows×Cols×LayersPerCell + remote links.
 * links: n_links × 8 i32 = src z,y,x,l + tgt z,y,x,l
 * out_coords:  n_cells × 4 i32 (z,y,x,l) hop order
 * out_indices: n_cells × i32 Index(z,y,x,l)
 * out_remote:  n_cells × 5 i32 (is_remote, tz,ty,tx,tl)
 */
WELVET_API welvet_status welvet_arch_topology(
    int32_t depth, int32_t rows, int32_t cols, int32_t layers_per_cell,
    const int32_t* links, int32_t n_links, int32_t* out_n_cells, int32_t* out_coords,
    int32_t* out_indices, int32_t* out_remote);

/* MHA self-attn baseline (ModeSelf). mask/pos match Go mha.MaskKind/PosKind.
 * Q: [q_dim*d], K/V: [kv_dim*d], O: [d*q_dim], x: [batch*seq*d]
 * pre: [batch*seq*q_dim], post: [batch*seq*d]
 */
WELVET_API welvet_status welvet_mha_forward_f32(
    int32_t batch, int32_t seq, int32_t d_model, int32_t num_heads, int32_t num_kv_heads,
    int32_t head_dim, int32_t max_seq_len, int32_t mask, int32_t pos, int32_t dtype,
    int32_t backend, const float* Q, const float* K, const float* V, const float* O,
    const float* x, float* pre, float* post);

/* SwiGLU baseline. Gate/Up: [inter*in], Down: [in*inter], x: [batch*seq*in]
 * pre (h): [batch*seq*inter], post: [batch*seq*in]
 */
WELVET_API welvet_status welvet_swiglu_forward_f32(
    int32_t batch, int32_t seq, int32_t input_dim, int32_t intermediate_dim, int32_t dtype,
    int32_t backend, const float* Gate, const float* Up, const float* Down, const float* x,
    float* pre, float* post);

WELVET_API const char* welvet_version(void);
WELVET_API int32_t welvet_gap_count(void);
WELVET_API void welvet_print_gaps(void);

#ifdef __cplusplus
}
#endif
