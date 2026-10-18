// Copied from welvet-to-rust_n_cpp/welvet_cpp (code only; not documentation).
#pragma once

#include <cstddef>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace welvet {
namespace dot_tile {

inline bool simdAvailable() {
#if defined(__AVX2__) && defined(__FMA__)
    return true;
#elif defined(__aarch64__) || defined(__ARM_NEON)
    return true;
#else
    return false;
#endif
}

// Portable scalar — f32×f32 products accumulated in f64.
inline double f32AccF64Scalar(const float* x, const float* w, int n, double prev) {
    double sum = prev;
    for (int i = 0; i < n; i++) {
        sum += static_cast<double>(x[i]) * static_cast<double>(w[i]);
    }
    return sum;
}

#if defined(__AVX2__) && defined(__FMA__)

// Matches welvet/simd/avx2_amd64.s ·dotF32AccF64Avx2:
// two 4-lane f64 accumulators retain the same eight lanes and reduction tree.
inline double f32AccF64Simd(const float* x, const float* w, int n, double prev) {
    __m256d a0 = _mm256_setzero_pd();
    __m256d a1 = _mm256_setzero_pd();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        a0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(x+i)),
                             _mm256_cvtps_pd(_mm_loadu_ps(w+i)), a0);
        a1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(x+i+4)),
                             _mm256_cvtps_pd(_mm_loadu_ps(w+i+4)), a1);
    }
    // Same eight independent lanes and final reduction tree as Go.
    const __m256d p = _mm256_add_pd(a0, a1);
    const __m128d s = _mm_add_pd(_mm256_castpd256_pd128(p), _mm256_extractf128_pd(p, 1));
    double sum = prev + _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)));

    for (; i < n; i++) {
        sum += static_cast<double>(x[i]) * static_cast<double>(w[i]);
    }
    return sum;
}

#elif defined(__aarch64__) || defined(__ARM_NEON)

// Match welvet arm64 NEON: 8-wide widen+FMA with four 2-lane f64 accs.
inline double f32AccF64Simd(const float* x, const float* w, int n, double prev) {
    float64x2_t a0 = vdupq_n_f64(0.0);
    float64x2_t a1 = vdupq_n_f64(0.0);
    float64x2_t a2 = vdupq_n_f64(0.0);
    float64x2_t a3 = vdupq_n_f64(0.0);

    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const float32x2_t x01 = vld1_f32(x + i);
        const float32x2_t x23 = vld1_f32(x + i + 2);
        const float32x2_t x45 = vld1_f32(x + i + 4);
        const float32x2_t x67 = vld1_f32(x + i + 6);
        const float32x2_t w01 = vld1_f32(w + i);
        const float32x2_t w23 = vld1_f32(w + i + 2);
        const float32x2_t w45 = vld1_f32(w + i + 4);
        const float32x2_t w67 = vld1_f32(w + i + 6);

        a0 = vfmaq_f64(a0, vcvt_f64_f32(x01), vcvt_f64_f32(w01));
        a1 = vfmaq_f64(a1, vcvt_f64_f32(x23), vcvt_f64_f32(w23));
        a2 = vfmaq_f64(a2, vcvt_f64_f32(x45), vcvt_f64_f32(w45));
        a3 = vfmaq_f64(a3, vcvt_f64_f32(x67), vcvt_f64_f32(w67));
    }

    a0 = vaddq_f64(a0, a2);
    a1 = vaddq_f64(a1, a3);
    a0 = vaddq_f64(a0, a1);
    double sum = prev + vgetq_lane_f64(a0, 0) + vgetq_lane_f64(a0, 1);

    for (; i < n; i++) {
        sum += static_cast<double>(x[i]) * static_cast<double>(w[i]);
    }
    return sum;
}

#else

inline double f32AccF64Simd(const float* x, const float* w, int n, double prev) {
    return f32AccF64Scalar(x, w, n, prev);
}

#endif

// Scalar f64×f64 → f64 (arm64 / no AVX2).
inline double f64AccF64Scalar(const double* x, const double* w, int n, double prev) {
    double sum = prev;
    for (int i = 0; i < n; i++) {
        sum += x[i] * w[i];
    }
    return sum;
}

#if defined(__AVX2__) && defined(__FMA__)

// Matches welvet/simd/dot_f64_amd64.s ·dotF64AccF64Avx2:
// dual 4-wide f64 accumulators, 8 elems/iter, then 4-wide, then scalar tail.
inline double f64AccF64Simd(const double* x, const double* w, int n, double prev) {
    __m256d a0 = _mm256_setzero_pd();
    __m256d a1 = _mm256_setzero_pd();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256d x0 = _mm256_loadu_pd(x + i);
        const __m256d x1 = _mm256_loadu_pd(x + i + 4);
        const __m256d w0 = _mm256_loadu_pd(w + i);
        const __m256d w1 = _mm256_loadu_pd(w + i + 4);
        a0 = _mm256_fmadd_pd(x0, w0, a0);
        a1 = _mm256_fmadd_pd(x1, w1, a1);
    }
    a0 = _mm256_add_pd(a0, a1);
    for (; i + 4 <= n; i += 4) {
        const __m256d xv = _mm256_loadu_pd(x + i);
        const __m256d wv = _mm256_loadu_pd(w + i);
        a0 = _mm256_fmadd_pd(xv, wv, a0);
    }
    const __m128d lo = _mm256_castpd256_pd128(a0);
    const __m128d hi = _mm256_extractf128_pd(a0, 1);
    __m128d s = _mm_add_pd(lo, hi);
    const __m128d shuf = _mm_permute_pd(s, 1);
    s = _mm_add_sd(s, shuf);
    double sum = prev + _mm_cvtsd_f64(s);
    for (; i < n; i++) {
        sum += x[i] * w[i];
    }
    return sum;
}

#else

inline double f64AccF64Simd(const double* x, const double* w, int n, double prev) {
    return f64AccF64Scalar(x, w, n, prev);
}

#endif

} // namespace dot_tile
} // namespace welvet
