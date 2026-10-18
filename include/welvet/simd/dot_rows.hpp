// Copied from welvet-to-rust_n_cpp/welvet_cpp (code only; not documentation).
#pragma once
#include "welvet/simd/dot_tile.hpp"
namespace welvet { namespace dot_tile {
inline void rowsF32(float* y, const float* x, const float* w, int rows, int n, double offset=0.0) {
    int r=0;
#if defined(__AVX2__) && defined(__FMA__)
    for(;r+2<=rows;r+=2) {
        auto a0=_mm256_setzero_pd(),a1=a0,b0=a0,b1=a0;
        const auto* wa=w+r*n;const auto* wb=wa+n;int i=0;
        for(;i+8<=n;i+=8) {
            const auto x0=_mm256_cvtps_pd(_mm_loadu_ps(x+i)),x1=_mm256_cvtps_pd(_mm_loadu_ps(x+i+4));
            a0=_mm256_fmadd_pd(x0,_mm256_cvtps_pd(_mm_loadu_ps(wa+i)),a0);
            a1=_mm256_fmadd_pd(x1,_mm256_cvtps_pd(_mm_loadu_ps(wa+i+4)),a1);
            b0=_mm256_fmadd_pd(x0,_mm256_cvtps_pd(_mm_loadu_ps(wb+i)),b0);
            b1=_mm256_fmadd_pd(x1,_mm256_cvtps_pd(_mm_loadu_ps(wb+i+4)),b1);
        }
        a0=_mm256_add_pd(a0,a1);b0=_mm256_add_pd(b0,b1);
        const auto a=_mm_add_pd(_mm256_castpd256_pd128(a0),_mm256_extractf128_pd(a0,1));
        const auto b=_mm_add_pd(_mm256_castpd256_pd128(b0),_mm256_extractf128_pd(b0,1));
        double sa=0.0+_mm_cvtsd_f64(_mm_add_sd(a,_mm_unpackhi_pd(a,a)));
        double sb=0.0+_mm_cvtsd_f64(_mm_add_sd(b,_mm_unpackhi_pd(b,b)));
        for(;i<n;++i) {sa+=double(x[i])*double(wa[i]);sb+=double(x[i])*double(wb[i]);}
        y[r]=static_cast<float>(sa+offset);y[r+1]=static_cast<float>(sb+offset);
    }
#endif
    for(;r<rows;++r) y[r]=static_cast<float>(f32AccF64Simd(x,w+r*n,n,0.0)+offset);
}
inline void rowsF64(float* y, const double* x, const double* w, int rows, int n) {
    int r=0;
#if defined(__AVX2__) && defined(__FMA__)
    for(;r+2<=rows;r+=2) {
        auto a0=_mm256_setzero_pd(),a1=a0,b0=a0,b1=a0;
        const auto* wa=w+r*n;const auto* wb=wa+n;int i=0;
        for(;i+8<=n;i+=8) {
            const auto x0=_mm256_loadu_pd(x+i),x1=_mm256_loadu_pd(x+i+4);
            a0=_mm256_fmadd_pd(x0,_mm256_loadu_pd(wa+i),a0);
            a1=_mm256_fmadd_pd(x1,_mm256_loadu_pd(wa+i+4),a1);
            b0=_mm256_fmadd_pd(x0,_mm256_loadu_pd(wb+i),b0);
            b1=_mm256_fmadd_pd(x1,_mm256_loadu_pd(wb+i+4),b1);
        }
        a0=_mm256_add_pd(a0,a1);b0=_mm256_add_pd(b0,b1);
        if(i+4<=n) {const auto xv=_mm256_loadu_pd(x+i);a0=_mm256_fmadd_pd(xv,_mm256_loadu_pd(wa+i),a0);b0=_mm256_fmadd_pd(xv,_mm256_loadu_pd(wb+i),b0);i+=4;}
        const auto a=_mm_add_pd(_mm256_castpd256_pd128(a0),_mm256_extractf128_pd(a0,1));
        const auto b=_mm_add_pd(_mm256_castpd256_pd128(b0),_mm256_extractf128_pd(b0,1));
        double sa=0.0+_mm_cvtsd_f64(_mm_add_sd(a,_mm_unpackhi_pd(a,a)));
        double sb=0.0+_mm_cvtsd_f64(_mm_add_sd(b,_mm_unpackhi_pd(b,b)));
        for(;i<n;++i) {sa+=double(x[i])*double(wa[i]);sb+=double(x[i])*double(wb[i]);}
        y[r]=static_cast<float>(sa);y[r+1]=static_cast<float>(sb);
    }
#endif
    for(;r<rows;++r) y[r]=static_cast<float>(f64AccF64Simd(x,w+r*n,n,0.0));
}
}} // namespace welvet::dot_tile
