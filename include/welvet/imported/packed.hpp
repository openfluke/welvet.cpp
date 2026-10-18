// Copied from welvet-to-rust_n_cpp/welvet_cpp (code only; not documentation).
#pragma once
#include "welvet/imported/decode.hpp"
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
namespace welvet_imported {
// Decode and multiply in registers; reduce each eight-element tile separately
// to preserve the packed Go kernel's floating-point order.
inline double packedTile8(const float* x, __m256 w) {
    const __m256 xv = _mm256_loadu_ps(x);
    const __m256d lo = _mm256_mul_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(xv)),
                                    _mm256_cvtps_pd(_mm256_castps256_ps128(w)));
    const __m256d hi = _mm256_mul_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(xv, 1)),
                                    _mm256_cvtps_pd(_mm256_extractf128_ps(w, 1)));
    const __m256d p = _mm256_add_pd(lo, hi);
    const __m128d s = _mm_add_pd(_mm256_castpd256_pd128(p), _mm256_extractf128_pd(p, 1));
    return _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)));
}
// Integer bit assembly matches welvet's FP8 decoding (including its
// zero-valued subnormals and canonical E5M2 NaNs), without slow gathers.
inline __m256 packedFP8x8(const std::uint8_t* w, int kind) {
    const __m256i c = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w)));
    const __m256i sign = _mm256_slli_epi32(_mm256_and_si256(c, _mm256_set1_epi32(128)), 24);
    __m256i e, m, bits;
    if (kind == 1) {
        e = _mm256_and_si256(_mm256_srli_epi32(c, 2), _mm256_set1_epi32(31));
        m = _mm256_and_si256(c, _mm256_set1_epi32(3));
        bits = _mm256_or_si256(_mm256_slli_epi32(_mm256_add_epi32(e, _mm256_set1_epi32(112)), 23), _mm256_slli_epi32(m,21));
        const __m256i special = _mm256_blendv_epi8(_mm256_set1_epi32(0x7fc00000), _mm256_set1_epi32(0x7f800000), _mm256_cmpeq_epi32(m, _mm256_setzero_si256()));
        bits = _mm256_blendv_epi8(bits, special, _mm256_cmpeq_epi32(e, _mm256_set1_epi32(31)));
    } else {
        e = _mm256_and_si256(_mm256_srli_epi32(c, 3), _mm256_set1_epi32(15));
        m = _mm256_and_si256(c, _mm256_set1_epi32(7));
        bits = _mm256_or_si256(_mm256_slli_epi32(_mm256_add_epi32(e, _mm256_set1_epi32(120)), 23), _mm256_slli_epi32(m,20));
        const __m256i maxCode = _mm256_cmpeq_epi32(_mm256_and_si256(c, _mm256_set1_epi32(127)), _mm256_set1_epi32(127));
        bits = _mm256_blendv_epi8(bits, _mm256_set1_epi32(0x43e00000), maxCode);
    }
    bits = _mm256_blendv_epi8(bits, _mm256_setzero_si256(), _mm256_cmpeq_epi32(e, _mm256_setzero_si256()));
    return _mm256_castsi256_ps(_mm256_or_si256(bits, sign));
}
inline __m256 packedFP4x8(const std::uint8_t* w) {
    std::uint32_t bits; std::memcpy(&bits, w, 4);
    const __m256i v = _mm256_set1_epi32(static_cast<int>(bits));
    const __m256i shifts = _mm256_setr_epi32(0,4,8,12,16,20,24,28);
    const __m256i codes = _mm256_and_si256(_mm256_srlv_epi32(v, shifts), _mm256_set1_epi32(15));
    const __m256 magnitude = _mm256_permutevar8x32_ps(_mm256_setr_ps(0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f), codes);
    return _mm256_castsi256_ps(_mm256_or_si256(_mm256_castps_si256(magnitude), _mm256_slli_epi32(_mm256_and_si256(codes, _mm256_set1_epi32(8)), 28)));
}
// Eight packed codes per iteration; all input loads stay within the payload.
// Signed formats use two's-complement codes; unsigned formats have a min prefix.
template<int Bits, bool Signed>
inline void expandBitsF32(const NativeStore& s, std::vector<float>& out) {
    const int n = s.rows*s.cols;
    constexpr int prefix = Signed ? 0 : 4;
    if (s.raw.size() < prefix + (static_cast<std::size_t>(n)*Bits+7)/8)
        throw std::runtime_error("packed native short");
    const auto* body = s.raw.data()+prefix;
    const float scale = s.scale == 0.f ? 1.f : s.scale;
    const float minV = Signed ? 0.f : loadF32(s.raw.data());
    const auto shifts = _mm256_setr_epi32(0,Bits,2*Bits,3*Bits,0,Bits,2*Bits,3*Bits);
    out.resize(n);
    int i=0;
    for (; i+8<=n; i+=8) {
        std::uint64_t word=0; std::memcpy(&word,body+i*Bits/8,Bits);
        const auto words = _mm256_setr_epi32(word,word,word,word,word>>(4*Bits),word>>(4*Bits),word>>(4*Bits),word>>(4*Bits));
        auto codes = _mm256_and_si256(_mm256_srlv_epi32(words,shifts),_mm256_set1_epi32((1<<Bits)-1));
        if constexpr (Signed) {
            const auto sign = _mm256_set1_epi32(1<<(Bits-1));
            codes = _mm256_sub_epi32(_mm256_xor_si256(codes,sign),sign);
        }
        auto v = _mm256_mul_ps(_mm256_cvtepi32_ps(codes), _mm256_set1_ps(scale));
        if constexpr (!Signed) v = _mm256_add_ps(v,_mm256_set1_ps(minV));
        _mm256_storeu_ps(out.data()+i,v);
    }
    for (;i<n;++i) out[i]=weightAt(s,i,n);
}
// Reuse decoded low-precision weights across a batch; no cross-forward cache.
#if defined(__F16C__)
template<DType Kind>
inline std::vector<float> expandLowp(const NativeStore& s) {
    const int n=s.rows*s.cols;std::vector<float> out(n);int i=0;
    const int bytes=(Kind==DType::Float16 || Kind==DType::BFloat16)?n*2:Kind==DType::FP4?(n+1)/2:n;
    if(s.raw.size()<static_cast<std::size_t>(bytes)) throw std::runtime_error("lowp native short");
    for(;i+8<=n;i+=8) {
        __m256 v;
        if constexpr(Kind==DType::Float16) v=_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(s.raw.data()+2*i)));
        else if constexpr(Kind==DType::BFloat16) v=_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(s.raw.data()+2*i))),16));
        else if constexpr(Kind==DType::FP4) v=packedFP4x8(s.raw.data()+i/2);
        else v=packedFP8x8(s.raw.data()+i,Kind==DType::FP8E5M2?1:0);
        _mm256_storeu_ps(out.data()+i,v);
    }
    for(;i<n;++i) out[i]=weightAt(s,i,n);
    return out;
}
inline std::vector<float> expandLowp(const NativeStore& s) {
    switch(s.dtype) {
    case DType::Float16:return expandLowp<DType::Float16>(s);
    case DType::BFloat16:return expandLowp<DType::BFloat16>(s);
    case DType::FP8E4M3:return expandLowp<DType::FP8E4M3>(s);
    case DType::FP8E5M2:return expandLowp<DType::FP8E5M2>(s);
    case DType::FP4:return expandLowp<DType::FP4>(s);
    default:throw std::runtime_error("not lowp");
    }
}
inline void groupedRows(float* y,const float* x,const float* w,int rows,int n) {
    int r=0;
    for(;r+2<=rows;r+=2) {
        const float *wa=w+r*n,*wb=wa+n;double a=0,b=0;int i=0;
        for(;i+8<=n;i+=8) {a+=packedTile8(x+i,_mm256_loadu_ps(wa+i));b+=packedTile8(x+i,_mm256_loadu_ps(wb+i));}
        for(;i<n;++i) {a+=double(x[i])*wa[i];b+=double(x[i])*wb[i];}
        y[r]=static_cast<float>(a);y[r+1]=static_cast<float>(b);
    }
    if(r<rows) {double a=0;int i=0;for(;i+8<=n;i+=8) a+=packedTile8(x+i,_mm256_loadu_ps(w+r*n+i));for(;i<n;++i) a+=double(x[i])*w[r*n+i];y[r]=static_cast<float>(a);}
}
#endif
template<int Bits, int Offset>
inline bool expandNarrowSIMD(const NativeStore& s,std::vector<std::int8_t>& out) {
    const int n=s.rows*s.cols;if(s.raw.size()<static_cast<std::size_t>((n*Bits+7)/8)) return false;
    out.resize(n);const auto shifts=_mm256_setr_epi32(0,Bits,2*Bits,3*Bits,4*Bits,5*Bits,6*Bits,7*Bits);int i=0;
    for(;i+8<=n;i+=8) {
        std::uint32_t word=0;std::memcpy(&word,s.raw.data()+i*Bits/8,Bits);
        auto q=_mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(word),shifts),_mm256_set1_epi32((1<<Bits)-1));
        if constexpr(Bits==4) q=_mm256_sub_epi32(_mm256_xor_si256(q,_mm256_set1_epi32(8)),_mm256_set1_epi32(8));
        else if constexpr(Bits==1) q=_mm256_sub_epi32(_mm256_slli_epi32(q,1),_mm256_set1_epi32(1));
        else q=_mm256_sub_epi32(q,_mm256_set1_epi32(Offset));
        const auto q16=_mm_packs_epi32(_mm256_castsi256_si128(q),_mm256_extracti128_si256(q,1));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(out.data()+i),_mm_packs_epi16(q16,q16));
    }
    for(;i<n;++i) {
        int q=(s.raw[i*Bits/8]>>((i*Bits)%8))&((1<<Bits)-1);
        if constexpr(Bits==4) q=(q^8)-8;else if constexpr(Bits==1) q=2*q-1;else q-=Offset;
        out[i]=static_cast<std::int8_t>(q);
    }
    return true;
}
inline void expandNF4(const NativeStore& s, std::vector<float>& out) {
    const int n=s.rows*s.cols;
    if(s.raw.size()<static_cast<std::size_t>((n+1)/2)) throw std::runtime_error("nf4 native short");
    out.resize(n); const float scale=s.scale==0.f?1.f:s.scale;
    const auto lo=_mm256_loadu_ps(kNF4),hi=_mm256_loadu_ps(kNF4+8);
    const auto shifts=_mm256_setr_epi32(0,4,8,12,16,20,24,28);
    int i=0;
    for(;i+8<=n;i+=8) {
        std::uint32_t bits;std::memcpy(&bits,s.raw.data()+i/2,4);
        const auto codes=_mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(bits),shifts),_mm256_set1_epi32(15));
        const auto a=_mm256_permutevar8x32_ps(lo,codes),b=_mm256_permutevar8x32_ps(hi,codes);
        const auto v=_mm256_blendv_ps(a,b,_mm256_castsi256_ps(_mm256_cmpgt_epi32(codes,_mm256_set1_epi32(7))));
        _mm256_storeu_ps(out.data()+i,_mm256_mul_ps(v,_mm256_set1_ps(scale)));
    }
    for(;i<n;++i) out[i]=weightAt(s,i,n);
}
inline bool expandPackedF32(const NativeStore& s, std::vector<float>& out) {
    switch(s.dtype) {
    case DType::NF4: expandNF4(s,out); return true;
    case DType::FP6: case DType::Int6: expandBitsF32<6,true>(s,out); return true;
    case DType::Int5: expandBitsF32<5,true>(s,out); return true;
    case DType::Int3: expandBitsF32<3,true>(s,out); return true;
    case DType::Uint6: expandBitsF32<6,false>(s,out); return true;
    case DType::Uint5: expandBitsF32<5,false>(s,out); return true;
    case DType::Uint4: expandBitsF32<4,false>(s,out); return true;
    case DType::Uint3: expandBitsF32<3,false>(s,out); return true;
    case DType::Uint2: expandBitsF32<2,false>(s,out); return true;
    default: return false;
    }
}
} // namespace welvet_imported
#endif
