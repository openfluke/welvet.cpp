// Copied from welvet-to-rust_n_cpp/welvet_cpp (code only; not documentation).
#pragma once

#include "welvet/imported/decode.hpp"
#include "welvet/imported/packed.hpp"
#include "welvet/simd/dot_i8.hpp"
#include "welvet/simd/dot_tile.hpp"
#include "welvet/simd/dot_rows.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace welvet_imported {

inline float quantizeActsI8(const float* x, int n, std::int8_t* out) {
    float maxAbs = 0.f;
    for (int i = 0; i < n; i++) {
        maxAbs = std::max(maxAbs, std::fabs(x[i]));
    }
    float scale = maxAbs / 127.f;
    if (scale == 0.f) {
        scale = 1.f;
    }
    for (int i = 0; i < n; i++) {
        int q = static_cast<int>(std::round(static_cast<double>(x[i] / scale)));
        if (q > 127) {
            q = 127;
        }
        if (q < -128) {
            q = -128;
        }
        out[i] = static_cast<std::int8_t>(q);
    }
    return scale;
}

inline bool expandNarrowToI8(const NativeStore& s, std::vector<std::int8_t>& out) {
#if defined(__AVX2__) && defined(__FMA__)
    switch(s.dtype) {
    case DType::Int4:return expandNarrowSIMD<4,0>(s,out);
    case DType::Int2:return expandNarrowSIMD<2,2>(s,out);
    case DType::Ternary:return expandNarrowSIMD<2,1>(s,out);
    case DType::Binary:return expandNarrowSIMD<1,0>(s,out);
    default:return false;
    }
#endif
    const int n = s.rows * s.cols;
    out.resize(static_cast<std::size_t>(n));
    const auto& raw = s.raw;
    switch (s.dtype) {
    case DType::Int4: {
        if (static_cast<int>(raw.size()) < (n + 1) / 2) {
            return false;
        }
        for (int i = 0; i < n; i++) {
            int q = (i % 2 == 0) ? (raw[i / 2] & 0xF) : (raw[i / 2] >> 4);
            if (q > 7) {
                q -= 16;
            }
            out[i] = static_cast<std::int8_t>(q);
        }
        return true;
    }
    case DType::Int2: {
        if (static_cast<int>(raw.size()) < (n + 3) / 4) {
            return false;
        }
        for (int i = 0; i < n; i++) {
            int code = (raw[i / 4] >> ((i % 4) * 2)) & 3;
            out[i] = static_cast<std::int8_t>(code - 2);
        }
        return true;
    }
    case DType::Ternary: {
        if (static_cast<int>(raw.size()) < (n + 3) / 4) {
            return false;
        }
        for (int i = 0; i < n; i++) {
            int code = (raw[i / 4] >> ((i % 4) * 2)) & 3;
            out[i] = static_cast<std::int8_t>(code - 1);
        }
        return true;
    }
    case DType::Binary: {
        if (static_cast<int>(raw.size()) < (n + 7) / 8) {
            return false;
        }
        for (int i = 0; i < n; i++) {
            out[i] = (raw[i / 8] & (1u << (i % 8))) ? 1 : -1;
        }
        return true;
    }
    default:
        return false;
    }
}

inline double dotF16Packed(const float* x, const std::uint8_t* w, int i0, int n) {
    double sum = 0.0;
    float buf[8];
    int i = 0;
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)

    for (; i + 8 <= n; i += 8) {
        sum += packedTile8(x+i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + (i0+i)*2))));
    }
#endif
    for (; i < n;) {
        int m = std::min(8, n - i);
        for (int k = 0; k < m; k++) {
            const std::uint16_t h = loadU16(w + (i0 + i + k) * 2);
            buf[k] = float16ToF32(h);
        }
        sum = ::welvet::dot_tile::f32AccF64Simd(x + i, buf, m, sum);
        i += m;
    }
    return sum;
}

inline double dotBF16Packed(const float* x, const std::uint8_t* w, int i0, int n) {
    double sum = 0.0;
    float buf[8];
    int i = 0;
#if defined(__AVX2__) && defined(__FMA__)

    for (; i + 8 <= n; i += 8) {
        sum += packedTile8(x+i, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + (i0+i)*2))), 16)));
    }
#endif
    for (; i < n;) {
        int m = std::min(8, n - i);
        for (int k = 0; k < m; k++) {
            buf[k] = bfloat16ToF32(loadU16(w + (i0 + i + k) * 2));
        }
        sum = ::welvet::dot_tile::f32AccF64Simd(x + i, buf, m, sum);
        i += m;
    }
    return sum;
}

inline double dotFP8Packed(const float* x, const std::uint8_t* w, int i0, int n, int kind) {
    double sum = 0.0;
    float buf[8];
    int i = 0;
#if defined(__AVX2__) && defined(__FMA__)

    for (; i + 8 <= n; i += 8) {
        sum += packedTile8(x+i, packedFP8x8(w + i0+i, kind));
    }
#endif
    for (; i < n;) {
        int m = std::min(8, n - i);
        for (int k = 0; k < m; k++) {
            const std::uint8_t b = w[i0 + i + k];
            buf[k] = (kind == 1) ? fp8e5m2ToF32(b) : fp8e4m3ToF32(b);
        }
        sum = ::welvet::dot_tile::f32AccF64Simd(x + i, buf, m, sum);
        i += m;
    }
    return sum;
}

inline double dotFP4Packed(const float* x, const std::uint8_t* w, int i0, int n) {
    double sum = 0.0;
    float buf[8];
    int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    if (i0 % 2 == 0)
    for (; i + 8 <= n; i += 8) {
        sum += packedTile8(x+i, packedFP4x8(w + (i0+i)/2));
    }
#endif
    for (; i < n;) {
        int m = std::min(8, n - i);
        for (int k = 0; k < m; k++) {
            const int flat = i0 + i + k;
            const std::uint8_t b = w[flat / 2];
            const std::uint8_t code = (flat % 2 == 0) ? (b & 0xf) : (b >> 4);
            buf[k] = kFP4[code & 0xf];
        }
        sum = ::welvet::dot_tile::f32AccF64Simd(x + i, buf, m, sum);
        i += m;
    }
    return sum;
}

inline void decodeRowF64(const NativeStore& s, int row, double* dst) {
    const int off = row * s.cols;
    const int n = s.rows * s.cols;
    if (s.dtype == DType::Float64) {
        for (int c = 0; c < s.cols; c++) {
            dst[c] = loadF64(s.raw.data() + (off + c) * 8);
        }
        return;
    }
    for (int c = 0; c < s.cols; c++) {
        dst[c] = static_cast<double>(weightAt(s, off + c, n));
    }
}

// Expand-once wire caches — match welvet v1.0.3 ForwardSIMD (no per-row DecodeRow in hot loop).
inline std::vector<float> expandStoreF32(const NativeStore& s) {
    const int n=s.rows*s.cols;
    std::vector<float> out(n);
#if defined(__AVX2__) && defined(__FMA__)
    if (expandPackedF32(s,out)) return out;
#endif
    const float scale=s.scale==0.f?1.f:s.scale;
    int width=0;
    switch(s.dtype) {
    case DType::Uint16: width=2; break;
    case DType::Uint32: width=4; break;
    case DType::Uint64: case DType::Uint: case DType::Uintptr: width=8; break;
    default: break;
    }
    if(width) {
        if(s.raw.size()<4+static_cast<std::size_t>(n)*width) throw std::runtime_error("unsigned native short");
        const float min=loadF32(s.raw.data()); const auto* raw=s.raw.data()+4;
        switch(width) {
        case 2: for(int i=0;i<n;++i) out[i]=static_cast<float>(loadU16(raw+2*i))*scale+min; break;
        case 4: for(int i=0;i<n;++i) out[i]=static_cast<float>(loadU32(raw+4*i))*scale+min; break;
        default: for(int i=0;i<n;++i) out[i]=static_cast<float>(loadU64(raw+8*i))*scale+min; break;
        }
    } else {
        for(int r=0;r<s.rows;++r) decodeRow(s,r,out.data()+r*s.cols);
    }
    return out;
}

inline std::vector<double> expandStoreF64(const NativeStore& s) {
    const int n = s.rows * s.cols;
    std::vector<double> out(static_cast<std::size_t>(n));
    for (int r = 0; r < s.rows; r++) {
        decodeRowF64(s, r, out.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(s.cols));
    }
    return out;
}

// Call after loading or changing native weights, before concurrent forwards.
inline void prepareWireF64(NativeStore& s) {
    s.wireF64.clear();
    switch (s.dtype) {
    case DType::Float64: case DType::Int64: case DType::Int32:
    case DType::Int16: case DType::Int: case DType::Complex64: case DType::Complex128:
        s.wireF64 = expandStoreF64(s); break;
    default: break;
    }
}

// --- per-dtype GEMV (matches welvet v1.0.3 ForwardSIMD) ---

inline void gemvMasterF32(float* y, const float* x, const NativeStore& s, int batch) {
    const int in = s.cols, out = s.rows;
    const float* w = reinterpret_cast<const float*>(s.raw.data());
    for (int b = 0; b < batch; b++) {
        const float* xRow = x + b * in;
        float* yRow = y + b * out;
        ::welvet::dot_tile::rowsF32(yRow,xRow,w,out,in);
    }
}

inline void gemvInt8(float* y, const float* x, const NativeStore& s, int batch) {
    const int in = s.cols, out = s.rows, n = out * in;
    const auto* wI8 = reinterpret_cast<const std::int8_t*>(s.raw.data());
    float scale = s.scale == 0.f ? 1.f : s.scale;
    std::vector<std::int8_t> xq(static_cast<std::size_t>(in));
    for (int b = 0; b < batch; b++) {
        const float* xRow = x + b * in;
        const float actScale = quantizeActsI8(xRow, in, xq.data());
        float* yRow = y + b * out;
        for (int o = 0; o < out; o++) {
            const std::int32_t acc = ::welvet::dot_i8::dotI8Tile(xq.data(), wI8 + o * in, in, 0);
            yRow[o] = static_cast<float>(acc) * actScale * scale;
        }
    }
    (void)n;
}

inline void gemvNarrowI8(float* y, const float* x, const NativeStore& s, int batch) {
    std::vector<std::int8_t> wI8;
    if (!expandNarrowToI8(s, wI8)) {
        throw std::runtime_error("narrow expand failed");
    }
    float scale = s.scale == 0.f ? 1.f : s.scale;
    const int in = s.cols, out = s.rows;
    std::vector<std::int8_t> xq(static_cast<std::size_t>(in));
    for (int b = 0; b < batch; b++) {
        const float* xRow = x + b * in;
        const float actScale = quantizeActsI8(xRow, in, xq.data());
        float* yRow = y + b * out;
        for (int o = 0; o < out; o++) {
            const std::int32_t acc = ::welvet::dot_i8::dotI8Tile(xq.data(), wI8.data() + o * in, in, 0);
            yRow[o] = static_cast<float>(acc) * actScale * scale;
        }
    }
}

inline void gemvUint8Affine(float* y, const float* x, const NativeStore& s, int batch) {
    const int in = s.cols, out = s.rows, n = out * in;
    if (static_cast<int>(s.raw.size()) < 4 + n) {
        throw std::runtime_error("uint8 affine short");
    }
    const float minV = loadF32(s.raw.data());
    float scale = s.scale == 0.f ? 1.f : s.scale;
    const std::uint8_t* body = s.raw.data() + 4;
    std::vector<float> scratch(static_cast<std::size_t>(n));
    for(int i=0;i<n;++i) scratch[i]=static_cast<float>(body[i])*scale;
    for (int b = 0; b < batch; b++) {
        const float* xRow = x + b * in;
        double sumX = 0.0;
        for (int i = 0; i < in; i++) {
            sumX += static_cast<double>(xRow[i]);
        }
        float* yRow = y + b * out;
        ::welvet::dot_tile::rowsF32(yRow,xRow,scratch.data(),out,in,static_cast<double>(minV)*sumX);
    }
}

inline void gemvLowpPacked(float* y, const float* x, const NativeStore& s, int batch) {
    const int in = s.cols, out = s.rows;
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    if(batch>=4) {
        const auto w=expandLowp(s);
        for(int b=0;b<batch;++b) groupedRows(y+b*out,x+b*in,w.data(),out,in);
        return;
    }
#endif
    for (int b = 0; b < batch; b++) {
        const float* xRow = x + b * in;
        float* yRow = y + b * out;
        for (int o = 0; o < out; o++) {
            const int i0 = o * in;
            double sum = 0.0;
            switch (s.dtype) {
            case DType::Float16:
                sum = dotF16Packed(xRow, s.raw.data(), i0, in);
                break;
            case DType::BFloat16:
                sum = dotBF16Packed(xRow, s.raw.data(), i0, in);
                break;
            case DType::FP8E4M3:
                sum = dotFP8Packed(xRow, s.raw.data(), i0, in, 0);
                break;
            case DType::FP8E5M2:
                sum = dotFP8Packed(xRow, s.raw.data(), i0, in, 1);
                break;
            case DType::FP4:
                sum = dotFP4Packed(xRow, s.raw.data(), i0, in);
                break;
            default:
                throw std::runtime_error("lowp unexpected");
            }
            yRow[o] = static_cast<float>(sum);
        }
    }
}

inline void gemvUintAffine(float* y, const float* x, const NativeStore& s, int batch) {
    // expand native→f32 once, then DotTile (v1.0.3)
    const auto wAll = expandStoreF32(s);
    const int in = s.cols, out = s.rows;
    for (int b = 0; b < batch; b++) {
        const float* xRow = x + b * in;
        float* yRow = y + b * out;
        ::welvet::dot_tile::rowsF32(yRow,xRow,wAll.data(),out,in);
    }
}

// NF4 / FP6 / Int3/5/6 — expand-once → DotTile f32 (not WireF64 stream).
inline void gemvExpandF32(float* y, const float* x, const NativeStore& s, int batch) {
    const auto wAll = expandStoreF32(s);
    const int in = s.cols, out = s.rows;
    for (int b = 0; b < batch; b++) {
        const float* xRow = x + b * in;
        float* yRow = y + b * out;
        ::welvet::dot_tile::rowsF32(yRow,xRow,wAll.data(),out,in);
    }
}

// WireF64 cache + DotTileF64 (AVX2 Plan 9 on amd64) — int*/complex*/float64.
inline void gemvWireF64(float* y, const float* x, const NativeStore& s, int batch) {
    const auto& wAll = s.wireF64;
    if (wAll.size() != static_cast<std::size_t>(s.rows) * s.cols)
        throw std::runtime_error("prepareWireF64 required after weight changes");
    const int in = s.cols, out = s.rows;
    std::vector<double> xRow(static_cast<std::size_t>(in));
    for (int b = 0; b < batch; b++) {
        const float* xin = x + b * in;
        for (int i = 0; i < in; i++) {
            xRow[i] = static_cast<double>(xin[i]);
        }
        float* yRow = y + b * out;
        ::welvet::dot_tile::rowsF64(yRow,xRow.data(),wAll.data(),out,in);
    }
}

inline void gemvSIMD(float* y, const float* x, const NativeStore& s, int batch) {
    switch (s.dtype) {
    case DType::Float32:
        gemvMasterF32(y, x, s, batch);
        return;
    case DType::Int8:
        gemvInt8(y, x, s, batch);
        return;
    case DType::Int4:
    case DType::Int2:
    case DType::Ternary:
    case DType::Binary:
        gemvNarrowI8(y, x, s, batch);
        return;
    case DType::Uint8:
        gemvUint8Affine(y, x, s, batch);
        return;
    case DType::Float16:
    case DType::BFloat16:
    case DType::FP8E4M3:
    case DType::FP8E5M2:
    case DType::FP4:
        gemvLowpPacked(y, x, s, batch);
        return;
    case DType::Uint4:
    case DType::Uint2:
    case DType::Uint16:
    case DType::Uint32:
    case DType::Uint64:
    case DType::Uint:
    case DType::Uintptr:
    case DType::Uint3:
    case DType::Uint5:
    case DType::Uint6:
        gemvUintAffine(y, x, s, batch);
        return;
    case DType::NF4:
    case DType::FP6:
    case DType::Int6:
    case DType::Int5:
    case DType::Int3:
        gemvExpandF32(y, x, s, batch);
        return;
    case DType::Float64:
    case DType::Int64:
    case DType::Int32:
    case DType::Int16:
    case DType::Int:
    case DType::Complex64:
    case DType::Complex128:
        gemvWireF64(y, x, s, batch);
        return;
    }
    throw std::runtime_error("unsupported dtype in gemvSIMD");
}

struct DenseNet {
    int batch = 0;
    int outDim = 0;
    std::vector<NativeStore> layers;
};

inline void forwardSIMD(const DenseNet& net, const float* input, float* output,
                        std::vector<float>& bufA, std::vector<float>& bufB) {
    const int batch = net.batch;
    const int in0 = net.layers.front().cols;
    bufA.resize(static_cast<std::size_t>(batch) * static_cast<std::size_t>(in0));
    std::copy(input, input + bufA.size(), bufA.begin());
    const float* curIn = bufA.data();
    float* curOut = nullptr;
    for (std::size_t li = 0; li < net.layers.size(); li++) {
        const auto& layer = net.layers[li];
        const int out = layer.rows;
        if (li % 2 == 0) {
            bufB.resize(static_cast<std::size_t>(batch) * static_cast<std::size_t>(out));
            curOut = bufB.data();
        } else {
            bufA.resize(static_cast<std::size_t>(batch) * static_cast<std::size_t>(out));
            curOut = bufA.data();
        }
        gemvSIMD(curOut, curIn, layer, batch);
        curIn = curOut;
    }
    const std::size_t n = static_cast<std::size_t>(batch) * static_cast<std::size_t>(net.layers.back().rows);
    std::copy(curIn, curIn + n, output);
}

inline std::uint64_t fnv1a64(const float* data, std::size_t n) {
    std::uint64_t h = 14695981039346656037ull;
    for (std::size_t i = 0; i < n; i++) {
        std::uint32_t bits;
        std::memcpy(&bits, &data[i], 4);
        for (int b = 0; b < 4; b++) {
            h ^= static_cast<std::uint64_t>((bits >> (8 * b)) & 0xffu);
            h *= 1099511628211ull;
        }
    }
    return h;
}

inline long long flops(const DenseNet& net) {
    long long n = 0;
    for (const auto& l : net.layers) {
        n += 2LL * net.batch * l.rows * l.cols;
    }
    return n;
}

} // namespace welvet_imported
