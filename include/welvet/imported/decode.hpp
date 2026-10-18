// Copied from welvet-to-rust_n_cpp/welvet_cpp (code only; not documentation).
#pragma once

// Exact FormatNone DecodeRow / weightAt port of welvet/weights/decode.go (+ native_ext).
// Used by ForwardSIMD stream path: decode weight row → DotTile.

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace welvet_imported {

enum class DType : std::uint32_t {
    Float64 = 0, Float32 = 1, Float16 = 2, BFloat16 = 3,
    FP8E4M3 = 4, FP8E5M2 = 5,
    Int64 = 6, Int32 = 7, Int16 = 8, Int8 = 9,
    Uint64 = 10, Uint32 = 11, Uint16 = 12, Uint8 = 13,
    Int4 = 14, Uint4 = 15, FP4 = 16, Int2 = 17, Uint2 = 18,
    Ternary = 19, Binary = 20,
    Int = 21, Uint = 22, Uintptr = 23, Complex64 = 24, Complex128 = 25,
    NF4 = 26, FP6 = 27, Int6 = 28, Uint6 = 29, Int5 = 30, Uint5 = 31,
    Int3 = 32, Uint3 = 33,
};

struct NativeStore {
    DType dtype = DType::Float32;
    int rows = 0;
    int cols = 0;
    float scale = 1.f;
    std::vector<double> wireF64; // derived cache; rebuild after native weight changes
    std::vector<std::uint8_t> raw; // master f32 LE bytes OR Native payload
};

inline float loadF32(const std::uint8_t* p) {
    std::uint32_t u;
    std::memcpy(&u, p, 4);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
inline double loadF64(const std::uint8_t* p) {
    std::uint64_t u;
    std::memcpy(&u, p, 8);
    double d;
    std::memcpy(&d, &u, 8);
    return d;
}
inline std::uint16_t loadU16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}
inline std::uint32_t loadU32(const std::uint8_t* p) {
    std::uint32_t u;
    std::memcpy(&u, p, 4);
    return u;
}
inline std::uint64_t loadU64(const std::uint8_t* p) {
    std::uint64_t u;
    std::memcpy(&u, p, 8);
    return u;
}

inline float float16ToF32(std::uint16_t h) {
    std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000) << 16;
    std::uint32_t expo = (h >> 10) & 0x1f;
    std::uint32_t mant = h & 0x3ff;
    std::uint32_t bits;
    if (expo == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            float v = (static_cast<float>(mant) / 1024.f) / 16384.f;
            std::uint32_t vb;
            std::memcpy(&vb, &v, 4);
            bits = sign | vb;
        }
    } else if (expo == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((expo - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

inline float bfloat16ToF32(std::uint16_t h) {
    std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

inline float fp8e4m3ToF32(std::uint8_t b) {
    std::uint32_t sign = (b & 0x80) ? 0x80000000u : 0;
    int e = (b >> 3) & 0xf;
    int m = b & 0x7;
    if (e == 0) {
        if (m == 0) {
            float z;
            std::memcpy(&z, &sign, 4);
            return z;
        }
        float z;
        std::memcpy(&z, &sign, 4);
        return z * static_cast<float>(m) / 8.f / 64.f;
    }
    if (e == 15 && m == 7) {
        float maxv = 448.f;
        std::uint32_t mb;
        std::memcpy(&mb, &maxv, 4);
        float out;
        std::uint32_t bits = sign | mb;
        std::memcpy(&out, &bits, 4);
        return out;
    }
    std::uint32_t bits = sign | static_cast<std::uint32_t>(e - 7 + 127) << 23 | static_cast<std::uint32_t>(m) << 20;
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

inline float fp8e5m2ToF32(std::uint8_t b) {
    std::uint32_t sign = (b & 0x80) ? 0x80000000u : 0;
    int e = (b >> 2) & 0x1f;
    int m = b & 0x3;
    if (e == 0) {
        if (m == 0) {
            float z;
            std::memcpy(&z, &sign, 4);
            return z;
        }
        float z;
        std::memcpy(&z, &sign, 4);
        return z * static_cast<float>(m) / 4.f / 16384.f;
    }
    if (e == 31) {
        std::uint32_t bits = sign | (m == 0 ? 0x7f800000u : 0x7fc00000u);
        float out;
        std::memcpy(&out, &bits, 4);
        return out;
    }
    std::uint32_t bits = sign | static_cast<std::uint32_t>(e - 15 + 127) << 23 | static_cast<std::uint32_t>(m) << 21;
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

static const float kFP4[16] = {
    0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -0.5f, -1, -1.5f, -2, -3, -4, -6,
};
static const float kNF4[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f,
};

inline std::uint64_t getBitCode(const std::uint8_t* raw, int i, int bits) {
    int bitOff = i * bits;
    std::uint64_t c = 0;
    for (int k = 0; k < bits; k++) {
        if (raw[(bitOff + k) / 8] & (1u << ((bitOff + k) % 8))) {
            c |= 1ull << k;
        }
    }
    return c;
}

inline std::int64_t signExtend(std::uint64_t code, int bits) {
    std::uint64_t signBit = 1ull << (bits - 1);
    std::uint64_t mask = (1ull << bits) - 1;
    code &= mask;
    if (code & signBit) {
        return static_cast<std::int64_t>(code) - static_cast<std::int64_t>(1ull << bits);
    }
    return static_cast<std::int64_t>(code);
}

inline float weightAt(const NativeStore& s, int i, int n) {
    float scale = s.scale == 0.f ? 1.f : s.scale;
    const auto& raw = s.raw;
    auto need = [&](std::size_t nbytes) {
        if (raw.size() < nbytes) {
            throw std::runtime_error("native short for dtype");
        }
    };

    switch (s.dtype) {
    case DType::Float32: {
        need(static_cast<std::size_t>(n) * 4);
        return loadF32(raw.data() + i * 4);
    }
    case DType::Float64: {
        need(static_cast<std::size_t>(n) * 8);
        return static_cast<float>(loadF64(raw.data() + i * 8));
    }
    case DType::Float16: {
        need(static_cast<std::size_t>(n) * 2);
        return float16ToF32(loadU16(raw.data() + i * 2));
    }
    case DType::BFloat16: {
        need(static_cast<std::size_t>(n) * 2);
        return bfloat16ToF32(loadU16(raw.data() + i * 2));
    }
    case DType::FP8E4M3: {
        need(static_cast<std::size_t>(n));
        return fp8e4m3ToF32(raw[i]);
    }
    case DType::FP8E5M2: {
        need(static_cast<std::size_t>(n));
        return fp8e5m2ToF32(raw[i]);
    }
    case DType::FP4: {
        need(static_cast<std::size_t>((n + 1) / 2));
        std::uint8_t code = (i % 2 == 0) ? (raw[i / 2] & 0xF) : (raw[i / 2] >> 4);
        return kFP4[code & 0xF];
    }
    case DType::Int64: {
        need(static_cast<std::size_t>(n) * 8);
        return static_cast<float>(static_cast<std::int64_t>(loadU64(raw.data() + i * 8))) * scale;
    }
    case DType::Int32: {
        need(static_cast<std::size_t>(n) * 4);
        return static_cast<float>(static_cast<std::int32_t>(loadU32(raw.data() + i * 4))) * scale;
    }
    case DType::Int16: {
        need(static_cast<std::size_t>(n) * 2);
        return static_cast<float>(static_cast<std::int16_t>(loadU16(raw.data() + i * 2))) * scale;
    }
    case DType::Int8: {
        need(static_cast<std::size_t>(n));
        return static_cast<float>(static_cast<std::int8_t>(raw[i])) * scale;
    }
    case DType::Uint8: {
        need(4 + static_cast<std::size_t>(n));
        float minV = loadF32(raw.data());
        return static_cast<float>(raw[4 + i]) * scale + minV;
    }
    case DType::Uint16: {
        need(4 + static_cast<std::size_t>(n) * 2);
        float minV = loadF32(raw.data());
        return static_cast<float>(loadU16(raw.data() + 4 + i * 2)) * scale + minV;
    }
    case DType::Uint32: {
        need(4 + static_cast<std::size_t>(n) * 4);
        float minV = loadF32(raw.data());
        return static_cast<float>(loadU32(raw.data() + 4 + i * 4)) * scale + minV;
    }
    case DType::Uint64: {
        need(4 + static_cast<std::size_t>(n) * 8);
        float minV = loadF32(raw.data());
        return static_cast<float>(loadU64(raw.data() + 4 + i * 8)) * scale + minV;
    }
    case DType::Int4: {
        need(static_cast<std::size_t>((n + 1) / 2));
        int q = (i % 2 == 0) ? (raw[i / 2] & 0xF) : (raw[i / 2] >> 4);
        if (q > 7) {
            q -= 16;
        }
        return static_cast<float>(q) * scale;
    }
    case DType::Uint4: {
        need(4 + static_cast<std::size_t>((n + 1) / 2));
        float minV = loadF32(raw.data());
        const auto* body = raw.data() + 4;
        int q = (i % 2 == 0) ? (body[i / 2] & 0xF) : (body[i / 2] >> 4);
        return static_cast<float>(q) * scale + minV;
    }
    case DType::Int2: {
        need(static_cast<std::size_t>((n + 3) / 4));
        int code = (raw[i / 4] >> ((i % 4) * 2)) & 3;
        return static_cast<float>(code - 2) * scale;
    }
    case DType::Uint2: {
        need(4 + static_cast<std::size_t>((n + 3) / 4));
        float minV = loadF32(raw.data());
        const auto* body = raw.data() + 4;
        int code = (body[i / 4] >> ((i % 4) * 2)) & 3;
        return static_cast<float>(code) * scale + minV;
    }
    case DType::Ternary: {
        need(static_cast<std::size_t>((n + 3) / 4));
        int code = (raw[i / 4] >> ((i % 4) * 2)) & 3;
        if (code == 0) {
            return -scale;
        }
        if (code == 2) {
            return scale;
        }
        return 0.f;
    }
    case DType::Binary: {
        need(static_cast<std::size_t>((n + 7) / 8));
        return (raw[i / 8] & (1u << (i % 8))) ? scale : -scale;
    }
    case DType::Int: {
        need(static_cast<std::size_t>(n) * 8); // assume 64-bit Go int
        return static_cast<float>(static_cast<std::int64_t>(loadU64(raw.data() + i * 8))) * scale;
    }
    case DType::Uint:
    case DType::Uintptr: {
        need(4 + static_cast<std::size_t>(n) * 8);
        float minV = loadF32(raw.data());
        return static_cast<float>(loadU64(raw.data() + 4 + i * 8)) * scale + minV;
    }
    case DType::Complex64: {
        need(static_cast<std::size_t>(n) * 8);
        return loadF32(raw.data() + i * 8); // real part
    }
    case DType::Complex128: {
        need(static_cast<std::size_t>(n) * 16);
        return static_cast<float>(loadF64(raw.data() + i * 16));
    }
    case DType::NF4: {
        need(static_cast<std::size_t>((n + 1) / 2));
        int code = (i % 2 == 0) ? (raw[i / 2] & 0xF) : (raw[i / 2] >> 4);
        return kNF4[code] * scale;
    }
    case DType::FP6:
    case DType::Int6: {
        need(static_cast<std::size_t>((n * 6 + 7) / 8));
        return static_cast<float>(signExtend(getBitCode(raw.data(), i, 6), 6)) * scale;
    }
    case DType::Uint6: {
        need(4 + static_cast<std::size_t>((n * 6 + 7) / 8));
        float minV = loadF32(raw.data());
        return static_cast<float>(getBitCode(raw.data() + 4, i, 6)) * scale + minV;
    }
    case DType::Int5: {
        need(static_cast<std::size_t>((n * 5 + 7) / 8));
        return static_cast<float>(signExtend(getBitCode(raw.data(), i, 5), 5)) * scale;
    }
    case DType::Uint5: {
        need(4 + static_cast<std::size_t>((n * 5 + 7) / 8));
        float minV = loadF32(raw.data());
        return static_cast<float>(getBitCode(raw.data() + 4, i, 5)) * scale + minV;
    }
    case DType::Int3: {
        need(static_cast<std::size_t>((n * 3 + 7) / 8));
        return static_cast<float>(signExtend(getBitCode(raw.data(), i, 3), 3)) * scale;
    }
    case DType::Uint3: {
        need(4 + static_cast<std::size_t>((n * 3 + 7) / 8));
        float minV = loadF32(raw.data());
        return static_cast<float>(getBitCode(raw.data() + 4, i, 3)) * scale + minV;
    }
    }
    throw std::runtime_error("unsupported dtype");
}

inline void decodeRow(const NativeStore& s, int row, float* dst) {
    if (row < 0 || row >= s.rows) {
        throw std::runtime_error("row OOB");
    }
    const int off = row * s.cols;
    const int n = s.rows * s.cols;
    if (s.dtype == DType::Float32) {
        if (static_cast<int>(s.raw.size()) < n * 4) {
            throw std::runtime_error("float32 master short");
        }
        std::memcpy(dst, s.raw.data() + off * 4, static_cast<std::size_t>(s.cols) * 4);
        return;
    }
    for (int c = 0; c < s.cols; c++) {
        dst[c] = weightAt(s, off + c, n);
    }
}

} // namespace welvet_imported
