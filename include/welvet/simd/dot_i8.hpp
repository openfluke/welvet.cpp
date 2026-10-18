// Copied from welvet-to-rust_n_cpp/welvet_cpp (code only; not documentation).
#pragma once

#include <cstdint>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace welvet {
namespace dot_i8 {

// Signed 16-wide AVX2 i8×i8 → i32; exact integer parity with welvet.
inline std::int32_t dotI8Tile(const std::int8_t* a, const std::int8_t* b, int n, std::int32_t prev) {
#if defined(__AVX2__)
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256i wa = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a+i)));
        const __m256i wb = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b+i)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(wa, wb));
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    std::int32_t sum = prev + _mm_cvtsi128_si32(s);
    for (; i < n; i++) {
        sum += static_cast<std::int32_t>(a[i]) * static_cast<std::int32_t>(b[i]);
    }
    return sum;
#else
    std::int32_t sum = prev;
    for (int i = 0; i < n; i++) {
        sum += static_cast<std::int32_t>(a[i]) * static_cast<std::int32_t>(b[i]);
    }
    return sum;
#endif
}

} // namespace dot_i8
} // namespace welvet
