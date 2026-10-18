// Binary wire tool for w2a go_v_cpp exact parity + timing.
//
// stdin  (little-endian) WVD3:
//   magic "WVD3" (4)
//   i32 batch, in, out, act, dtype, backend   // backend 0=cpu 1=simd
//   f32 W_init[out*in]
//   u32 bias_len
//   f64 bias[bias_len]
//   f32 x[batch*in]
//
// WVD2 accepted as backend=CPU. WVD1 accepted as dtype=Float32 backend=CPU.
//
// stdout:
//   optional bench header "WVB1" + i64 ns + i32 iters
//   magic "WVO1" (4)
//   f32 pre[batch*out]
//   f32 post[batch*out]

#include "welvet_c.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

template <typename T>
bool read_exact(T* dst, std::size_t n) {
  return std::fread(dst, sizeof(T), n, stdin) == n;
}

template <typename T>
bool write_exact(const T* src, std::size_t n) {
  return std::fwrite(src, sizeof(T), n, stdout) == n;
}

} // namespace

int main(int argc, char** argv) {
  int bench_iters = 0;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--gaps") == 0) {
      welvet_print_gaps();
      return 0;
    }
    if (std::strcmp(argv[i], "--version") == 0) {
      std::printf("%s\n", welvet_version());
      return 0;
    }
    if (std::strncmp(argv[i], "--bench=", 8) == 0) {
      bench_iters = std::atoi(argv[i] + 8);
    }
  }

  char magic[4];
  if (!read_exact(magic, 4)) {
    std::fprintf(stderr, "welvet_dense_fwd: short magic\n");
    return 2;
  }
  const bool v3 = std::memcmp(magic, "WVD3", 4) == 0;
  const bool v2 = std::memcmp(magic, "WVD2", 4) == 0;
  const bool v1 = std::memcmp(magic, "WVD1", 4) == 0;
  if (!v1 && !v2 && !v3) {
    std::fprintf(stderr, "welvet_dense_fwd: bad magic (want WVD1/WVD2/WVD3)\n");
    return 2;
  }

  int32_t batch = 0, in = 0, out = 0, act = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in, 1) || !read_exact(&out, 1) || !read_exact(&act, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short header\n");
    return 2;
  }
  if ((v2 || v3) && !read_exact(&dtype, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short dtype\n");
    return 2;
  }
  if (v3 && !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short backend\n");
    return 2;
  }
  if (batch <= 0 || in <= 0 || out <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad dims\n");
    return 2;
  }

  std::vector<float> W(static_cast<std::size_t>(out * in));
  if (!read_exact(W.data(), W.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short weights\n");
    return 2;
  }

  uint32_t bias_len = 0;
  if (!read_exact(&bias_len, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short bias_len\n");
    return 2;
  }
  std::vector<double> bias(bias_len);
  if (bias_len > 0 && !read_exact(bias.data(), bias.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short bias\n");
    return 2;
  }

  std::vector<float> x(static_cast<std::size_t>(batch * in));
  if (!read_exact(x.data(), x.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short input\n");
    return 2;
  }

  std::vector<float> pre(static_cast<std::size_t>(batch * out));
  std::vector<float> post(static_cast<std::size_t>(batch * out));
  const double* bias_ptr = bias_len ? bias.data() : nullptr;

  auto once = [&]() -> welvet_status {
    return welvet_dense_forward_f32(batch, in, out, act, dtype, backend, W.data(), bias_ptr,
                                    bias_len, x.data(), pre.data(), post.data());
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 5; ++i) {
      welvet_status st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      welvet_status st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) {
      return 2;
    }
  } else {
    welvet_status st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  if (!write_exact("WVO1", 4) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(post.data(), post.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: write failed\n");
    return 2;
  }
  return 0;
}
