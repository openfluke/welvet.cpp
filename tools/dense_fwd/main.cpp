// Binary wire tool for w2a go_v_cpp exact parity + timing + architecture baseline.
//
// Magics:
//   WVD3 — Dense forward (see prior); WVD2/WVD1 accepted
//   WVA1 — architecture topology
//   WVF1 — volume forward (Dense cells only)
//
// WVA1 stdin:
//   magic "WVA1"
//   i32 depth, rows, cols, lpc
//   i32 n_links
//   n_links × (i32 sz,sy,sx,sl, tz,ty,tx,tl)
// WVA1 stdout:
//   magic "WVAO"
//   i32 n_cells
//   n_cells × (i32 z,y,x,l, index, is_remote, tz,ty,tx,tl)
//
// WVF1 stdin:
//   magic "WVF1"
//   i32 depth, rows, cols, lpc, batch, backend
//   i32 n_links ; links…
//   for each cell in hop order:
//     i32 enabled
//     if enabled: i32 in,out,act,dtype ; f32 W[out*in] ; u32 bias_len ; f64 bias[]
//   f32 x[batch * first_enabled_in]
// WVF1 stdout:
//   magic "WVFO"
//   i32 out_len
//   f32 out[out_len]

#include "welvet_c.h"

#include "welvet/architecture.hpp"
#include "welvet/runtime/forward.hpp"

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

bool write_magic(const char* m) { return write_exact(m, 4); }

int run_arch() {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVA1 header\n");
    return 2;
  }
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVA1 links\n");
    return 2;
  }

  const int32_t max_cells = (depth < 1 ? 1 : depth) * (rows < 1 ? 1 : rows) *
                            (cols < 1 ? 1 : cols) * (lpc < 1 ? 1 : lpc) + 8;
  std::vector<int32_t> coords(static_cast<std::size_t>(max_cells) * 4);
  std::vector<int32_t> indices(static_cast<std::size_t>(max_cells));
  std::vector<int32_t> remote(static_cast<std::size_t>(max_cells) * 5);
  int32_t n_cells = 0;
  welvet_status st =
      welvet_arch_topology(depth, rows, cols, lpc, links.empty() ? nullptr : links.data(), n_links,
                           &n_cells, coords.data(), indices.data(), remote.data());
  if (st.code != 0) {
    std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
    return 1;
  }
  if (!write_magic("WVAO") || !write_exact(&n_cells, 1)) {
    return 2;
  }
  for (int32_t i = 0; i < n_cells; ++i) {
    int32_t row[10] = {coords[static_cast<std::size_t>(i) * 4 + 0],
                       coords[static_cast<std::size_t>(i) * 4 + 1],
                       coords[static_cast<std::size_t>(i) * 4 + 2],
                       coords[static_cast<std::size_t>(i) * 4 + 3],
                       indices[static_cast<std::size_t>(i)],
                       remote[static_cast<std::size_t>(i) * 5 + 0],
                       remote[static_cast<std::size_t>(i) * 5 + 1],
                       remote[static_cast<std::size_t>(i) * 5 + 2],
                       remote[static_cast<std::size_t>(i) * 5 + 3],
                       remote[static_cast<std::size_t>(i) * 5 + 4]};
    if (!write_exact(row, 10)) {
      return 2;
    }
  }
  return 0;
}

int run_volume(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVF1 header\n");
    return 2;
  }
  if (batch <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad batch\n");
    return 2;
  }

  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  for (int32_t i = 0; i < n_links; ++i) {
    int32_t L[8];
    if (!read_exact(L, 8)) {
      std::fprintf(stderr, "welvet_dense_fwd: short WVF1 links\n");
      return 2;
    }
    auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5], L[6],
                                                    L[7]);
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  const auto hops = welvet::architecture::hop_order(g);
  int first_in = -1;
  for (const auto& coord : hops) {
    int32_t enabled = 0;
    if (!read_exact(&enabled, 1)) {
      std::fprintf(stderr, "welvet_dense_fwd: short WVF1 enabled\n");
      return 2;
    }
    if (!enabled) {
      continue;
    }
    int32_t in = 0, out = 0, act = 0, dtype = 1;
    if (!read_exact(&in, 1) || !read_exact(&out, 1) || !read_exact(&act, 1) ||
        !read_exact(&dtype, 1)) {
      std::fprintf(stderr, "welvet_dense_fwd: short WVF1 cell dims\n");
      return 2;
    }
    std::vector<float> W(static_cast<std::size_t>(out * in));
    if (!read_exact(W.data(), W.size())) {
      std::fprintf(stderr, "welvet_dense_fwd: short WVF1 weights\n");
      return 2;
    }
    uint32_t bias_len = 0;
    if (!read_exact(&bias_len, 1)) {
      return 2;
    }
    std::vector<double> bias(bias_len);
    if (bias_len > 0 && !read_exact(bias.data(), bias.size())) {
      return 2;
    }
    if (first_in < 0) {
      first_in = in;
    }

    welvet::dense::Layer layer;
    auto st = welvet::dense::new_layer(in, out, static_cast<welvet::Activation>(act),
                                       static_cast<welvet::DType>(dtype), W.data(), &layer);
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
    layer.backend = backend == 1 ? welvet::Backend::SIMD : welvet::Backend::CPUTiled;
    if (!bias.empty()) {
      layer.weights.bias = std::move(bias);
    }
    st = welvet::architecture::bind_dense(g, coord.z, coord.y, coord.x, coord.l, std::move(layer));
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  if (first_in < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_in};
  input.data.resize(static_cast<std::size_t>(batch * first_in));
  if (!read_exact(input.data.data(), input.data.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVF1 input\n");
    return 2;
  }

  welvet::runtime::forward::Result result;
  auto once = [&]() -> welvet::Status {
    return welvet::runtime::forward::volume_dense(g, input, &result);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = once();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = once();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
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
    auto st = once();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  const int32_t out_len = static_cast<int32_t>(result.output.data.size());
  if (!write_magic("WVFO") || !write_exact(&out_len, 1) ||
      !write_exact(result.output.data.data(), result.output.data.size())) {
    return 2;
  }
  return 0;
}

int run_dense(bool v1, bool v2, bool v3, int bench_iters) {
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
  if (std::memcmp(magic, "WVA1", 4) == 0) {
    return run_arch();
  }
  if (std::memcmp(magic, "WVF1", 4) == 0) {
    return run_volume(bench_iters);
  }
  const bool v3 = std::memcmp(magic, "WVD3", 4) == 0;
  const bool v2 = std::memcmp(magic, "WVD2", 4) == 0;
  const bool v1 = std::memcmp(magic, "WVD1", 4) == 0;
  if (!v1 && !v2 && !v3) {
    std::fprintf(stderr, "welvet_dense_fwd: bad magic (want WVD1/2/3, WVA1, WVF1)\n");
    return 2;
  }
  return run_dense(v1, v2, v3, bench_iters);
}
