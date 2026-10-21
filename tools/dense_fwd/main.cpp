// Binary wire tool for w2a go_v_cpp exact parity + timing + architecture baseline.
//
// Magics:
//   WVD3 — Dense forward (see prior); WVD2/WVD1 accepted
//   WVA1 — architecture topology
//   WVF1 — volume forward (Dense cells only)
//   WVM1 — MHA ModeSelf forward
//   WVS1 — SwiGLU forward
//   WVR1 — RMSNorm forward
//   WVL1 — LayerNorm forward
//   WVC1 / WVC2 / WVC3 — CNN1/2/3 forward
//   WVT1 / WVT2 / WVT3 — ConvT1/2/3 forward
//   WRN1 — RNN forward
//   WLS1 — LSTM forward
//   WEM1 — Embedding gather forward
//   WSM1 — Softmax forward (all Kind variants)
//   WKM1 — KMeans soft assignment forward
//   WSQ1 — Sequential Dense chain forward
//   WRE1 — Residual Dense F(x)+x forward
//   WMB1 — Mamba minimal SSM forward
//   WMC1 — Metacognition Observed Dense + rules forward
//   WGD1 — GDN (Gated DeltaNet) Float32 forward
//   WSX1 — seqmix Kind name smoke (no tensors)
//   WPL1 — Parallel Dense branches + combine modes
//   WDB1 — Dense NormalBP leaf (fwd + bwd + optional SGD)
//   WVN1 — Volume Dense NormalBP (fwd tape + MSE + bwd + SGD)
//   WRB1 — RMSNorm NormalBP leaf
//   WRV1 — Volume RMSNorm NormalBP
//   WLB1 — LayerNorm NormalBP leaf
//   WLV1 — Volume LayerNorm NormalBP
//   WSB1 — SwiGLU NormalBP leaf
//   WSV1 — Volume SwiGLU NormalBP
//   WXB1 — Softmax NormalBP leaf (all Kind; dX-only)
//   WXV1 — Volume Softmax NormalBP
//   WEB1 — Embedding NormalBP leaf
//   WEV1 — Volume Embedding NormalBP
//   WHB1 — MHA NormalBP leaf (ModeSelf)
//   WHV1 — Volume MHA NormalBP
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
#include "welvet/layernorm.hpp"
#include "welvet/rmsnorm.hpp"
#include "welvet/runtime/backward.hpp"
#include "welvet/runtime/forward.hpp"
#include "welvet/runtime/training.hpp"
#include "welvet/softmax.hpp"
#include "welvet/swiglu.hpp"

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

int run_mha(int bench_iters) {
  int32_t batch = 0, seq = 0, d_model = 0, num_heads = 0, num_kv = 0, head_dim = 0, msl = 0;
  int32_t mask = 0, pos = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&d_model, 1) ||
      !read_exact(&num_heads, 1) || !read_exact(&num_kv, 1) || !read_exact(&head_dim, 1) ||
      !read_exact(&msl, 1) || !read_exact(&mask, 1) || !read_exact(&pos, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVM1 header\n");
    return 2;
  }
  if (num_heads <= 0 || d_model <= 0 || d_model % num_heads != 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad mha dims\n");
    return 2;
  }
  const int hd = head_dim > 0 ? head_dim : d_model / num_heads;
  const int kv_heads = num_kv > 0 ? num_kv : num_heads;
  const int q_dim = num_heads * hd;
  const int kv_dim = kv_heads * hd;

  std::vector<float> Q(static_cast<std::size_t>(q_dim * d_model));
  std::vector<float> K(static_cast<std::size_t>(kv_dim * d_model));
  std::vector<float> V(static_cast<std::size_t>(kv_dim * d_model));
  std::vector<float> O(static_cast<std::size_t>(d_model * q_dim));
  if (!read_exact(Q.data(), Q.size()) || !read_exact(K.data(), K.size()) ||
      !read_exact(V.data(), V.size()) || !read_exact(O.data(), O.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVM1 weights\n");
    return 2;
  }
  std::vector<float> x(static_cast<std::size_t>(batch * seq * d_model));
  if (!read_exact(x.data(), x.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVM1 input\n");
    return 2;
  }
  std::vector<float> pre(static_cast<std::size_t>(batch * seq * q_dim));
  std::vector<float> post(static_cast<std::size_t>(batch * seq * d_model));

  auto once = [&]() -> welvet_status {
    return welvet_mha_forward_f32(batch, seq, d_model, num_heads, num_kv, head_dim, msl, mask, pos,
                                  dtype, backend, Q.data(), K.data(), V.data(), O.data(), x.data(),
                                  pre.data(), post.data());
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
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

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_magic("WVMO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

int run_swiglu(int bench_iters) {
  int32_t batch = 0, seq = 0, input_dim = 0, inter = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&input_dim, 1) ||
      !read_exact(&inter, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVS1 header\n");
    return 2;
  }
  if (batch <= 0 || seq <= 0 || input_dim <= 0 || inter <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad swiglu dims\n");
    return 2;
  }

  std::vector<float> Gate(static_cast<std::size_t>(inter * input_dim));
  std::vector<float> Up(static_cast<std::size_t>(inter * input_dim));
  std::vector<float> Down(static_cast<std::size_t>(input_dim * inter));
  if (!read_exact(Gate.data(), Gate.size()) || !read_exact(Up.data(), Up.size()) ||
      !read_exact(Down.data(), Down.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVS1 weights\n");
    return 2;
  }
  std::vector<float> x(static_cast<std::size_t>(batch * seq * input_dim));
  if (!read_exact(x.data(), x.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVS1 input\n");
    return 2;
  }
  std::vector<float> pre(static_cast<std::size_t>(batch * seq * inter));
  std::vector<float> post(static_cast<std::size_t>(batch * seq * input_dim));

  auto once = [&]() -> welvet_status {
    return welvet_swiglu_forward_f32(batch, seq, input_dim, inter, dtype, backend, Gate.data(),
                                     Up.data(), Down.data(), x.data(), pre.data(), post.data());
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
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

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_magic("WVSO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

int run_rmsnorm(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, dtype = 1, backend = 0;
  double eps = 0.0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) ||
      !read_exact(&eps, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVR1 header\n");
    return 2;
  }
  if (batch <= 0 || seq <= 0 || dim <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad rmsnorm dims\n");
    return 2;
  }

  std::vector<float> gamma(static_cast<std::size_t>(dim));
  if (!read_exact(gamma.data(), gamma.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVR1 gamma\n");
    return 2;
  }
  std::vector<float> x(static_cast<std::size_t>(batch * seq * dim));
  if (!read_exact(x.data(), x.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVR1 input\n");
    return 2;
  }
  std::vector<float> pre(static_cast<std::size_t>(batch * seq * dim));
  std::vector<float> post(static_cast<std::size_t>(batch * seq * dim));

  auto once = [&]() -> welvet_status {
    return welvet_rmsnorm_forward_f32(batch, seq, dim, eps, dtype, backend, gamma.data(), x.data(),
                                      pre.data(), post.data());
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
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

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_magic("WVRO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

int run_layernorm(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, dtype = 1, backend = 0;
  double eps = 0.0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) ||
      !read_exact(&eps, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVL1 header\n");
    return 2;
  }
  if (batch <= 0 || seq <= 0 || dim <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad layernorm dims\n");
    return 2;
  }

  std::vector<float> gamma(static_cast<std::size_t>(dim));
  std::vector<float> beta(static_cast<std::size_t>(dim));
  if (!read_exact(gamma.data(), gamma.size()) || !read_exact(beta.data(), beta.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVL1 affine\n");
    return 2;
  }
  std::vector<float> x(static_cast<std::size_t>(batch * seq * dim));
  if (!read_exact(x.data(), x.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVL1 input\n");
    return 2;
  }
  std::vector<float> pre(static_cast<std::size_t>(batch * seq * dim));
  std::vector<float> post(static_cast<std::size_t>(batch * seq * dim));

  auto once = [&]() -> welvet_status {
    return welvet_layernorm_forward_f32(batch, seq, dim, eps, dtype, backend, gamma.data(),
                                        beta.data(), x.data(), pre.data(), post.data());
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
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

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_magic("WVLO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

int write_cnn_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_magic("WVCO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

template <typename Fn>
int bench_or_once(int bench_iters, Fn&& once, std::vector<float>& pre, std::vector<float>& post) {
  (void)pre;
  (void)post;
  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
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
  return 0;
}

int run_cnn1(int bench_iters) {
  int32_t batch = 0, in_c = 0, filters = 0, seq = 0, kernel = 0, stride = 0, padding = 0;
  int32_t act = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in_c, 1) || !read_exact(&filters, 1) ||
      !read_exact(&seq, 1) || !read_exact(&kernel, 1) || !read_exact(&stride, 1) ||
      !read_exact(&padding, 1) || !read_exact(&act, 1) || !read_exact(&dtype, 1) ||
      !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVC1 header\n");
    return 2;
  }
  const int patch = in_c * kernel;
  const int out_len = (seq + 2 * padding - kernel) / (stride <= 0 ? 1 : stride) + 1;
  std::vector<float> W(static_cast<std::size_t>(filters * patch));
  if (!read_exact(W.data(), W.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVC1 weights\n");
    return 2;
  }
  uint32_t bias_len = 0;
  if (!read_exact(&bias_len, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVC1 bias_len\n");
    return 2;
  }
  std::vector<double> bias(bias_len);
  if (bias_len && !read_exact(bias.data(), bias.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVC1 bias\n");
    return 2;
  }
  std::vector<float> x(static_cast<std::size_t>(batch * in_c * seq));
  if (!read_exact(x.data(), x.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVC1 input\n");
    return 2;
  }
  std::vector<float> pre(static_cast<std::size_t>(batch * filters * out_len));
  std::vector<float> post(pre.size());
  auto once = [&]() -> welvet_status {
    return welvet_cnn1_forward_f32(batch, in_c, filters, seq, kernel, stride, padding, act, dtype,
                                   backend, W.data(), bias.data(), bias.size(), x.data(),
                                   pre.data(), post.data());
  };
  if (int rc = bench_or_once(bench_iters, once, pre, post); rc != 0) return rc;
  return write_cnn_out(pre, post);
}

int run_cnn2(int bench_iters) {
  int32_t batch = 0, in_c = 0, filters = 0, height = 0, width = 0, kernel = 0, stride = 0,
          padding = 0;
  int32_t act = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in_c, 1) || !read_exact(&filters, 1) ||
      !read_exact(&height, 1) || !read_exact(&width, 1) || !read_exact(&kernel, 1) ||
      !read_exact(&stride, 1) || !read_exact(&padding, 1) || !read_exact(&act, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVC2 header\n");
    return 2;
  }
  const int s = stride <= 0 ? 1 : stride;
  const int out_h = (height + 2 * padding - kernel) / s + 1;
  const int out_w = (width + 2 * padding - kernel) / s + 1;
  const int patch = in_c * kernel * kernel;
  std::vector<float> W(static_cast<std::size_t>(filters * patch));
  if (!read_exact(W.data(), W.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVC2 weights\n");
    return 2;
  }
  uint32_t bias_len = 0;
  if (!read_exact(&bias_len, 1)) return 2;
  std::vector<double> bias(bias_len);
  if (bias_len && !read_exact(bias.data(), bias.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * in_c * height * width));
  if (!read_exact(x.data(), x.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVC2 input\n");
    return 2;
  }
  std::vector<float> pre(static_cast<std::size_t>(batch * filters * out_h * out_w));
  std::vector<float> post(pre.size());
  auto once = [&]() -> welvet_status {
    return welvet_cnn2_forward_f32(batch, in_c, filters, height, width, kernel, stride, padding,
                                   act, dtype, backend, W.data(), bias.data(), bias.size(),
                                   x.data(), pre.data(), post.data());
  };
  if (int rc = bench_or_once(bench_iters, once, pre, post); rc != 0) return rc;
  return write_cnn_out(pre, post);
}

int run_cnn3(int bench_iters) {
  int32_t batch = 0, in_c = 0, filters = 0, depth = 0, height = 0, width = 0, kernel = 0,
          stride = 0, padding = 0;
  int32_t act = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in_c, 1) || !read_exact(&filters, 1) ||
      !read_exact(&depth, 1) || !read_exact(&height, 1) || !read_exact(&width, 1) ||
      !read_exact(&kernel, 1) || !read_exact(&stride, 1) || !read_exact(&padding, 1) ||
      !read_exact(&act, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVC3 header\n");
    return 2;
  }
  const int s = stride <= 0 ? 1 : stride;
  const int out_d = (depth + 2 * padding - kernel) / s + 1;
  const int out_h = (height + 2 * padding - kernel) / s + 1;
  const int out_w = (width + 2 * padding - kernel) / s + 1;
  const int patch = in_c * kernel * kernel * kernel;
  std::vector<float> W(static_cast<std::size_t>(filters * patch));
  if (!read_exact(W.data(), W.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVC3 weights\n");
    return 2;
  }
  uint32_t bias_len = 0;
  if (!read_exact(&bias_len, 1)) return 2;
  std::vector<double> bias(bias_len);
  if (bias_len && !read_exact(bias.data(), bias.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * in_c * depth * height * width));
  if (!read_exact(x.data(), x.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVC3 input\n");
    return 2;
  }
  std::vector<float> pre(static_cast<std::size_t>(batch * filters * out_d * out_h * out_w));
  std::vector<float> post(pre.size());
  auto once = [&]() -> welvet_status {
    return welvet_cnn3_forward_f32(batch, in_c, filters, depth, height, width, kernel, stride,
                                   padding, act, dtype, backend, W.data(), bias.data(),
                                   bias.size(), x.data(), pre.data(), post.data());
  };
  if (int rc = bench_or_once(bench_iters, once, pre, post); rc != 0) return rc;
  return write_cnn_out(pre, post);
}

int write_convt_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_magic("WVTO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

int run_convt1(int bench_iters) {
  int32_t batch = 0, in_c = 0, filters = 0, seq = 0, kernel = 0, stride = 0, padding = 0,
          out_pad = 0, act = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in_c, 1) || !read_exact(&filters, 1) ||
      !read_exact(&seq, 1) || !read_exact(&kernel, 1) || !read_exact(&stride, 1) ||
      !read_exact(&padding, 1) || !read_exact(&out_pad, 1) || !read_exact(&act, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVT1 header\n");
    return 2;
  }
  const int s = stride <= 0 ? 1 : stride;
  const int out_len = (seq - 1) * s - 2 * padding + kernel + out_pad;
  const int patch = in_c * kernel;
  std::vector<float> W(static_cast<std::size_t>(filters * patch));
  if (!read_exact(W.data(), W.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * in_c * seq));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(static_cast<std::size_t>(batch * filters * out_len));
  std::vector<float> post(pre.size());
  auto once = [&]() -> welvet_status {
    return welvet_convt1_forward_f32(batch, in_c, filters, seq, kernel, stride, padding, out_pad,
                                     act, dtype, backend, W.data(), x.data(), pre.data(),
                                     post.data());
  };
  if (int rc = bench_or_once(bench_iters, once, pre, post); rc != 0) return rc;
  return write_convt_out(pre, post);
}

int run_convt2(int bench_iters) {
  int32_t batch = 0, in_c = 0, filters = 0, height = 0, width = 0, kernel = 0, stride = 0,
          padding = 0, out_pad = 0, act = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in_c, 1) || !read_exact(&filters, 1) ||
      !read_exact(&height, 1) || !read_exact(&width, 1) || !read_exact(&kernel, 1) ||
      !read_exact(&stride, 1) || !read_exact(&padding, 1) || !read_exact(&out_pad, 1) ||
      !read_exact(&act, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVT2 header\n");
    return 2;
  }
  const int s = stride <= 0 ? 1 : stride;
  const int out_h = (height - 1) * s - 2 * padding + kernel + out_pad;
  const int out_w = (width - 1) * s - 2 * padding + kernel + out_pad;
  const int patch = in_c * kernel * kernel;
  std::vector<float> W(static_cast<std::size_t>(filters * patch));
  if (!read_exact(W.data(), W.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * in_c * height * width));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(static_cast<std::size_t>(batch * filters * out_h * out_w));
  std::vector<float> post(pre.size());
  auto once = [&]() -> welvet_status {
    return welvet_convt2_forward_f32(batch, in_c, filters, height, width, kernel, stride, padding,
                                     out_pad, act, dtype, backend, W.data(), x.data(), pre.data(),
                                     post.data());
  };
  if (int rc = bench_or_once(bench_iters, once, pre, post); rc != 0) return rc;
  return write_convt_out(pre, post);
}

int run_convt3(int bench_iters) {
  int32_t batch = 0, in_c = 0, filters = 0, depth = 0, height = 0, width = 0, kernel = 0,
          stride = 0, padding = 0, out_pad = 0, act = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in_c, 1) || !read_exact(&filters, 1) ||
      !read_exact(&depth, 1) || !read_exact(&height, 1) || !read_exact(&width, 1) ||
      !read_exact(&kernel, 1) || !read_exact(&stride, 1) || !read_exact(&padding, 1) ||
      !read_exact(&out_pad, 1) || !read_exact(&act, 1) || !read_exact(&dtype, 1) ||
      !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVT3 header\n");
    return 2;
  }
  const int s = stride <= 0 ? 1 : stride;
  const int out_d = (depth - 1) * s - 2 * padding + kernel + out_pad;
  const int out_h = (height - 1) * s - 2 * padding + kernel + out_pad;
  const int out_w = (width - 1) * s - 2 * padding + kernel + out_pad;
  const int patch = in_c * kernel * kernel * kernel;
  std::vector<float> W(static_cast<std::size_t>(filters * patch));
  if (!read_exact(W.data(), W.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * in_c * depth * height * width));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(static_cast<std::size_t>(batch * filters * out_d * out_h * out_w));
  std::vector<float> post(pre.size());
  auto once = [&]() -> welvet_status {
    return welvet_convt3_forward_f32(batch, in_c, filters, depth, height, width, kernel, stride,
                                     padding, out_pad, act, dtype, backend, W.data(), x.data(),
                                     pre.data(), post.data());
  };
  if (int rc = bench_or_once(bench_iters, once, pre, post); rc != 0) return rc;
  return write_convt_out(pre, post);
}

int write_rnn_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_magic("WRNO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

int write_lstm_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_magic("WLSO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

int run_rnn(int bench_iters) {
  int32_t batch = 0, input_size = 0, hidden_size = 0, seq_len = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&input_size, 1) || !read_exact(&hidden_size, 1) ||
      !read_exact(&seq_len, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WRN1 header\n");
    return 2;
  }
  const int wn = hidden_size * input_size + hidden_size * hidden_size + hidden_size;
  std::vector<float> packed(static_cast<std::size_t>(wn));
  if (!read_exact(packed.data(), packed.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq_len * input_size));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(static_cast<std::size_t>(batch * seq_len * hidden_size));
  std::vector<float> post(pre.size());
  auto once = [&]() -> welvet_status {
    return welvet_rnn_forward_f32(batch, input_size, hidden_size, seq_len, dtype, backend,
                                  packed.data(), x.data(), pre.data(), post.data());
  };
  if (int rc = bench_or_once(bench_iters, once, pre, post); rc != 0) return rc;
  return write_rnn_out(pre, post);
}

int run_lstm(int bench_iters) {
  int32_t batch = 0, input_size = 0, hidden_size = 0, seq_len = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&input_size, 1) || !read_exact(&hidden_size, 1) ||
      !read_exact(&seq_len, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WLS1 header\n");
    return 2;
  }
  const int gate = hidden_size * input_size + hidden_size * hidden_size + hidden_size;
  const int wn = 4 * gate;
  std::vector<float> packed(static_cast<std::size_t>(wn));
  if (!read_exact(packed.data(), packed.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq_len * input_size));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(static_cast<std::size_t>(batch * seq_len * 5 * hidden_size));
  std::vector<float> post(static_cast<std::size_t>(batch * seq_len * hidden_size));
  auto once = [&]() -> welvet_status {
    return welvet_lstm_forward_f32(batch, input_size, hidden_size, seq_len, dtype, backend,
                                   packed.data(), x.data(), pre.data(), post.data());
  };
  if (int rc = bench_or_once(bench_iters, once, pre, post); rc != 0) return rc;
  return write_lstm_out(pre, post);
}

int write_emb_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_magic("WEMO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

int run_embedding(int bench_iters) {
  int32_t batch = 0, vocab = 0, emb = 0, seq = 0, in_rank = 2, in_last = 1, act = -1, dtype = 1,
          backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&vocab, 1) || !read_exact(&emb, 1) ||
      !read_exact(&seq, 1) || !read_exact(&in_rank, 1) || !read_exact(&in_last, 1) ||
      !read_exact(&act, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WEM1 header\n");
    return 2;
  }
  std::vector<float> W(static_cast<std::size_t>(vocab * emb));
  if (!read_exact(W.data(), W.size())) return 2;
  const int x_n = batch * seq * in_last;
  std::vector<float> x(static_cast<std::size_t>(x_n));
  if (!read_exact(x.data(), x.size())) return 2;
  const int out_n = (in_rank == 1 ? seq : batch * seq) * emb;
  std::vector<float> pre(static_cast<std::size_t>(out_n));
  std::vector<float> post(pre.size());
  auto once = [&]() -> welvet_status {
    return welvet_embedding_forward_f32(batch, vocab, emb, seq, in_rank, in_last, act, dtype,
                                        backend, W.data(), x.data(), pre.data(), post.data());
  };
  if (int rc = bench_or_once(bench_iters, once, pre, post); rc != 0) return rc;
  return write_emb_out(pre, post);
}

int write_sm_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_magic("WSMO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

int run_softmax(int bench_iters) {
  int32_t n = 0, dim = 0, seq = 0, kind = 0, rows = 0, cols = 0, gumbel_seed = 0, dtype = 1,
          backend = 0, rank = 0, mask_len = 0, hier_len = 0;
  double temperature = 1.0, entmax_alpha = 0.0;
  if (!read_exact(&n, 1) || !read_exact(&dim, 1) || !read_exact(&seq, 1) || !read_exact(&kind, 1) ||
      !read_exact(&rows, 1) || !read_exact(&cols, 1) || !read_exact(&temperature, 1) ||
      !read_exact(&entmax_alpha, 1) || !read_exact(&gumbel_seed, 1) || !read_exact(&dtype, 1) ||
      !read_exact(&backend, 1) || !read_exact(&rank, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WSM1 header\n");
    return 2;
  }
  if (rank < 1 || rank > 8) {
    std::fprintf(stderr, "welvet_dense_fwd: bad softmax rank\n");
    return 2;
  }
  std::vector<int32_t> shape(static_cast<std::size_t>(rank));
  if (!read_exact(shape.data(), shape.size())) return 2;
  if (!read_exact(&mask_len, 1)) return 2;
  std::vector<uint8_t> mask(static_cast<std::size_t>(mask_len > 0 ? mask_len : 0));
  if (mask_len > 0 && !read_exact(mask.data(), mask.size())) return 2;
  if (!read_exact(&hier_len, 1)) return 2;
  std::vector<int32_t> hier(static_cast<std::size_t>(hier_len > 0 ? hier_len : 0));
  if (hier_len > 0 && !read_exact(hier.data(), hier.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(n));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(static_cast<std::size_t>(n));
  std::vector<float> post(pre.size());
  auto once = [&]() -> welvet_status {
    return welvet_softmax_forward_f32(
        n, dim, seq, kind, rows, cols, temperature, entmax_alpha, gumbel_seed, dtype, backend, rank,
        shape.data(), mask_len > 0 ? mask.data() : nullptr, mask_len,
        hier_len > 0 ? hier.data() : nullptr, hier_len, x.data(), pre.data(), post.data());
  };
  if (int rc = bench_or_once(bench_iters, once, pre, post); rc != 0) return rc;
  return write_sm_out(pre, post);
}

int write_km_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_magic("WKMO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

int run_kmeans(int bench_iters) {
  int32_t n = 0, feature_dim = 0, num_clusters = 0, output_mode = 0, act = -1, dtype = 1,
          backend = 0, rank = 0;
  double temperature = 1.0;
  if (!read_exact(&n, 1) || !read_exact(&feature_dim, 1) || !read_exact(&num_clusters, 1) ||
      !read_exact(&output_mode, 1) || !read_exact(&act, 1) || !read_exact(&temperature, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1) || !read_exact(&rank, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WKM1 header\n");
    return 2;
  }
  if (rank < 1 || rank > 8) {
    std::fprintf(stderr, "welvet_dense_fwd: bad kmeans rank\n");
    return 2;
  }
  std::vector<int32_t> shape(static_cast<std::size_t>(rank));
  if (!read_exact(shape.data(), shape.size())) return 2;
  std::vector<float> W(static_cast<std::size_t>(num_clusters * feature_dim));
  if (!read_exact(W.data(), W.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(n));
  if (!read_exact(x.data(), x.size())) return 2;
  int batch = 1;
  for (int i = 0; i < rank - 1; ++i) batch *= shape[static_cast<std::size_t>(i)];
  const int out_dim = output_mode == 1 ? feature_dim : num_clusters;
  std::vector<float> pre(static_cast<std::size_t>(batch * out_dim));
  std::vector<float> post(pre.size());
  auto once = [&]() -> welvet_status {
    return welvet_kmeans_forward_f32(n, feature_dim, num_clusters, output_mode, act, temperature,
                                     dtype, backend, rank, shape.data(), W.data(), x.data(),
                                     pre.data(), post.data());
  };
  if (int rc = bench_or_once(bench_iters, once, pre, post); rc != 0) return rc;
  return write_km_out(pre, post);
}

int write_sq_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_magic("WSQO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

int run_sequential(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, depth = 0, child_act = -1, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) ||
      !read_exact(&depth, 1) || !read_exact(&child_act, 1) || !read_exact(&dtype, 1) ||
      !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WSQ1 header\n");
    return 2;
  }
  const int wn = depth * dim * dim;
  std::vector<float> packed(static_cast<std::size_t>(wn));
  if (!read_exact(packed.data(), packed.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq * dim));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(x.size());
  std::vector<float> post(x.size());
  auto once = [&]() -> welvet_status {
    return welvet_sequential_forward_f32(batch, seq, dim, depth, child_act, dtype, backend,
                                         packed.data(), x.data(), pre.data(), post.data());
  };
  if (int rc = bench_or_once(bench_iters, once, pre, post); rc != 0) return rc;
  return write_sq_out(pre, post);
}

int write_re_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_magic("WREO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

int run_residual(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, depth = 0, child_act = -1, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) ||
      !read_exact(&depth, 1) || !read_exact(&child_act, 1) || !read_exact(&dtype, 1) ||
      !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WRE1 header\n");
    return 2;
  }
  const int wn = depth * dim * dim;
  std::vector<float> packed(static_cast<std::size_t>(wn));
  if (!read_exact(packed.data(), packed.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq * dim));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(x.size());
  std::vector<float> post(x.size());
  auto once = [&]() -> welvet_status {
    return welvet_residual_forward_f32(batch, seq, dim, depth, child_act, dtype, backend,
                                       packed.data(), x.data(), pre.data(), post.data());
  };
  if (int rc = bench_or_once(bench_iters, once, pre, post); rc != 0) return rc;
  return write_re_out(pre, post);
}

int write_mb_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_magic("WMBO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

int run_mamba(int bench_iters) {
  int32_t batch = 0, seq = 0, d_model = 0, d_state = 0, expand = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&d_model, 1) ||
      !read_exact(&d_state, 1) || !read_exact(&expand, 1) || !read_exact(&dtype, 1) ||
      !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WMB1 header\n");
    return 2;
  }
  int expand_eff = expand;
  if (expand_eff <= 0) expand_eff = 2;
  const int inner = expand_eff * d_model;
  const int wn = inner * (3 * d_model + 2);
  std::vector<float> packed(static_cast<std::size_t>(wn));
  if (!read_exact(packed.data(), packed.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq * d_model));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(x.size());
  std::vector<float> post(x.size());
  auto once = [&]() -> welvet_status {
    return welvet_mamba_forward_f32(batch, seq, d_model, d_state, expand, dtype, backend,
                                    packed.data(), x.data(), pre.data(), post.data());
  };
  if (int rc = bench_or_once(bench_iters, once, pre, post); rc != 0) return rc;
  return write_mb_out(pre, post);
}

int write_mc_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_magic("WMCO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

int run_metacognition(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WMC1 header\n");
    return 2;
  }
  const int wn = dim * dim;
  std::vector<float> packed(static_cast<std::size_t>(wn));
  if (!read_exact(packed.data(), packed.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq * dim));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(x.size());
  std::vector<float> post(x.size());
  auto once = [&]() -> welvet_status {
    return welvet_metacognition_forward_f32(batch, seq, dim, dtype, backend, packed.data(),
                                            x.data(), pre.data(), post.data());
  };
  if (int rc = bench_or_once(bench_iters, once, pre, post); rc != 0) return rc;
  return write_mc_out(pre, post);
}

int write_gd_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_magic("WGDO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

int run_gdn(int bench_iters) {
  int32_t batch = 0, seq = 0, hidden = 0, nk = 0, nv = 0, hd_k = 0, hd_v = 0, conv_k = 0;
  int32_t dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&hidden, 1) ||
      !read_exact(&nk, 1) || !read_exact(&nv, 1) || !read_exact(&hd_k, 1) || !read_exact(&hd_v, 1) ||
      !read_exact(&conv_k, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WGD1 header\n");
    return 2;
  }
  int conv_eff = conv_k;
  if (conv_eff < 1) conv_eff = 4;
  const int key_d = nk * hd_k;
  const int val_d = nv * hd_v;
  const int conv_d = key_d * 2 + val_d;
  const int wn = conv_d * hidden + val_d * hidden + nv * hidden + nv * hidden + hidden * val_d +
                 conv_d * conv_eff + nv + nv + hd_v;
  std::vector<float> packed(static_cast<std::size_t>(wn));
  if (!read_exact(packed.data(), packed.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq * hidden));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(x.size());
  std::vector<float> post(x.size());
  auto once = [&]() -> welvet_status {
    return welvet_gdn_forward_f32(batch, seq, hidden, nk, nv, hd_k, hd_v, conv_k, dtype, backend,
                                  packed.data(), x.data(), pre.data(), post.data());
  };
  if (int rc = bench_or_once(bench_iters, once, pre, post); rc != 0) return rc;
  return write_gd_out(pre, post);
}

int run_seqmix(int /*bench_iters*/) {
  int32_t n = 0;
  if (!read_exact(&n, 1) || n < 0 || n > 64) {
    std::fprintf(stderr, "welvet_dense_fwd: short/bad WSX1 header\n");
    return 2;
  }
  std::vector<int32_t> kinds(static_cast<std::size_t>(n));
  if (n > 0 && !read_exact(kinds.data(), kinds.size())) return 2;
  if (!write_magic("WSXO") || !write_exact(&n, 1)) return 2;
  for (int32_t k : kinds) {
    char name[64]{};
    welvet_status st = welvet_seqmix_kind_name(k, name, static_cast<int32_t>(sizeof(name)));
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: seqmix kind %d: %s\n", static_cast<int>(k),
                   st.message);
      return 2;
    }
    const int32_t len = static_cast<int32_t>(std::strlen(name));
    if (!write_exact(&len, 1) || !write_exact(name, static_cast<std::size_t>(len))) return 2;
  }
  // Mixer tags: mha, mamba, gdn
  const int32_t tags[3] = {welvet_seqmix_mha_kind(), welvet_seqmix_mamba_kind(),
                           welvet_seqmix_gdn_kind()};
  const int32_t n_tags = 3;
  if (!write_exact(&n_tags, 1) || !write_exact(tags, 3)) return 2;
  return 0;
}

int write_pl_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_magic("WPLO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

int run_parallel(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, out_feat = 0, branches = 0, combine = 0, sparse_k = 0;
  int32_t dtype = 1, backend = 0;
  double disagree_beta = 0.0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) ||
      !read_exact(&out_feat, 1) || !read_exact(&branches, 1) || !read_exact(&combine, 1) ||
      !read_exact(&sparse_k, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1) ||
      !read_exact(&disagree_beta, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WPL1 header\n");
    return 2;
  }
  int gate_n = 0;
  if (combine == 3) gate_n = branches * dim; // filter
  const int wn = branches * out_feat * dim + gate_n;
  std::vector<float> packed(static_cast<std::size_t>(wn));
  if (!read_exact(packed.data(), packed.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq * dim));
  if (!read_exact(x.data(), x.size())) return 2;
  int out_d = out_feat;
  if (combine == 0) out_d = branches * out_feat; // concat
  std::vector<float> pre(x.size());
  std::vector<float> post(static_cast<std::size_t>(batch * seq * out_d));
  auto once = [&]() -> welvet_status {
    return welvet_parallel_forward_f32(batch, seq, dim, out_feat, branches, combine, sparse_k,
                                       dtype, backend, disagree_beta, packed.data(), x.data(),
                                       pre.data(), post.data());
  };
  if (int rc = bench_or_once(bench_iters, once, pre, post); rc != 0) return rc;
  return write_pl_out(pre, post);
}

// WDB1 — Dense NormalBP leaf
// stdin: batch,in,out,act,dtype,backend ; W ; bias_len,bias ; x ; use_target ; grad_or_target ; lr
// stdout WDBO: pre_n,pre ; post_n,post ; dx_n,dX ; dw_n,dW ; w_n,W_after ; loss
int run_dense_normalbp(int bench_iters) {
  int32_t batch = 0, in = 0, out = 0, act = -1, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in, 1) || !read_exact(&out, 1) || !read_exact(&act, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WDB1 header\n");
    return 2;
  }
  std::vector<float> W(static_cast<std::size_t>(out * in));
  if (!read_exact(W.data(), W.size())) return 2;
  uint32_t bias_len = 0;
  if (!read_exact(&bias_len, 1)) return 2;
  std::vector<double> bias(bias_len);
  if (bias_len > 0 && !read_exact(bias.data(), bias.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * in));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(static_cast<std::size_t>(batch * out));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  std::vector<float> pre(static_cast<std::size_t>(batch * out));
  std::vector<float> post(static_cast<std::size_t>(batch * out));
  std::vector<float> dX(static_cast<std::size_t>(batch * in));
  std::vector<float> dW(static_cast<std::size_t>(out * in));
  std::vector<float> W_after(static_cast<std::size_t>(out * in));
  double loss = 0.0;

  auto once = [&]() -> welvet_status {
    return welvet_dense_normalbp_f32(batch, in, out, act, dtype, backend, W.data(),
                                     bias.empty() ? nullptr : bias.data(), bias.size(), x.data(),
                                     gyt.data(), use_target, lr, pre.data(), post.data(), dX.data(),
                                     dW.data(), W_after.data(), &loss);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    // restore W each bench iter via ABI re-init (once rebuilds from W)
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = once();
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
    auto st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = static_cast<int32_t>(dW.size());
  const int32_t w_n = static_cast<int32_t>(W_after.size());
  if (!write_magic("WDBO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(dW.data(), dW.size()) || !write_exact(&w_n, 1) ||
      !write_exact(W_after.data(), W_after.size()) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

// WVN1 — Volume Dense NormalBP (same cell layout as WVF1 + target + lr)
// stdout: optional WVB1 bench prefix; WVNO: loss ; gin_n,gradIn ; n_cells ; per cell ; out_n,post
int run_volume_normalbp(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVN1 header\n");
    return 2;
  }
  if (batch <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad batch\n");
    return 2;
  }
  (void)backend;

  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int in = 0, out = 0, act = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
    std::vector<double> bias;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_in = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    int32_t in = 0, out = 0, act = 0, dtype = 1;
    if (!read_exact(&in, 1) || !read_exact(&out, 1) || !read_exact(&act, 1) || !read_exact(&dtype, 1)) {
      return 2;
    }
    c.in = in;
    c.out = out;
    c.act = act;
    c.dtype = dtype;
    c.W.resize(static_cast<std::size_t>(out * in));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    uint32_t bias_len = 0;
    if (!read_exact(&bias_len, 1)) return 2;
    c.bias.resize(bias_len);
    if (bias_len > 0 && !read_exact(c.bias.data(), c.bias.size())) return 2;
    if (first_in < 0) first_in = in;
    cells.push_back(std::move(c));
  }
  if (first_in < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_in};
  input.data.resize(static_cast<std::size_t>(batch * first_in));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_out = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (it->enabled) {
      final_out = it->out;
      break;
    }
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_out};
  target.data.resize(static_cast<std::size_t>(batch * final_out));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::dense::Layer layer;
      auto st = welvet::dense::new_layer(c.in, c.out, static_cast<welvet::Activation>(c.act),
                                         static_cast<welvet::DType>(c.dtype), c.W.data(), &layer);
      if (!st) return st;
      layer.backend = welvet::Backend::CPUTiled;
      if (!c.bias.empty()) layer.weights.bias = c.bias;
      st = welvet::architecture::bind_dense(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                            std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) {
      return st;
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
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
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  if (!write_magic("WVNO") || !write_exact(&loss, 1)) return 2;
  const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
  if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
    return 2;
  }
  int32_t n_cells = 0;
  for (const auto& c : cells) {
    if (c.enabled) ++n_cells;
  }
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int32_t o = c.out, i = c.in;
    std::vector<float> Wa(static_cast<std::size_t>(o * i));
    auto est = cell->dense_op.weights.export_f32(Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    if (!write_exact(&o, 1) || !write_exact(&i, 1) || !write_exact(Wa.data(), Wa.size())) return 2;
  }
  const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
  if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
    return 2;
  }
  return 0;
}

// WRB1 — RMSNorm NormalBP leaf
// stdin: batch,seq,dim,eps,dtype,backend ; gamma ; x ; use_target ; grad_or_target ; lr
// stdout WRBO: pre ; post ; dX ; dW(dγ) ; gamma_after ; loss
int run_rmsnorm_normalbp(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, dtype = 1, backend = 0;
  double eps = 0.0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) || !read_exact(&eps, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WRB1 header\n");
    return 2;
  }
  std::vector<float> gamma(static_cast<std::size_t>(dim));
  if (!read_exact(gamma.data(), gamma.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch) * static_cast<std::size_t>(seq) *
                       static_cast<std::size_t>(dim));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(x.size());
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  std::vector<float> pre(x.size()), post(x.size()), dX(x.size()), dW(static_cast<std::size_t>(dim)),
      gamma_after(static_cast<std::size_t>(dim));
  double loss = 0.0;

  auto once = [&]() -> welvet_status {
    return welvet_rmsnorm_normalbp_f32(batch, seq, dim, eps, dtype, backend, gamma.data(), x.data(),
                                       gyt.data(), use_target, lr, pre.data(), post.data(), dX.data(),
                                       dW.data(), gamma_after.data(), &loss);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = once();
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
    auto st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = static_cast<int32_t>(dW.size());
  const int32_t g_n = static_cast<int32_t>(gamma_after.size());
  if (!write_magic("WRBO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(dW.data(), dW.size()) || !write_exact(&g_n, 1) ||
      !write_exact(gamma_after.data(), gamma_after.size()) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

// WRV1 — Volume RMSNorm NormalBP
// stdin: depth,rows,cols,lpc,batch,backend,n_links ; links ; per hop: enabled[,dim,dtype,eps,gamma]
//        x ; target ; lr
// stdout: optional WVB1; WRVO: loss ; gin ; n_cells ; per cell dim+gamma_after ; post
int run_volume_rmsnorm_normalbp(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WRV1 header\n");
    return 2;
  }
  if (batch <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad batch\n");
    return 2;
  }
  (void)backend;

  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int dim = 0;
    int dtype = 1;
    double eps = 0.0;
    welvet::architecture::Coord coord{};
    std::vector<float> gamma;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_dim = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    int32_t dim = 0, dtype = 1;
    double eps = 0.0;
    if (!read_exact(&dim, 1) || !read_exact(&dtype, 1) || !read_exact(&eps, 1)) return 2;
    c.dim = dim;
    c.dtype = dtype;
    c.eps = eps;
    c.gamma.resize(static_cast<std::size_t>(dim));
    if (!read_exact(c.gamma.data(), c.gamma.size())) return 2;
    if (first_dim < 0) first_dim = dim;
    cells.push_back(std::move(c));
  }
  if (first_dim < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_dim};
  input.data.resize(static_cast<std::size_t>(batch * first_dim));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_dim = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (it->enabled) {
      final_dim = it->dim;
      break;
    }
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_dim};
  target.data.resize(static_cast<std::size_t>(batch * final_dim));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::rmsnorm::Config cfg;
      cfg.dim = c.dim;
      cfg.eps = c.eps;
      welvet::rmsnorm::Layer layer;
      auto st = welvet::rmsnorm::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                                c.gamma.data(), &layer);
      if (!st) return st;
      layer.backend = welvet::Backend::CPUTiled;
      st = welvet::architecture::bind_rmsnorm(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                              std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) {
      return st;
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
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
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  if (!write_magic("WRVO") || !write_exact(&loss, 1)) return 2;
  const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
  if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
    return 2;
  }
  int32_t n_cells = 0;
  for (const auto& c : cells) {
    if (c.enabled) ++n_cells;
  }
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int32_t d = c.dim;
    std::vector<float> ga(static_cast<std::size_t>(d));
    auto est = cell->rmsnorm_op.gamma.export_f32(ga.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    if (!write_exact(&d, 1) || !write_exact(ga.data(), ga.size())) return 2;
  }
  const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
  if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
    return 2;
  }
  return 0;
}

// WLB1 — LayerNorm NormalBP leaf
// stdin: batch,seq,dim,eps,dtype,backend ; gamma ; beta ; x ; use_target ; grad_or_target ; lr
// stdout WLBO: pre ; post ; dX ; dW(dγ|dβ) ; gamma_after ; beta_after ; loss
int run_layernorm_normalbp(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, dtype = 1, backend = 0;
  double eps = 0.0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) || !read_exact(&eps, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WLB1 header\n");
    return 2;
  }
  std::vector<float> gamma(static_cast<std::size_t>(dim));
  std::vector<float> beta(static_cast<std::size_t>(dim));
  if (!read_exact(gamma.data(), gamma.size()) || !read_exact(beta.data(), beta.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch) * static_cast<std::size_t>(seq) *
                       static_cast<std::size_t>(dim));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(x.size());
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  std::vector<float> pre(x.size()), post(x.size()), dX(x.size()),
      dW(static_cast<std::size_t>(2 * dim)), gamma_after(static_cast<std::size_t>(dim)),
      beta_after(static_cast<std::size_t>(dim));
  double loss = 0.0;

  auto once = [&]() -> welvet_status {
    return welvet_layernorm_normalbp_f32(batch, seq, dim, eps, dtype, backend, gamma.data(),
                                         beta.data(), x.data(), gyt.data(), use_target, lr,
                                         pre.data(), post.data(), dX.data(), dW.data(),
                                         gamma_after.data(), beta_after.data(), &loss);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = once();
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
    auto st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = static_cast<int32_t>(dW.size());
  const int32_t g_n = static_cast<int32_t>(gamma_after.size());
  const int32_t b_n = static_cast<int32_t>(beta_after.size());
  if (!write_magic("WLBO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(dW.data(), dW.size()) || !write_exact(&g_n, 1) ||
      !write_exact(gamma_after.data(), gamma_after.size()) || !write_exact(&b_n, 1) ||
      !write_exact(beta_after.data(), beta_after.size()) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

// WLV1 — Volume LayerNorm NormalBP
// stdin: depth,rows,cols,lpc,batch,backend,n_links ; links ;
//        per hop: enabled[,dim,dtype,eps,gamma,beta] ; x ; target ; lr
// stdout: optional WVB1; WLVO: loss ; gin ; n_cells ; per cell dim+γ+β after ; post
int run_volume_layernorm_normalbp(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WLV1 header\n");
    return 2;
  }
  if (batch <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad batch\n");
    return 2;
  }
  (void)backend;

  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int dim = 0;
    int dtype = 1;
    double eps = 0.0;
    welvet::architecture::Coord coord{};
    std::vector<float> gamma;
    std::vector<float> beta;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_dim = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    int32_t dim = 0, dtype = 1;
    double eps = 0.0;
    if (!read_exact(&dim, 1) || !read_exact(&dtype, 1) || !read_exact(&eps, 1)) return 2;
    c.dim = dim;
    c.dtype = dtype;
    c.eps = eps;
    c.gamma.resize(static_cast<std::size_t>(dim));
    c.beta.resize(static_cast<std::size_t>(dim));
    if (!read_exact(c.gamma.data(), c.gamma.size()) || !read_exact(c.beta.data(), c.beta.size())) {
      return 2;
    }
    if (first_dim < 0) first_dim = dim;
    cells.push_back(std::move(c));
  }
  if (first_dim < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_dim};
  input.data.resize(static_cast<std::size_t>(batch * first_dim));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_dim = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (it->enabled) {
      final_dim = it->dim;
      break;
    }
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_dim};
  target.data.resize(static_cast<std::size_t>(batch * final_dim));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::layernorm::Config cfg;
      cfg.dim = c.dim;
      cfg.eps = c.eps;
      welvet::layernorm::Layer layer;
      auto st = welvet::layernorm::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                                  c.gamma.data(), c.beta.data(), &layer);
      if (!st) return st;
      layer.backend = welvet::Backend::CPUTiled;
      st = welvet::architecture::bind_layernorm(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                                std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) {
      return st;
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
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
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  if (!write_magic("WLVO") || !write_exact(&loss, 1)) return 2;
  const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
  if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
    return 2;
  }
  int32_t n_cells = 0;
  for (const auto& c : cells) {
    if (c.enabled) ++n_cells;
  }
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int32_t d = c.dim;
    std::vector<float> ga(static_cast<std::size_t>(d));
    std::vector<float> ba(static_cast<std::size_t>(d));
    auto eg = cell->layernorm_op.gamma.export_f32(ga.data());
    if (!eg) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", eg.message.c_str());
      return 1;
    }
    auto eb = cell->layernorm_op.beta.export_f32(ba.data());
    if (!eb) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", eb.message.c_str());
      return 1;
    }
    if (!write_exact(&d, 1) || !write_exact(ga.data(), ga.size()) ||
        !write_exact(ba.data(), ba.size())) {
      return 2;
    }
  }
  const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
  if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
    return 2;
  }
  return 0;
}

// WSB1 — SwiGLU NormalBP leaf
// stdin: batch,seq,in,inter,dtype,backend ; Gate ; Up ; Down ; x ; use_target ; gyt ; lr
// stdout WSBO: pre ; post ; dX ; dW ; W_after ; loss
int run_swiglu_normalbp(int bench_iters) {
  int32_t batch = 0, seq = 0, in = 0, inter = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&in, 1) || !read_exact(&inter, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WSB1 header\n");
    return 2;
  }
  std::vector<float> Gate(static_cast<std::size_t>(inter * in));
  std::vector<float> Up(static_cast<std::size_t>(inter * in));
  std::vector<float> Down(static_cast<std::size_t>(in * inter));
  if (!read_exact(Gate.data(), Gate.size()) || !read_exact(Up.data(), Up.size()) ||
      !read_exact(Down.data(), Down.size())) {
    return 2;
  }
  std::vector<float> x(static_cast<std::size_t>(batch) * static_cast<std::size_t>(seq) *
                       static_cast<std::size_t>(in));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(x.size());
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  const int nw = 2 * inter * in + in * inter;
  std::vector<float> pre(static_cast<std::size_t>(batch) * static_cast<std::size_t>(seq) *
                         static_cast<std::size_t>(inter));
  std::vector<float> post(x.size()), dX(x.size()), dW(static_cast<std::size_t>(nw)),
      W_after(static_cast<std::size_t>(nw));
  double loss = 0.0;

  auto once = [&]() -> welvet_status {
    return welvet_swiglu_normalbp_f32(batch, seq, in, inter, dtype, backend, Gate.data(), Up.data(),
                                      Down.data(), x.data(), gyt.data(), use_target, lr, pre.data(),
                                      post.data(), dX.data(), dW.data(), W_after.data(), &loss);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = once();
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
    auto st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = static_cast<int32_t>(dW.size());
  const int32_t w_n = static_cast<int32_t>(W_after.size());
  if (!write_magic("WSBO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(dW.data(), dW.size()) || !write_exact(&w_n, 1) ||
      !write_exact(W_after.data(), W_after.size()) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

// WSV1 — Volume SwiGLU NormalBP
int run_volume_swiglu_normalbp(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WSV1 header\n");
    return 2;
  }
  if (batch <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad batch\n");
    return 2;
  }
  (void)backend;

  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int in = 0, inter = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> Gate, Up, Down;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_in = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    int32_t in = 0, inter = 0, dtype = 1;
    if (!read_exact(&in, 1) || !read_exact(&inter, 1) || !read_exact(&dtype, 1)) return 2;
    c.in = in;
    c.inter = inter;
    c.dtype = dtype;
    c.Gate.resize(static_cast<std::size_t>(inter * in));
    c.Up.resize(static_cast<std::size_t>(inter * in));
    c.Down.resize(static_cast<std::size_t>(in * inter));
    if (!read_exact(c.Gate.data(), c.Gate.size()) || !read_exact(c.Up.data(), c.Up.size()) ||
        !read_exact(c.Down.data(), c.Down.size())) {
      return 2;
    }
    if (first_in < 0) first_in = in;
    cells.push_back(std::move(c));
  }
  if (first_in < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_in};
  input.data.resize(static_cast<std::size_t>(batch * first_in));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_in = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (it->enabled) {
      final_in = it->in;
      break;
    }
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_in};
  target.data.resize(static_cast<std::size_t>(batch * final_in));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::swiglu::Config cfg;
      cfg.input_dim = c.in;
      cfg.intermediate_dim = c.inter;
      welvet::swiglu::Layer layer;
      auto st = welvet::swiglu::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                               c.Gate.data(), c.Up.data(), c.Down.data(), &layer);
      if (!st) return st;
      layer.backend = welvet::Backend::CPUTiled;
      st = welvet::architecture::bind_swiglu(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                             std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) {
      return st;
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
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
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  if (!write_magic("WSVO") || !write_exact(&loss, 1)) return 2;
  const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
  if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
    return 2;
  }
  int32_t n_cells = 0;
  for (const auto& c : cells) {
    if (c.enabled) ++n_cells;
  }
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int32_t nw = 2 * c.inter * c.in + c.in * c.inter;
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::swiglu::export_weights_f32(cell->swiglu_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    if (!write_exact(&nw, 1) || !write_exact(Wa.data(), Wa.size())) return 2;
  }
  const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
  if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
    return 2;
  }
  return 0;
}

// WXB1 — Softmax NormalBP leaf (weightless dX)
// stdin: same header as WSM1 + x + use_target + gyt + lr
// stdout WXBO: pre ; post ; dX ; dW(0) ; W_after(0) ; loss
int run_softmax_normalbp(int bench_iters) {
  int32_t n = 0, dim = 0, seq = 0, kind = 0, rows = 0, cols = 0, gumbel_seed = 0, dtype = 1,
          backend = 0, rank = 0, mask_len = 0, hier_len = 0;
  double temperature = 1.0, entmax_alpha = 0.0;
  if (!read_exact(&n, 1) || !read_exact(&dim, 1) || !read_exact(&seq, 1) || !read_exact(&kind, 1) ||
      !read_exact(&rows, 1) || !read_exact(&cols, 1) || !read_exact(&temperature, 1) ||
      !read_exact(&entmax_alpha, 1) || !read_exact(&gumbel_seed, 1) || !read_exact(&dtype, 1) ||
      !read_exact(&backend, 1) || !read_exact(&rank, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WXB1 header\n");
    return 2;
  }
  if (rank < 1 || rank > 8) {
    std::fprintf(stderr, "welvet_dense_fwd: bad softmax nbp rank\n");
    return 2;
  }
  std::vector<int32_t> shape(static_cast<std::size_t>(rank));
  if (!read_exact(shape.data(), shape.size())) return 2;
  if (!read_exact(&mask_len, 1)) return 2;
  std::vector<uint8_t> mask(static_cast<std::size_t>(mask_len > 0 ? mask_len : 0));
  if (mask_len > 0 && !read_exact(mask.data(), mask.size())) return 2;
  if (!read_exact(&hier_len, 1)) return 2;
  std::vector<int32_t> hier(static_cast<std::size_t>(hier_len > 0 ? hier_len : 0));
  if (hier_len > 0 && !read_exact(hier.data(), hier.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(n));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(static_cast<std::size_t>(n));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  std::vector<float> pre(static_cast<std::size_t>(n));
  std::vector<float> post(pre.size()), dX(pre.size());
  double loss = 0.0;

  auto once = [&]() -> welvet_status {
    return welvet_softmax_normalbp_f32(
        n, dim, seq, kind, rows, cols, temperature, entmax_alpha, gumbel_seed, dtype, backend, rank,
        shape.data(), mask_len > 0 ? mask.data() : nullptr, mask_len,
        hier_len > 0 ? hier.data() : nullptr, hier_len, x.data(), gyt.data(), use_target, lr,
        pre.data(), post.data(), dX.data(), &loss);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = once();
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
    auto st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = 0;
  const int32_t w_n = 0;
  if (!write_magic("WXBO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(&w_n, 1) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

// WXV1 — Volume Softmax NormalBP
int run_volume_softmax_normalbp(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WXV1 header\n");
    return 2;
  }
  if (batch <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad batch\n");
    return 2;
  }
  (void)backend;

  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int dim = 0, seq_len = 1, kind = 0, rows = 0, cols = 0, gumbel_seed = 0, dtype = 1;
    double temperature = 1.0, entmax_alpha = 0.0;
    std::vector<uint8_t> mask;
    std::vector<int32_t> hier;
    welvet::architecture::Coord coord{};
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_dim = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    int32_t dim = 0, seq = 0, kind = 0, crow = 0, ccol = 0, gumbel_seed = 0, dtype = 1;
    double temperature = 1.0, entmax_alpha = 0.0;
    int32_t mask_len = 0, hier_len = 0;
    if (!read_exact(&dim, 1) || !read_exact(&seq, 1) || !read_exact(&kind, 1) ||
        !read_exact(&crow, 1) || !read_exact(&ccol, 1) || !read_exact(&temperature, 1) ||
        !read_exact(&entmax_alpha, 1) || !read_exact(&gumbel_seed, 1) || !read_exact(&dtype, 1) ||
        !read_exact(&mask_len, 1)) {
      return 2;
    }
    c.dim = dim;
    c.seq_len = seq;
    c.kind = kind;
    c.rows = crow;
    c.cols = ccol;
    c.temperature = temperature;
    c.entmax_alpha = entmax_alpha;
    c.gumbel_seed = gumbel_seed;
    c.dtype = dtype;
    c.mask.resize(static_cast<std::size_t>(mask_len > 0 ? mask_len : 0));
    if (mask_len > 0 && !read_exact(c.mask.data(), c.mask.size())) return 2;
    if (!read_exact(&hier_len, 1)) return 2;
    c.hier.resize(static_cast<std::size_t>(hier_len > 0 ? hier_len : 0));
    if (hier_len > 0 && !read_exact(c.hier.data(), c.hier.size())) return 2;
    if (first_dim < 0) first_dim = dim;
    cells.push_back(std::move(c));
  }
  if (first_dim < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_dim};
  input.data.resize(static_cast<std::size_t>(batch * first_dim));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_dim = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (it->enabled) {
      final_dim = it->dim;
      break;
    }
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_dim};
  target.data.resize(static_cast<std::size_t>(batch * final_dim));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::softmax::Config cfg;
      cfg.dim = c.dim;
      cfg.seq_len = c.seq_len <= 0 ? 1 : c.seq_len;
      cfg.rows = c.rows;
      cfg.cols = c.cols;
      cfg.temperature = c.temperature;
      cfg.kind = static_cast<welvet::softmax::Kind>(c.kind);
      cfg.entmax_alpha = c.entmax_alpha;
      cfg.gumbel_seed = static_cast<uint32_t>(c.gumbel_seed);
      if (!c.mask.empty()) {
        cfg.mask.resize(c.mask.size());
        for (std::size_t i = 0; i < c.mask.size(); ++i) {
          cfg.mask[i] = c.mask[i] != 0;
        }
      }
      if (!c.hier.empty()) {
        cfg.hierarchy_levels.assign(c.hier.begin(), c.hier.end());
      }
      welvet::softmax::Layer layer;
      auto st = welvet::softmax::new_configured(cfg, static_cast<welvet::DType>(c.dtype), &layer);
      if (!st) return st;
      layer.backend = welvet::Backend::CPUTiled;
      st = welvet::architecture::bind_softmax(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                              std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) {
      return st;
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
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
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  if (!write_magic("WXVO") || !write_exact(&loss, 1)) return 2;
  const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
  if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
    return 2;
  }
  int32_t n_cells = 0;
  for (const auto& c : cells) {
    if (c.enabled) ++n_cells;
  }
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    const int32_t nw = 0; // weightless
    if (!write_exact(&nw, 1)) return 2;
  }
  const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
  if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
    return 2;
  }
  return 0;
}

// WEB1 — Embedding NormalBP leaf
// stdin: batch,vocab,emb,seq,in_rank,in_last,act,dtype,backend ; W ; x ; use_target ; gyt ; lr
// stdout WEBO: pre ; post ; dX ; dW ; W_after ; loss
int run_embedding_normalbp(int bench_iters) {
  int32_t batch = 0, vocab = 0, emb = 0, seq = 0, in_rank = 2, in_last = 1, act = -1, dtype = 1,
          backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&vocab, 1) || !read_exact(&emb, 1) ||
      !read_exact(&seq, 1) || !read_exact(&in_rank, 1) || !read_exact(&in_last, 1) ||
      !read_exact(&act, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WEB1 header\n");
    return 2;
  }
  std::vector<float> W(static_cast<std::size_t>(vocab * emb));
  if (!read_exact(W.data(), W.size())) return 2;
  const int x_n = batch * seq * in_last;
  std::vector<float> x(static_cast<std::size_t>(x_n));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  const int out_n = (in_rank == 1 ? seq : batch * seq) * emb;
  std::vector<float> gyt(static_cast<std::size_t>(out_n));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  std::vector<float> pre(static_cast<std::size_t>(out_n));
  std::vector<float> post(pre.size()), dX(static_cast<std::size_t>(x_n)),
      dW(static_cast<std::size_t>(vocab * emb)), W_after(dW.size());
  double loss = 0.0;

  auto once = [&]() -> welvet_status {
    return welvet_embedding_normalbp_f32(batch, vocab, emb, seq, in_rank, in_last, act, dtype,
                                         backend, W.data(), x.data(), gyt.data(), use_target, lr,
                                         pre.data(), post.data(), dX.data(), dW.data(),
                                         W_after.data(), &loss);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = once();
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
    auto st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = static_cast<int32_t>(dW.size());
  const int32_t w_n = static_cast<int32_t>(W_after.size());
  if (!write_magic("WEBO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(dW.data(), dW.size()) || !write_exact(&w_n, 1) ||
      !write_exact(W_after.data(), W_after.size()) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

// WEV1 — Volume Embedding NormalBP
int run_volume_embedding_normalbp(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WEV1 header\n");
    return 2;
  }
  if (batch <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad batch\n");
    return 2;
  }
  (void)backend;

  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int vocab = 0, emb = 0, seq = 0, act = -1, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_seq = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    int32_t vocab = 0, emb = 0, seq = 0, act = -1, dtype = 1;
    if (!read_exact(&vocab, 1) || !read_exact(&emb, 1) || !read_exact(&seq, 1) ||
        !read_exact(&act, 1) || !read_exact(&dtype, 1)) {
      return 2;
    }
    c.vocab = vocab;
    c.emb = emb;
    c.seq = seq;
    c.act = act;
    c.dtype = dtype;
    c.W.resize(static_cast<std::size_t>(vocab * emb));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_seq < 0) first_seq = seq;
    cells.push_back(std::move(c));
  }
  if (first_seq < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_seq};
  input.data.resize(static_cast<std::size_t>(batch * first_seq));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_emb = 0, final_seq = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (it->enabled) {
      final_emb = it->emb;
      final_seq = it->seq;
      break;
    }
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_seq, final_emb};
  target.data.resize(static_cast<std::size_t>(batch * final_seq * final_emb));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::embedding::Config cfg;
      cfg.vocab_size = c.vocab;
      cfg.embedding_dim = c.emb;
      cfg.seq_len = c.seq;
      welvet::embedding::Layer layer;
      auto st = welvet::embedding::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                                  static_cast<welvet::Activation>(c.act), c.W.data(),
                                                  &layer);
      if (!st) return st;
      layer.backend = welvet::Backend::CPUTiled;
      st = welvet::architecture::bind_embedding(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                                std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) {
      return st;
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
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
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  if (!write_magic("WEVO") || !write_exact(&loss, 1)) return 2;
  const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
  if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
    return 2;
  }
  int32_t n_cells = 0;
  for (const auto& c : cells) {
    if (c.enabled) ++n_cells;
  }
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int32_t nw = c.vocab * c.emb;
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = cell->embedding_op.weights.export_f32(Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    if (!write_exact(&nw, 1) || !write_exact(Wa.data(), Wa.size())) return 2;
  }
  const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
  if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
    return 2;
  }
  return 0;
}

// WHB1 — MHA NormalBP leaf (ModeSelf)
// stdin: same header as WVM1 + x + use_target + gyt + lr
// stdout WHBO: pre ; post ; dX ; dW ; W_after ; loss
int run_mha_normalbp(int bench_iters) {
  int32_t batch = 0, seq = 0, d_model = 0, num_heads = 0, num_kv = 0, head_dim = 0, msl = 0;
  int32_t mask = 0, pos = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&d_model, 1) ||
      !read_exact(&num_heads, 1) || !read_exact(&num_kv, 1) || !read_exact(&head_dim, 1) ||
      !read_exact(&msl, 1) || !read_exact(&mask, 1) || !read_exact(&pos, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WHB1 header\n");
    return 2;
  }
  if (num_heads <= 0 || d_model <= 0 || d_model % num_heads != 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad mha dims\n");
    return 2;
  }
  const int hd = head_dim > 0 ? head_dim : d_model / num_heads;
  const int kv_heads = num_kv > 0 ? num_kv : num_heads;
  const int q_dim = num_heads * hd;
  const int kv_dim = kv_heads * hd;
  const int nw = q_dim * d_model + kv_dim * d_model + kv_dim * d_model + d_model * q_dim;

  std::vector<float> Q(static_cast<std::size_t>(q_dim * d_model));
  std::vector<float> K(static_cast<std::size_t>(kv_dim * d_model));
  std::vector<float> V(static_cast<std::size_t>(kv_dim * d_model));
  std::vector<float> O(static_cast<std::size_t>(d_model * q_dim));
  if (!read_exact(Q.data(), Q.size()) || !read_exact(K.data(), K.size()) ||
      !read_exact(V.data(), V.size()) || !read_exact(O.data(), O.size())) {
    return 2;
  }
  std::vector<float> x(static_cast<std::size_t>(batch * seq * d_model));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(x.size());
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  std::vector<float> pre(static_cast<std::size_t>(batch * seq * q_dim));
  std::vector<float> post(x.size()), dX(x.size()), dW(static_cast<std::size_t>(nw)),
      W_after(static_cast<std::size_t>(nw));
  double loss = 0.0;

  auto once = [&]() -> welvet_status {
    return welvet_mha_normalbp_f32(batch, seq, d_model, num_heads, num_kv, head_dim, msl, mask, pos,
                                   dtype, backend, Q.data(), K.data(), V.data(), O.data(), x.data(),
                                   gyt.data(), use_target, lr, pre.data(), post.data(), dX.data(),
                                   dW.data(), W_after.data(), &loss);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = once();
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
    auto st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = static_cast<int32_t>(dW.size());
  const int32_t w_n = static_cast<int32_t>(W_after.size());
  if (!write_magic("WHBO") || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(dW.data(), dW.size()) || !write_exact(&w_n, 1) ||
      !write_exact(W_after.data(), W_after.size()) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

// WHV1 — Volume MHA NormalBP
int run_volume_mha_normalbp(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WHV1 header\n");
    return 2;
  }
  if (batch <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad batch\n");
    return 2;
  }
  (void)backend;

  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int seq = 0, d_model = 0, num_heads = 0, num_kv = 0, head_dim = 0, msl = 0;
    int mask = 0, pos = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> Q, K, V, O;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_seq = -1, first_d = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    int32_t seq = 0, d_model = 0, num_heads = 0, num_kv = 0, head_dim = 0, msl = 0;
    int32_t mask = 0, pos = 0, dtype = 1;
    if (!read_exact(&seq, 1) || !read_exact(&d_model, 1) || !read_exact(&num_heads, 1) ||
        !read_exact(&num_kv, 1) || !read_exact(&head_dim, 1) || !read_exact(&msl, 1) ||
        !read_exact(&mask, 1) || !read_exact(&pos, 1) || !read_exact(&dtype, 1)) {
      return 2;
    }
    c.seq = seq;
    c.d_model = d_model;
    c.num_heads = num_heads;
    c.num_kv = num_kv;
    c.head_dim = head_dim;
    c.msl = msl;
    c.mask = mask;
    c.pos = pos;
    c.dtype = dtype;
    const int hd = head_dim > 0 ? head_dim : d_model / num_heads;
    const int kv_heads = num_kv > 0 ? num_kv : num_heads;
    const int q_dim = num_heads * hd;
    const int kv_dim = kv_heads * hd;
    c.Q.resize(static_cast<std::size_t>(q_dim * d_model));
    c.K.resize(static_cast<std::size_t>(kv_dim * d_model));
    c.V.resize(static_cast<std::size_t>(kv_dim * d_model));
    c.O.resize(static_cast<std::size_t>(d_model * q_dim));
    if (!read_exact(c.Q.data(), c.Q.size()) || !read_exact(c.K.data(), c.K.size()) ||
        !read_exact(c.V.data(), c.V.size()) || !read_exact(c.O.data(), c.O.size())) {
      return 2;
    }
    if (first_seq < 0) {
      first_seq = seq;
      first_d = d_model;
    }
    cells.push_back(std::move(c));
  }
  if (first_seq < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_seq, first_d};
  input.data.resize(static_cast<std::size_t>(batch * first_seq * first_d));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_seq = 0, final_d = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (it->enabled) {
      final_seq = it->seq;
      final_d = it->d_model;
      break;
    }
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_seq, final_d};
  target.data.resize(static_cast<std::size_t>(batch * final_seq * final_d));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::mha::Config cfg;
      cfg.d_model = c.d_model;
      cfg.num_heads = c.num_heads;
      cfg.num_kv_heads = c.num_kv;
      cfg.head_dim = c.head_dim;
      cfg.max_seq_len = c.msl;
      cfg.mask = static_cast<welvet::mha::MaskKind>(c.mask);
      cfg.pos = static_cast<welvet::mha::PosKind>(c.pos);
      cfg.mode = welvet::mha::AttnMode::Self;
      cfg.causal = (cfg.mask == welvet::mha::MaskKind::Causal);
      welvet::mha::Layer layer;
      auto st = welvet::mha::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.Q.data(),
                                            c.K.data(), c.V.data(), c.O.data(), &layer);
      if (!st) return st;
      layer.backend = welvet::Backend::CPUTiled;
      st = welvet::architecture::bind_mha(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                          std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) {
      return st;
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
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
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  if (!write_magic("WHVO") || !write_exact(&loss, 1)) return 2;
  const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
  if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
    return 2;
  }
  int32_t n_cells = 0;
  for (const auto& c : cells) {
    if (c.enabled) ++n_cells;
  }
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int32_t nw = welvet::mha::grad_w_size(cell->mha_op);
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::mha::export_weights_f32(cell->mha_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    if (!write_exact(&nw, 1) || !write_exact(Wa.data(), Wa.size())) return 2;
  }
  const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
  if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
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
  if (std::memcmp(magic, "WVM1", 4) == 0) {
    return run_mha(bench_iters);
  }
  if (std::memcmp(magic, "WVS1", 4) == 0) {
    return run_swiglu(bench_iters);
  }
  if (std::memcmp(magic, "WVR1", 4) == 0) {
    return run_rmsnorm(bench_iters);
  }
  if (std::memcmp(magic, "WVL1", 4) == 0) {
    return run_layernorm(bench_iters);
  }
  if (std::memcmp(magic, "WVC1", 4) == 0) {
    return run_cnn1(bench_iters);
  }
  if (std::memcmp(magic, "WVC2", 4) == 0) {
    return run_cnn2(bench_iters);
  }
  if (std::memcmp(magic, "WVC3", 4) == 0) {
    return run_cnn3(bench_iters);
  }
  if (std::memcmp(magic, "WVT1", 4) == 0) {
    return run_convt1(bench_iters);
  }
  if (std::memcmp(magic, "WVT2", 4) == 0) {
    return run_convt2(bench_iters);
  }
  if (std::memcmp(magic, "WVT3", 4) == 0) {
    return run_convt3(bench_iters);
  }
  if (std::memcmp(magic, "WRN1", 4) == 0) {
    return run_rnn(bench_iters);
  }
  if (std::memcmp(magic, "WLS1", 4) == 0) {
    return run_lstm(bench_iters);
  }
  if (std::memcmp(magic, "WEM1", 4) == 0) {
    return run_embedding(bench_iters);
  }
  if (std::memcmp(magic, "WSM1", 4) == 0) {
    return run_softmax(bench_iters);
  }
  if (std::memcmp(magic, "WKM1", 4) == 0) {
    return run_kmeans(bench_iters);
  }
  if (std::memcmp(magic, "WSQ1", 4) == 0) {
    return run_sequential(bench_iters);
  }
  if (std::memcmp(magic, "WRE1", 4) == 0) {
    return run_residual(bench_iters);
  }
  if (std::memcmp(magic, "WMB1", 4) == 0) {
    return run_mamba(bench_iters);
  }
  if (std::memcmp(magic, "WMC1", 4) == 0) {
    return run_metacognition(bench_iters);
  }
  if (std::memcmp(magic, "WGD1", 4) == 0) {
    return run_gdn(bench_iters);
  }
  if (std::memcmp(magic, "WSX1", 4) == 0) {
    return run_seqmix(bench_iters);
  }
  if (std::memcmp(magic, "WPL1", 4) == 0) {
    return run_parallel(bench_iters);
  }
  if (std::memcmp(magic, "WDB1", 4) == 0) {
    return run_dense_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WVN1", 4) == 0) {
    return run_volume_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WRB1", 4) == 0) {
    return run_rmsnorm_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WRV1", 4) == 0) {
    return run_volume_rmsnorm_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WLB1", 4) == 0) {
    return run_layernorm_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WLV1", 4) == 0) {
    return run_volume_layernorm_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WSB1", 4) == 0) {
    return run_swiglu_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WSV1", 4) == 0) {
    return run_volume_swiglu_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WXB1", 4) == 0) {
    return run_softmax_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WXV1", 4) == 0) {
    return run_volume_softmax_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WEB1", 4) == 0) {
    return run_embedding_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WEV1", 4) == 0) {
    return run_volume_embedding_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WHB1", 4) == 0) {
    return run_mha_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WHV1", 4) == 0) {
    return run_volume_mha_normalbp(bench_iters);
  }
  const bool v3 = std::memcmp(magic, "WVD3", 4) == 0;
  const bool v2 = std::memcmp(magic, "WVD2", 4) == 0;
  const bool v1 = std::memcmp(magic, "WVD1", 4) == 0;
  if (!v1 && !v2 && !v3) {
    std::fprintf(stderr, "welvet_dense_fwd: bad magic (want WVD*/WVA1/WVF1/WVM1/WVS1/WVR1/WVL1/"
                         "WVC1/2/3/WVT1/2/3/WRN1/WLS1/WEM1/WSM1/WKM1/WSQ1/WRE1/WMB1/WMC1/WGD1/WSX1/"
                         "WPL1/WDB1/WVN1/WRB1/WRV1/WLB1/WLV1/WSB1/WSV1/WXB1/WXV1/WEB1/WEV1/"
                         "WHB1/WHV1)\n");
    return 2;
  }
  return run_dense(v1, v2, v3, bench_iters);
}
