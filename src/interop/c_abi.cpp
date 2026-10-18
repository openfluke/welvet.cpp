#include "welvet_c.h"

#include "welvet/architecture.hpp"
#include "welvet/dense.hpp"
#include "welvet/gaps.hpp"

#include <cstdio>
#include <cstring>
#include <iostream>

namespace {

welvet_status to_c(const welvet::Status& st) {
  welvet_status out{};
  out.code = static_cast<int32_t>(st.code);
  std::snprintf(out.message, sizeof(out.message), "%s", st.message.c_str());
  return out;
}

} // namespace

extern "C" {

welvet_status welvet_dense_forward_f32(int32_t batch, int32_t in, int32_t out, int32_t act,
                                       int32_t dtype, int32_t backend, const float* W_init,
                                       const double* bias, size_t bias_len, const float* x,
                                       float* pre, float* post) {
  if (batch <= 0 || in <= 0 || out <= 0 || !W_init || !x || !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "invalid argument");
    return st;
  }
  if (dtype < 0 || dtype >= welvet::kNumDTypes) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "bad dtype %d", dtype);
    return st;
  }
  if (backend != 0 && backend != 1) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "bad backend %d (0=cpu,1=simd)", backend);
    return st;
  }

  welvet::dense::Layer layer;
  auto st = welvet::dense::new_layer(in, out, static_cast<welvet::Activation>(act),
                                     static_cast<welvet::DType>(dtype), W_init, &layer);
  if (!st) {
    return to_c(st);
  }
  layer.backend = backend == 1 ? welvet::Backend::SIMD : welvet::Backend::CPUTiled;
  if (bias && bias_len > 0) {
    layer.weights.bias.assign(bias, bias + bias_len);
  }

  welvet::TensorF32 input;
  input.shape = {batch, in};
  input.data.assign(x, x + static_cast<std::size_t>(batch * in));

  welvet::TensorF32 pre_t, post_t;
  st = welvet::dense::forward(layer, input, &pre_t, &post_t);
  if (!st) {
    return to_c(st);
  }
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_dense_forward_f32_cpu(int32_t batch, int32_t in, int32_t out, int32_t act,
                                           int32_t dtype, const float* W_init, const double* bias,
                                           size_t bias_len, const float* x, float* pre, float* post) {
  return welvet_dense_forward_f32(batch, in, out, act, dtype, /*cpu*/ 0, W_init, bias, bias_len, x,
                                  pre, post);
}

welvet_status welvet_dense_forward_f32_weights(int32_t batch, int32_t in, int32_t out, int32_t act,
                                               const float* W, const double* bias, size_t bias_len,
                                               const float* x, float* pre, float* post) {
  return welvet_dense_forward_f32(batch, in, out, act, /*Float32*/ 1, /*cpu*/ 0, W, bias, bias_len,
                                  x, pre, post);
}

welvet_status welvet_arch_topology(int32_t depth, int32_t rows, int32_t cols,
                                   int32_t layers_per_cell, const int32_t* links, int32_t n_links,
                                   int32_t* out_n_cells, int32_t* out_coords, int32_t* out_indices,
                                   int32_t* out_remote) {
  if (!out_n_cells || !out_coords || !out_indices || !out_remote) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "nil output buffers");
    return st;
  }
  if (n_links < 0 || (n_links > 0 && !links)) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "bad links");
    return st;
  }

  auto g = welvet::architecture::new_grid(depth, rows, cols, layers_per_cell);
  for (int32_t i = 0; i < n_links; ++i) {
    const int32_t* L = links + static_cast<std::size_t>(i) * 8;
    auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5], L[6],
                                                    L[7]);
    if (!st) {
      return to_c(st);
    }
  }

  const auto hops = welvet::architecture::hop_order(g);
  *out_n_cells = static_cast<int32_t>(hops.size());
  for (std::size_t i = 0; i < hops.size(); ++i) {
    const auto& c = hops[i];
    out_coords[i * 4 + 0] = c.z;
    out_coords[i * 4 + 1] = c.y;
    out_coords[i * 4 + 2] = c.x;
    out_coords[i * 4 + 3] = c.l;
    out_indices[i] = welvet::architecture::index(g, c.z, c.y, c.x, c.l);
    const welvet::architecture::Cell* cell = welvet::architecture::at(g, c.z, c.y, c.x, c.l);
    out_remote[i * 5 + 0] = cell && cell->is_remote_link ? 1 : 0;
    out_remote[i * 5 + 1] = cell ? cell->target_z : 0;
    out_remote[i * 5 + 2] = cell ? cell->target_y : 0;
    out_remote[i * 5 + 3] = cell ? cell->target_x : 0;
    out_remote[i * 5 + 4] = cell ? cell->target_l : 0;
  }
  return to_c(welvet::Status::Ok());
}

const char* welvet_version(void) { return "0.1.3-m1-arch-baseline"; }

int32_t welvet_gap_count(void) { return static_cast<int32_t>(welvet::gaps::count_gaps()); }

void welvet_print_gaps(void) { welvet::gaps::print_census(std::cout); }

} // extern "C"
