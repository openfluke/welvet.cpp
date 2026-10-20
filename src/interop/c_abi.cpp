#include "welvet_c.h"

#include "welvet/architecture.hpp"
#include "welvet/cnn1.hpp"
#include "welvet/cnn2.hpp"
#include "welvet/cnn3.hpp"
#include "welvet/convt1.hpp"
#include "welvet/convt2.hpp"
#include "welvet/convt3.hpp"
#include "welvet/dense.hpp"
#include "welvet/gaps.hpp"
#include "welvet/layernorm.hpp"
#include "welvet/lstm.hpp"
#include "welvet/mha.hpp"
#include "welvet/rmsnorm.hpp"
#include "welvet/rnn.hpp"
#include "welvet/swiglu.hpp"

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

welvet_status welvet_mha_forward_f32(int32_t batch, int32_t seq, int32_t d_model, int32_t num_heads,
                                     int32_t num_kv_heads, int32_t head_dim, int32_t max_seq_len,
                                     int32_t mask, int32_t pos, int32_t dtype, int32_t backend,
                                     const float* Q, const float* K, const float* V, const float* O,
                                     const float* x, float* pre, float* post) {
  if (batch <= 0 || seq <= 0 || d_model <= 0 || num_heads <= 0 || !Q || !K || !V || !O || !x ||
      !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "mha: invalid argument");
    return st;
  }
  if (backend != 0 && backend != 1) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "mha: bad backend");
    return st;
  }

  welvet::mha::Config cfg;
  cfg.d_model = d_model;
  cfg.num_heads = num_heads;
  cfg.num_kv_heads = num_kv_heads;
  cfg.head_dim = head_dim;
  cfg.max_seq_len = max_seq_len;
  cfg.mask = static_cast<welvet::mha::MaskKind>(mask);
  cfg.pos = static_cast<welvet::mha::PosKind>(pos);
  cfg.mode = welvet::mha::AttnMode::Self;
  cfg.causal = (cfg.mask == welvet::mha::MaskKind::Causal);

  welvet::mha::Layer layer;
  auto st = welvet::mha::new_configured(cfg, static_cast<welvet::DType>(dtype), Q, K, V, O, &layer);
  if (!st) {
    return to_c(st);
  }
  layer.backend = backend == 1 ? welvet::Backend::SIMD : welvet::Backend::CPUTiled;

  welvet::TensorF32 input;
  input.shape = {batch, seq, d_model};
  input.data.assign(x, x + static_cast<std::size_t>(batch * seq * d_model));

  welvet::TensorF32 pre_t, post_t;
  st = welvet::mha::forward(layer, input, &pre_t, &post_t);
  if (!st) {
    return to_c(st);
  }
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_swiglu_forward_f32(int32_t batch, int32_t seq, int32_t input_dim,
                                        int32_t intermediate_dim, int32_t dtype, int32_t backend,
                                        const float* Gate, const float* Up, const float* Down,
                                        const float* x, float* pre, float* post) {
  if (batch <= 0 || seq <= 0 || input_dim <= 0 || intermediate_dim <= 0 || !Gate || !Up || !Down ||
      !x || !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "swiglu: invalid argument");
    return st;
  }
  if (backend != 0 && backend != 1) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "swiglu: bad backend");
    return st;
  }

  welvet::swiglu::Config cfg;
  cfg.input_dim = input_dim;
  cfg.intermediate_dim = intermediate_dim;

  welvet::swiglu::Layer layer;
  auto st = welvet::swiglu::new_configured(cfg, static_cast<welvet::DType>(dtype), Gate, Up, Down,
                                           &layer);
  if (!st) {
    return to_c(st);
  }
  layer.backend = backend == 1 ? welvet::Backend::SIMD : welvet::Backend::CPUTiled;

  welvet::TensorF32 input;
  if (seq == 1) {
    input.shape = {batch, input_dim};
  } else {
    input.shape = {batch, seq, input_dim};
  }
  input.data.assign(x, x + static_cast<std::size_t>(batch * seq * input_dim));

  welvet::TensorF32 pre_t, post_t;
  st = welvet::swiglu::forward(layer, input, &pre_t, &post_t);
  if (!st) {
    return to_c(st);
  }
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_rmsnorm_forward_f32(int32_t batch, int32_t seq, int32_t dim, double eps,
                                         int32_t dtype, int32_t backend, const float* gamma,
                                         const float* x, float* pre, float* post) {
  if (batch <= 0 || seq <= 0 || dim <= 0 || !gamma || !x || !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "rmsnorm: invalid argument");
    return st;
  }
  if (backend != 0 && backend != 1) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "rmsnorm: bad backend");
    return st;
  }

  welvet::rmsnorm::Config cfg;
  cfg.dim = dim;
  cfg.eps = eps;

  welvet::rmsnorm::Layer layer;
  auto st = welvet::rmsnorm::new_configured(cfg, static_cast<welvet::DType>(dtype), gamma, &layer);
  if (!st) {
    return to_c(st);
  }
  layer.backend = backend == 1 ? welvet::Backend::SIMD : welvet::Backend::CPUTiled;

  welvet::TensorF32 input;
  if (seq == 1) {
    input.shape = {batch, dim};
  } else {
    input.shape = {batch, seq, dim};
  }
  input.data.assign(x, x + static_cast<std::size_t>(batch * seq * dim));

  welvet::TensorF32 pre_t, post_t;
  st = welvet::rmsnorm::forward(layer, input, &pre_t, &post_t);
  if (!st) {
    return to_c(st);
  }
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_layernorm_forward_f32(int32_t batch, int32_t seq, int32_t dim, double eps,
                                           int32_t dtype, int32_t backend, const float* gamma,
                                           const float* beta, const float* x, float* pre,
                                           float* post) {
  if (batch <= 0 || seq <= 0 || dim <= 0 || !gamma || !beta || !x || !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "layernorm: invalid argument");
    return st;
  }
  if (backend != 0 && backend != 1) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "layernorm: bad backend");
    return st;
  }

  welvet::layernorm::Config cfg;
  cfg.dim = dim;
  cfg.eps = eps;

  welvet::layernorm::Layer layer;
  auto st =
      welvet::layernorm::new_configured(cfg, static_cast<welvet::DType>(dtype), gamma, beta, &layer);
  if (!st) {
    return to_c(st);
  }
  layer.backend = backend == 1 ? welvet::Backend::SIMD : welvet::Backend::CPUTiled;

  welvet::TensorF32 input;
  if (seq == 1) {
    input.shape = {batch, dim};
  } else {
    input.shape = {batch, seq, dim};
  }
  input.data.assign(x, x + static_cast<std::size_t>(batch * seq * dim));

  welvet::TensorF32 pre_t, post_t;
  st = welvet::layernorm::forward(layer, input, &pre_t, &post_t);
  if (!st) {
    return to_c(st);
  }
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_cnn1_forward_f32(int32_t batch, int32_t in_c, int32_t filters, int32_t seq_len,
                                      int32_t kernel, int32_t stride, int32_t padding, int32_t act,
                                      int32_t dtype, int32_t backend, const float* W,
                                      const double* bias, size_t bias_len, const float* x,
                                      float* pre, float* post) {
  if (batch <= 0 || in_c <= 0 || filters <= 0 || seq_len <= 0 || kernel <= 0 || !W || !x || !pre ||
      !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "cnn1: invalid argument");
    return st;
  }
  if (backend != 0 && backend != 1) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "cnn1: bad backend");
    return st;
  }

  welvet::cnn1::Config cfg;
  cfg.in_channels = in_c;
  cfg.filters = filters;
  cfg.seq_len = seq_len;
  cfg.kernel = kernel;
  cfg.stride = stride;
  cfg.padding = padding;
  cfg.activation = static_cast<welvet::Activation>(act);

  welvet::cnn1::Layer layer;
  auto st = welvet::cnn1::new_configured(cfg, static_cast<welvet::DType>(dtype), W, &layer);
  if (!st) return to_c(st);
  if (bias && bias_len > 0) {
    layer.proj.weights.bias.assign(bias, bias + bias_len);
  }
  layer.backend = backend == 1 ? welvet::Backend::SIMD : welvet::Backend::CPUTiled;

  welvet::TensorF32 input;
  input.shape = {batch, in_c, seq_len};
  input.data.assign(x, x + static_cast<std::size_t>(batch * in_c * seq_len));

  welvet::TensorF32 pre_t, post_t;
  st = welvet::cnn1::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_cnn2_forward_f32(int32_t batch, int32_t in_c, int32_t filters, int32_t height,
                                      int32_t width, int32_t kernel, int32_t stride, int32_t padding,
                                      int32_t act, int32_t dtype, int32_t backend, const float* W,
                                      const double* bias, size_t bias_len, const float* x,
                                      float* pre, float* post) {
  if (batch <= 0 || in_c <= 0 || filters <= 0 || height <= 0 || width <= 0 || kernel <= 0 || !W ||
      !x || !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "cnn2: invalid argument");
    return st;
  }
  if (backend != 0 && backend != 1) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "cnn2: bad backend");
    return st;
  }

  welvet::cnn2::Config cfg;
  cfg.in_channels = in_c;
  cfg.filters = filters;
  cfg.height = height;
  cfg.width = width;
  cfg.kernel = kernel;
  cfg.stride = stride;
  cfg.padding = padding;
  cfg.activation = static_cast<welvet::Activation>(act);

  welvet::cnn2::Layer layer;
  auto st = welvet::cnn2::new_configured(cfg, static_cast<welvet::DType>(dtype), W, &layer);
  if (!st) return to_c(st);
  if (bias && bias_len > 0) {
    layer.proj.weights.bias.assign(bias, bias + bias_len);
  }
  layer.backend = backend == 1 ? welvet::Backend::SIMD : welvet::Backend::CPUTiled;

  welvet::TensorF32 input;
  input.shape = {batch, in_c, height, width};
  input.data.assign(x, x + static_cast<std::size_t>(batch * in_c * height * width));

  welvet::TensorF32 pre_t, post_t;
  st = welvet::cnn2::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_cnn3_forward_f32(int32_t batch, int32_t in_c, int32_t filters, int32_t depth,
                                      int32_t height, int32_t width, int32_t kernel, int32_t stride,
                                      int32_t padding, int32_t act, int32_t dtype, int32_t backend,
                                      const float* W, const double* bias, size_t bias_len,
                                      const float* x, float* pre, float* post) {
  if (batch <= 0 || in_c <= 0 || filters <= 0 || depth <= 0 || height <= 0 || width <= 0 ||
      kernel <= 0 || !W || !x || !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "cnn3: invalid argument");
    return st;
  }
  if (backend != 0 && backend != 1) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "cnn3: bad backend");
    return st;
  }

  welvet::cnn3::Config cfg;
  cfg.in_channels = in_c;
  cfg.filters = filters;
  cfg.depth = depth;
  cfg.height = height;
  cfg.width = width;
  cfg.kernel = kernel;
  cfg.stride = stride;
  cfg.padding = padding;
  cfg.activation = static_cast<welvet::Activation>(act);

  welvet::cnn3::Layer layer;
  auto st = welvet::cnn3::new_configured(cfg, static_cast<welvet::DType>(dtype), W, &layer);
  if (!st) return to_c(st);
  if (bias && bias_len > 0) {
    layer.proj.weights.bias.assign(bias, bias + bias_len);
  }
  layer.backend = backend == 1 ? welvet::Backend::SIMD : welvet::Backend::CPUTiled;

  welvet::TensorF32 input;
  input.shape = {batch, in_c, depth, height, width};
  input.data.assign(x, x + static_cast<std::size_t>(batch * in_c * depth * height * width));

  welvet::TensorF32 pre_t, post_t;
  st = welvet::cnn3::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_convt1_forward_f32(int32_t batch, int32_t in_c, int32_t filters,
                                        int32_t seq_len, int32_t kernel, int32_t stride,
                                        int32_t padding, int32_t output_padding, int32_t act,
                                        int32_t dtype, int32_t backend, const float* W,
                                        const float* x, float* pre, float* post) {
  if (batch <= 0 || in_c <= 0 || filters <= 0 || seq_len <= 0 || kernel <= 0 || !W || !x || !pre ||
      !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "convt1: invalid argument");
    return st;
  }
  if (backend != 0 && backend != 1) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "convt1: bad backend");
    return st;
  }
  welvet::convt1::Config cfg;
  cfg.in_channels = in_c;
  cfg.filters = filters;
  cfg.seq_len = seq_len;
  cfg.kernel = kernel;
  cfg.stride = stride;
  cfg.padding = padding;
  cfg.output_padding = output_padding;
  cfg.activation = static_cast<welvet::Activation>(act);
  welvet::convt1::Layer layer;
  auto st = welvet::convt1::new_configured(cfg, static_cast<welvet::DType>(dtype), W, &layer);
  if (!st) return to_c(st);
  layer.backend = backend == 1 ? welvet::Backend::SIMD : welvet::Backend::CPUTiled;
  welvet::TensorF32 input;
  input.shape = {batch, in_c, seq_len};
  input.data.assign(x, x + static_cast<std::size_t>(batch * in_c * seq_len));
  welvet::TensorF32 pre_t, post_t;
  st = welvet::convt1::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_convt2_forward_f32(int32_t batch, int32_t in_c, int32_t filters,
                                        int32_t height, int32_t width, int32_t kernel,
                                        int32_t stride, int32_t padding, int32_t output_padding,
                                        int32_t act, int32_t dtype, int32_t backend, const float* W,
                                        const float* x, float* pre, float* post) {
  if (batch <= 0 || in_c <= 0 || filters <= 0 || height <= 0 || width <= 0 || kernel <= 0 || !W ||
      !x || !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "convt2: invalid argument");
    return st;
  }
  if (backend != 0 && backend != 1) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "convt2: bad backend");
    return st;
  }
  welvet::convt2::Config cfg;
  cfg.in_channels = in_c;
  cfg.filters = filters;
  cfg.height = height;
  cfg.width = width;
  cfg.kernel = kernel;
  cfg.stride = stride;
  cfg.padding = padding;
  cfg.output_padding = output_padding;
  cfg.activation = static_cast<welvet::Activation>(act);
  welvet::convt2::Layer layer;
  auto st = welvet::convt2::new_configured(cfg, static_cast<welvet::DType>(dtype), W, &layer);
  if (!st) return to_c(st);
  layer.backend = backend == 1 ? welvet::Backend::SIMD : welvet::Backend::CPUTiled;
  welvet::TensorF32 input;
  input.shape = {batch, in_c, height, width};
  input.data.assign(x, x + static_cast<std::size_t>(batch * in_c * height * width));
  welvet::TensorF32 pre_t, post_t;
  st = welvet::convt2::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_convt3_forward_f32(int32_t batch, int32_t in_c, int32_t filters, int32_t depth,
                                        int32_t height, int32_t width, int32_t kernel,
                                        int32_t stride, int32_t padding, int32_t output_padding,
                                        int32_t act, int32_t dtype, int32_t backend, const float* W,
                                        const float* x, float* pre, float* post) {
  if (batch <= 0 || in_c <= 0 || filters <= 0 || depth <= 0 || height <= 0 || width <= 0 ||
      kernel <= 0 || !W || !x || !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "convt3: invalid argument");
    return st;
  }
  if (backend != 0 && backend != 1) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "convt3: bad backend");
    return st;
  }
  welvet::convt3::Config cfg;
  cfg.in_channels = in_c;
  cfg.filters = filters;
  cfg.depth = depth;
  cfg.height = height;
  cfg.width = width;
  cfg.kernel = kernel;
  cfg.stride = stride;
  cfg.padding = padding;
  cfg.output_padding = output_padding;
  cfg.activation = static_cast<welvet::Activation>(act);
  welvet::convt3::Layer layer;
  auto st = welvet::convt3::new_configured(cfg, static_cast<welvet::DType>(dtype), W, &layer);
  if (!st) return to_c(st);
  layer.backend = backend == 1 ? welvet::Backend::SIMD : welvet::Backend::CPUTiled;
  welvet::TensorF32 input;
  input.shape = {batch, in_c, depth, height, width};
  input.data.assign(x, x + static_cast<std::size_t>(batch * in_c * depth * height * width));
  welvet::TensorF32 pre_t, post_t;
  st = welvet::convt3::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_rnn_forward_f32(int32_t batch, int32_t input_size, int32_t hidden_size,
                                     int32_t seq_len, int32_t dtype, int32_t backend,
                                     const float* packed, const float* x, float* pre, float* post) {
  if (batch <= 0 || input_size <= 0 || hidden_size <= 0 || seq_len <= 0 || !packed || !x || !pre ||
      !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "rnn: invalid argument");
    return st;
  }
  if (backend != 0 && backend != 1) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "rnn: bad backend");
    return st;
  }
  welvet::rnn::Config cfg;
  cfg.input_size = input_size;
  cfg.hidden_size = hidden_size;
  cfg.seq_len = seq_len;
  welvet::rnn::Layer layer;
  auto st = welvet::rnn::new_configured(cfg, static_cast<welvet::DType>(dtype), packed, &layer);
  if (!st) return to_c(st);
  layer.backend = backend == 1 ? welvet::Backend::SIMD : welvet::Backend::CPUTiled;
  welvet::TensorF32 input;
  input.shape = {batch, seq_len, input_size};
  input.data.assign(x, x + static_cast<std::size_t>(batch * seq_len * input_size));
  welvet::TensorF32 pre_t, post_t;
  st = welvet::rnn::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_lstm_forward_f32(int32_t batch, int32_t input_size, int32_t hidden_size,
                                      int32_t seq_len, int32_t dtype, int32_t backend,
                                      const float* packed, const float* x, float* pre, float* post) {
  if (batch <= 0 || input_size <= 0 || hidden_size <= 0 || seq_len <= 0 || !packed || !x || !pre ||
      !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "lstm: invalid argument");
    return st;
  }
  if (backend != 0 && backend != 1) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "lstm: bad backend");
    return st;
  }
  welvet::lstm::Config cfg;
  cfg.input_size = input_size;
  cfg.hidden_size = hidden_size;
  cfg.seq_len = seq_len;
  welvet::lstm::Layer layer;
  auto st = welvet::lstm::new_configured(cfg, static_cast<welvet::DType>(dtype), packed, &layer);
  if (!st) return to_c(st);
  layer.backend = backend == 1 ? welvet::Backend::SIMD : welvet::Backend::CPUTiled;
  welvet::TensorF32 input;
  input.shape = {batch, seq_len, input_size};
  input.data.assign(x, x + static_cast<std::size_t>(batch * seq_len * input_size));
  welvet::TensorF32 pre_t, post_t;
  st = welvet::lstm::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

const char* welvet_version(void) { return "0.1.10-m1-rnn"; }

int32_t welvet_gap_count(void) { return static_cast<int32_t>(welvet::gaps::count_gaps()); }

void welvet_print_gaps(void) { welvet::gaps::print_census(std::cout); }

} // extern "C"
