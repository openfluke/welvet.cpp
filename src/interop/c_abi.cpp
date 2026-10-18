#include "welvet_c.h"

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
                                       int32_t dtype, const float* W_init, const double* bias,
                                       size_t bias_len, const float* x, float* pre, float* post) {
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

  welvet::dense::Layer layer;
  auto st = welvet::dense::new_layer(in, out, static_cast<welvet::Activation>(act),
                                     static_cast<welvet::DType>(dtype), W_init, &layer);
  if (!st) {
    return to_c(st);
  }
  if (bias && bias_len > 0) {
    layer.weights.bias.assign(bias, bias + bias_len);
  }

  welvet::TensorF32 input;
  input.shape = {batch, in};
  input.data.assign(x, x + static_cast<std::size_t>(batch * in));

  welvet::TensorF32 pre_t, post_t;
  st = welvet::dense::forward_f32(layer, input, &pre_t, &post_t);
  if (!st) {
    return to_c(st);
  }
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_dense_forward_f32_weights(int32_t batch, int32_t in, int32_t out, int32_t act,
                                               const float* W, const double* bias, size_t bias_len,
                                               const float* x, float* pre, float* post) {
  return welvet_dense_forward_f32(batch, in, out, act, /*Float32*/ 1, W, bias, bias_len, x, pre,
                                  post);
}

const char* welvet_version(void) { return "0.1.1-m1-dense-forward-34dtypes"; }

int32_t welvet_gap_count(void) { return static_cast<int32_t>(welvet::gaps::count_gaps()); }

void welvet_print_gaps(void) { welvet::gaps::print_census(std::cout); }

} // extern "C"
