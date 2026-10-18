// Bridge: welvet WeightStore → imported gemvSIMD (from welvet-to-rust_n_cpp).
#include "welvet/dense.hpp"
#include "welvet/imported/dense.hpp"

#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace welvet::dense {
namespace {

welvet_imported::NativeStore to_native(const WeightStore& w) {
  welvet_imported::NativeStore s;
  s.dtype = static_cast<welvet_imported::DType>(static_cast<int>(w.dtype));
  s.rows = w.rows;
  s.cols = w.cols;
  s.scale = w.scale == 0.f ? 1.f : w.scale;
  if (w.dtype == DType::Float32) {
    const auto n = static_cast<std::size_t>(w.rows) * static_cast<std::size_t>(w.cols);
    s.raw.resize(n * sizeof(float));
    if (!w.f32.empty()) {
      std::memcpy(s.raw.data(), w.f32.data(), s.raw.size());
    }
  } else {
    s.raw = w.native;
  }
  welvet_imported::prepareWireF64(s);
  return s;
}

void apply_bias_act(float* pre, float* post, const std::vector<double>& bias, Activation act,
                    int batch, int out) {
  const int n = batch * out;
  const bool linear_no_bias = (act == Activation::Linear) && bias.empty();
  if (linear_no_bias) {
    if (pre != post) {
      for (int i = 0; i < n; ++i) {
        post[i] = pre[i];
      }
    }
    return;
  }
  for (int b = 0; b < batch; ++b) {
    for (int o = 0; o < out; ++o) {
      const int i = b * out + o;
      double v = static_cast<double>(pre[i]);
      if (o < static_cast<int>(bias.size())) {
        v += bias[static_cast<std::size_t>(o)];
        pre[i] = static_cast<float>(v);
      }
      post[i] = static_cast<float>(activate_f64(v, act));
    }
  }
}

} // namespace

Status forward_f32_simd(const Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "dense simd: nil out");
  }
  if (input.shape.size() < 2) {
    return Status::Err(StatusCode::InvalidArgument, "dense simd: bad shape");
  }
  const int batch = static_cast<int>(input.shape[0]);
  const int in = layer.input_height;
  const int out = layer.output_height;
  if (input.shape[1] != in || layer.weights.rows != out || layer.weights.cols != in) {
    return Status::Err(StatusCode::InvalidArgument, "dense simd: dims");
  }

  *pre = TensorF32::zeros(batch, out);
  *post = TensorF32::zeros(batch, out);

  try {
    auto store = to_native(layer.weights);
    welvet_imported::gemvSIMD(pre->data.data(), input.data.data(), store, batch);
  } catch (const std::exception& e) {
    return Status::Err(StatusCode::Internal, std::string("dense simd: ") + e.what());
  }

  apply_bias_act(pre->data.data(), post->data.data(), layer.weights.bias, layer.activation, batch,
                 out);
  return Status::Ok();
}

} // namespace welvet::dense
