#include "welvet/dense.hpp"

#include "welvet/gaps.hpp"

namespace welvet::dense {

Status new_layer(int in_h, int out_h, Activation act, DType dt, const float* init_or_null,
                 Layer* layer) {
  if (!layer || in_h <= 0 || out_h <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "dense: bad dims");
  }
  layer->input_height = in_h;
  layer->output_height = out_h;
  layer->activation = act;
  layer->multi_core = false;
  return WeightStore::make(out_h, in_h, dt, init_or_null, &layer->weights);
}

Status new_f32(int in_h, int out_h, Activation act, Layer* layer) {
  return new_layer(in_h, out_h, act, DType::Float32, nullptr, layer);
}

static void apply_bias_act(float* pre, float* post, const std::vector<double>& bias,
                           Activation act, int batch, int out) {
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

Status forward_f32(const Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "dense: nil out");
  }
  if (input.shape.size() < 2) {
    return Status::Err(StatusCode::InvalidArgument, "dense: input shape need [batch,in]");
  }
  const int batch = static_cast<int>(input.shape[0]);
  const int in = layer.input_height;
  const int out = layer.output_height;
  if (input.shape[1] != in) {
    return Status::Err(StatusCode::InvalidArgument, "dense: input width mismatch");
  }
  if (layer.weights.rows != out || layer.weights.cols != in) {
    return Status::Err(StatusCode::InvalidArgument, "dense: weight shape mismatch");
  }
  if (layer.multi_core) {
    (void)gaps::mark("dense.forward multicore tiled");
  }

  *pre = TensorF32::zeros(batch, out);
  *post = TensorF32::zeros(batch, out);

  for (int b = 0; b < batch; ++b) {
    const float* x = input.data.data() + static_cast<std::size_t>(b * in);
    float* y = pre->data.data() + static_cast<std::size_t>(b * out);
    if (auto st = layer.weights.matvec_f32(x, y); !st) {
      return st;
    }
  }
  apply_bias_act(pre->data.data(), post->data.data(), layer.weights.bias, layer.activation, batch,
                 out);
  return Status::Ok();
}

Status forward(const Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (layer.backend == Backend::SIMD) {
    return forward_f32_simd(layer, input, pre, post);
  }
  return forward_f32(layer, input, pre, post);
}

} // namespace welvet::dense

