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

Status backward_f32(const Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "dense: nil grad out");
  }
  if (input.shape.size() < 2 || grad_out.shape.size() < 2 || pre.shape.size() < 2) {
    return Status::Err(StatusCode::InvalidArgument, "dense: bad shapes");
  }
  const int batch = static_cast<int>(input.shape[0]);
  const int in = layer.input_height;
  const int out = layer.output_height;
  if (static_cast<int>(input.shape[1]) != in || static_cast<int>(grad_out.shape[0]) != batch ||
      static_cast<int>(grad_out.shape[1]) != out || static_cast<int>(pre.shape[0]) != batch ||
      static_cast<int>(pre.shape[1]) != out) {
    return Status::Err(StatusCode::InvalidArgument, "dense: bwd shape mismatch");
  }
  if (layer.weights.rows != out || layer.weights.cols != in) {
    return Status::Err(StatusCode::InvalidArgument, "dense: weight shape mismatch");
  }

  std::vector<float> d_pre(static_cast<std::size_t>(batch * out));
  for (int i = 0; i < batch * out; ++i) {
    const double g = static_cast<double>(grad_out.data[static_cast<std::size_t>(i)]) *
                     activate_deriv_f64(static_cast<double>(pre.data[static_cast<std::size_t>(i)]),
                                        layer.activation);
    d_pre[static_cast<std::size_t>(i)] = static_cast<float>(g);
  }

  *grad_in = TensorF32::zeros(batch, in);
  *grad_w = TensorF32::zeros(out, in);

  for (int b = 0; b < batch; ++b) {
    const float* gy = d_pre.data() + static_cast<std::size_t>(b * out);
    float* gx = grad_in->data.data() + static_cast<std::size_t>(b * in);
    if (auto st = layer.weights.matvect_f32(gy, gx); !st) {
      return st;
    }
    const float* x_row = input.data.data() + static_cast<std::size_t>(b * in);
    for (int o = 0; o < out; ++o) {
      const double g = static_cast<double>(gy[o]);
      float* dw = grad_w->data.data() + static_cast<std::size_t>(o * in);
      for (int i = 0; i < in; ++i) {
        dw[i] = static_cast<float>(static_cast<double>(dw[i]) + g * static_cast<double>(x_row[i]));
      }
    }
  }
  return Status::Ok();
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  if (!layer) {
    return Status::Err(StatusCode::InvalidArgument, "dense: ApplyGradSGD nil");
  }
  const int n = layer->weights.rows * layer->weights.cols;
  if (static_cast<int>(dW.data.size()) < n) {
    return Status::Err(StatusCode::InvalidArgument, "dense: dW short");
  }
  std::vector<double> buf(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    buf[static_cast<std::size_t>(i)] = static_cast<double>(dW.data[static_cast<std::size_t>(i)]);
  }
  return layer->weights.apply_sgd(buf.data(), lr);
}

} // namespace welvet::dense

