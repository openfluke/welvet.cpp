#pragma once

#include "welvet/activate.hpp"
#include "welvet/backend.hpp"
#include "welvet/dtype.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"
#include "welvet/weights.hpp"

namespace welvet::dense {

struct Layer {
  int input_height = 0;
  int output_height = 0;
  Activation activation = Activation::Linear;
  WeightStore weights;
  Backend backend = Backend::CPUTiled; // CPUTiled = exact Go CPU path; SIMD = imported DotTile
  bool multi_core = false;             // gap? tiled multicore path
};

Status new_layer(int in_h, int out_h, Activation act, DType dt, const float* init_or_null,
                 Layer* layer);

Status new_f32(int in_h, int out_h, Activation act, Layer* layer);

// ForwardCPUTiled-equivalent for FormatNone × any dtype (serial MatVec + bias + act).
Status forward_f32(const Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

// Imported SIMD GEMV (welvet-to-rust_n_cpp DotTile / DotI8 / packed expand).
Status forward_f32_simd(const Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

// Dispatch by layer.backend.
Status forward(const Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

} // namespace welvet::dense

