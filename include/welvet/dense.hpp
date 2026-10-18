#pragma once

#include "welvet/activate.hpp"
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
  bool multi_core = false; // gap? tiled multicore path — serial used for exact parity
};

Status new_layer(int in_h, int out_h, Activation act, DType dt, const float* init_or_null,
                 Layer* layer);

Status new_f32(int in_h, int out_h, Activation act, Layer* layer);

// ForwardCPUTiled-equivalent for FormatNone × any dtype (serial MatVec + bias + act).
// Activation host type is float32 (matches w2a dense smoke Numeric float32 acts).
Status forward_f32(const Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

} // namespace welvet::dense
