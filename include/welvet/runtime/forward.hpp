#pragma once

#include "welvet/architecture.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

#include <vector>

namespace welvet::runtime::forward {

struct Step {
  architecture::Coord coord;
  TensorF32 input;
  TensorF32 pre;
  TensorF32 post;
};

struct Result {
  TensorF32 output;
  std::vector<Step> steps;
};

// Walk HopOrder; Dense ops only. Remote-link cells take input from target Post.
// Other LayerType / missing Op → Status::Gap / Invalid.
Status volume_dense(architecture::Grid& g, const TensorF32& input, Result* out);

} // namespace welvet::runtime::forward
