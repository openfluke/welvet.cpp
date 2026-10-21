#include "welvet/runtime/forward.hpp"

#include <map>
#include <sstream>
#include <utility>

namespace welvet::runtime::forward {
namespace {

struct CoordLess {
  bool operator()(const architecture::Coord& a, const architecture::Coord& b) const {
    if (a.z != b.z) return a.z < b.z;
    if (a.y != b.y) return a.y < b.y;
    if (a.x != b.x) return a.x < b.x;
    return a.l < b.l;
  }
};

Status dispatch_cell(architecture::Cell& cell, const TensorF32& in, TensorF32* pre,
                     TensorF32* post) {
  switch (cell.op_kind) {
  case architecture::OpKind::Dense:
    return dense::forward(cell.dense_op, in, pre, post);
  case architecture::OpKind::RMSNorm:
    return rmsnorm::forward(cell.rmsnorm_op, in, pre, post);
  case architecture::OpKind::LayerNorm:
    return layernorm::forward(cell.layernorm_op, in, pre, post);
  case architecture::OpKind::SwiGLU:
    return swiglu::forward(cell.swiglu_op, in, pre, post);
  case architecture::OpKind::Softmax:
    return softmax::forward(cell.softmax_op, in, pre, post);
  default:
    return Status::Gap("runtime.forward: unsupported cell Op");
  }
}

} // namespace

Status volume_dense(architecture::Grid& g, const TensorF32& input, Result* out) {
  if (!out) {
    return Status::Err(StatusCode::InvalidArgument, "runtime.forward: nil result");
  }
  if (input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "runtime.forward: empty input");
  }

  TensorF32 current = input;
  std::map<architecture::Coord, TensorF32, CoordLess> posts;
  Result result;

  for (const auto& coord : architecture::hop_order(g)) {
    architecture::Cell* cell = architecture::at(g, coord.z, coord.y, coord.x, coord.l);
    if (!cell || cell->meta.is_disabled) {
      continue;
    }
    if (cell->op_kind == architecture::OpKind::None) {
      std::ostringstream oss;
      oss << "runtime.forward: no op at z=" << coord.z << " y=" << coord.y << " x=" << coord.x
          << " l=" << coord.l;
      return Status::Err(StatusCode::InvalidArgument, oss.str());
    }

    const TensorF32* in = &current;
    TensorF32 remote_holder;
    if (cell->is_remote_link) {
      architecture::Coord tc{cell->target_z, cell->target_y, cell->target_x, cell->target_l};
      auto it = posts.find(tc);
      if (it == posts.end()) {
        std::ostringstream oss;
        oss << "runtime.forward: remote hop has no recorded activation → z=" << tc.z
            << " y=" << tc.y << " x=" << tc.x << " l=" << tc.l;
        return Status::Err(StatusCode::InvalidArgument, oss.str());
      }
      remote_holder = it->second;
      in = &remote_holder;
    }

    TensorF32 pre, post;
    Status st = dispatch_cell(*cell, *in, &pre, &post);
    if (!st) {
      return st;
    }

    Step step;
    step.coord = coord;
    step.input = *in;
    step.pre = std::move(pre);
    step.post = post;
    posts[coord] = post;
    current = std::move(post);
    result.steps.push_back(std::move(step));
  }

  if (result.steps.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "runtime.forward: grid has no enabled cells with ops");
  }
  result.output = std::move(current);
  *out = std::move(result);
  return Status::Ok();
}

} // namespace welvet::runtime::forward
