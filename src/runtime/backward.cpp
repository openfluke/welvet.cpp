#include "welvet/runtime/backward.hpp"

#include <sstream>
#include <utility>

namespace welvet::runtime::backward {

Status volume_dense(architecture::Grid& g, const forward::Result& fwd, const TensorF32& grad_out,
                    Result* out) {
  if (!out) {
    return Status::Err(StatusCode::InvalidArgument, "runtime.backward: nil result");
  }
  if (fwd.steps.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "runtime.backward: empty tape");
  }
  if (grad_out.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "runtime.backward: empty gradOut");
  }

  Result result;
  result.grad_ws.reserve(fwd.steps.size());
  TensorF32 gy = grad_out;

  for (int i = static_cast<int>(fwd.steps.size()) - 1; i >= 0; --i) {
    const auto& st = fwd.steps[static_cast<std::size_t>(i)];
    architecture::Cell* cell = architecture::at(g, st.coord.z, st.coord.y, st.coord.x, st.coord.l);
    if (!cell) {
      std::ostringstream oss;
      oss << "runtime.backward: missing cell at z=" << st.coord.z << " y=" << st.coord.y
          << " x=" << st.coord.x << " l=" << st.coord.l;
      return Status::Err(StatusCode::InvalidArgument, oss.str());
    }

    TensorF32 gx, dw;
    Status st_b = Status::Ok();
    switch (cell->op_kind) {
    case architecture::OpKind::Dense:
      st_b = dense::backward_f32(cell->dense_op, gy, st.input, st.pre, &gx, &dw);
      break;
    case architecture::OpKind::RMSNorm:
      st_b = rmsnorm::backward_f32(cell->rmsnorm_op, gy, st.input, st.pre, &gx, &dw);
      break;
    case architecture::OpKind::LayerNorm:
      st_b = layernorm::backward_f32(cell->layernorm_op, gy, st.input, st.pre, &gx, &dw);
      break;
    case architecture::OpKind::SwiGLU:
      st_b = swiglu::backward_f32(cell->swiglu_op, gy, st.input, st.pre, &gx, &dw);
      break;
    case architecture::OpKind::Softmax:
      st_b = softmax::backward_f32(cell->softmax_op, gy, st.input, st.pre, &gx, &dw);
      break;
    default: {
      std::ostringstream oss;
      oss << "runtime.backward: unsupported Op at z=" << st.coord.z << " y=" << st.coord.y
          << " x=" << st.coord.x << " l=" << st.coord.l;
      return Status::Err(StatusCode::InvalidArgument, oss.str());
    }
    }
    if (!st_b) {
      return st_b;
    }
    // Softmax (weightless) returns empty dW — skip like Go (nil GradW omitted).
    if (!dw.data.empty()) {
      result.grad_ws.push_back(GradW{st.coord, std::move(dw)});
    }
    gy = std::move(gx);
  }

  result.grad_in = std::move(gy);
  *out = std::move(result);
  return Status::Ok();
}

} // namespace welvet::runtime::backward
