#include "welvet/weights.hpp"

#include <vector>

namespace welvet {

Status WeightStore::make(int rows, int cols, DType dt, const float* init_or_null, WeightStore* out) {
  if (!out || rows <= 0 || cols <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "weights: bad shape");
  }
  out->rows = rows;
  out->cols = cols;
  out->bias.clear();
  const int n = rows * cols;
  std::vector<float> scratch(static_cast<std::size_t>(n), 0.0f);
  if (init_or_null) {
    for (int i = 0; i < n; ++i) {
      scratch[static_cast<std::size_t>(i)] = init_or_null[i];
    }
  }
  return out->encode_format_none(dt, scratch.data(), n);
}

Status WeightStore::matvec_f32(const float* x, float* y) const {
  if (!x || !y) {
    return Status::Err(StatusCode::InvalidArgument, "weights: nil ptr");
  }
  if (format != QuantFormat::None) {
    return Status::Gap("weights.MatVec packed quant formats");
  }
  std::vector<double> w_row(static_cast<std::size_t>(cols));
  for (int r = 0; r < rows; ++r) {
    if (auto st = decode_row_f64(r, w_row.data()); !st) {
      return st;
    }
    double sum = 0.0;
    for (int c = 0; c < cols; ++c) {
      sum += w_row[static_cast<std::size_t>(c)] * static_cast<double>(x[c]);
    }
    y[r] = static_cast<float>(sum);
  }
  return Status::Ok();
}

} // namespace welvet
