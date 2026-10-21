#include "welvet/architecture.hpp"

#include <sstream>

namespace welvet::architecture {
namespace {

Status bad_coord(const char* which, int z, int y, int x, int l) {
  std::ostringstream oss;
  oss << "architecture: bad " << which << " coord z=" << z << " y=" << y << " x=" << x
      << " l=" << l;
  return Status::Err(StatusCode::InvalidArgument, oss.str());
}

} // namespace

Grid new_grid(int depth, int rows, int cols, int layers_per_cell) {
  if (depth < 1) depth = 1;
  if (rows < 1) rows = 1;
  if (cols < 1) cols = 1;
  if (layers_per_cell < 1) layers_per_cell = 1;

  Grid g;
  g.depth = depth;
  g.rows = rows;
  g.cols = cols;
  g.layers_per_cell = layers_per_cell;
  g.native_exact = true;
  g.exec = ExecConfig{Backend::CPUTiled, true, 32};
  g.cells.resize(static_cast<std::size_t>(depth * rows * cols * layers_per_cell));

  for (int z = 0; z < depth; ++z) {
    for (int y = 0; y < rows; ++y) {
      for (int x = 0; x < cols; ++x) {
        for (int l = 0; l < layers_per_cell; ++l) {
          const int idx = index(g, z, y, x, l);
          Cell& c = g.cells[static_cast<std::size_t>(idx)];
          c.meta.type = LayerType::Dense;
          c.meta.dtype = DType::Float32;
          c.meta.activation = Activation::ReLU;
          c.meta.z = z;
          c.meta.y = y;
          c.meta.x = x;
          c.meta.l = l;
          c.meta.tile_size = 32;
          c.meta.multi_core = true;
          c.op_kind = OpKind::None;
        }
      }
    }
  }
  return g;
}

int index(const Grid& g, int z, int y, int x, int l) {
  return (((z * g.rows + y) * g.cols + x) * g.layers_per_cell + l);
}

Cell* at(Grid& g, int z, int y, int x, int l) {
  if (z < 0 || z >= g.depth || y < 0 || y >= g.rows || x < 0 || x >= g.cols || l < 0 ||
      l >= g.layers_per_cell) {
    return nullptr;
  }
  const int idx = index(g, z, y, x, l);
  if (idx < 0 || idx >= static_cast<int>(g.cells.size())) {
    return nullptr;
  }
  return &g.cells[static_cast<std::size_t>(idx)];
}

const Cell* at(const Grid& g, int z, int y, int x, int l) {
  return at(const_cast<Grid&>(g), z, y, x, l);
}

std::vector<Coord> hop_order(const Grid& g) {
  std::vector<Coord> out;
  out.reserve(static_cast<std::size_t>(stack_layer_count(g)));
  for (int z = 0; z < g.depth; ++z) {
    for (int y = 0; y < g.rows; ++y) {
      for (int x = 0; x < g.cols; ++x) {
        for (int l = 0; l < g.layers_per_cell; ++l) {
          out.push_back(Coord{z, y, x, l});
        }
      }
    }
  }
  return out;
}

Status set_remote_link(Grid& g, int z, int y, int x, int l, int tz, int ty, int tx, int tl) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("source", z, y, x, l);
  }
  if (!at(g, tz, ty, tx, tl)) {
    return bad_coord("target", tz, ty, tx, tl);
  }
  c->is_remote_link = true;
  c->target_z = tz;
  c->target_y = ty;
  c->target_x = tx;
  c->target_l = tl;
  return Status::Ok();
}

void clear_remote_link(Grid& g, int z, int y, int x, int l) {
  if (Cell* c = at(g, z, y, x, l)) {
    c->is_remote_link = false;
    c->target_z = c->target_y = c->target_x = c->target_l = 0;
  }
}

Cell* resolve_hop(Grid& g, Cell* c) {
  if (!c) {
    return nullptr;
  }
  if (c->is_remote_link) {
    return at(g, c->target_z, c->target_y, c->target_x, c->target_l);
  }
  return c;
}

const Cell* resolve_hop(const Grid& g, const Cell* c) {
  return resolve_hop(const_cast<Grid&>(g), const_cast<Cell*>(c));
}

Status bind_dense(Grid& g, int z, int y, int x, int l, dense::Layer op) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("bind", z, y, x, l);
  }
  c->meta.type = LayerType::Dense;
  c->meta.dtype = op.weights.dtype;
  c->meta.activation = op.activation;
  c->meta.input_height = op.input_height;
  c->meta.output_height = op.output_height;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = OpKind::Dense;
  c->dense_op = std::move(op);
  return Status::Ok();
}

Status bind_rmsnorm(Grid& g, int z, int y, int x, int l, rmsnorm::Layer op) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("bind", z, y, x, l);
  }
  c->meta = op.meta;
  c->meta.type = LayerType::RMSNorm;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = OpKind::RMSNorm;
  c->rmsnorm_op = std::move(op);
  return Status::Ok();
}

Status bind_layernorm(Grid& g, int z, int y, int x, int l, layernorm::Layer op) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("bind", z, y, x, l);
  }
  c->meta = op.meta;
  c->meta.type = LayerType::LayerNorm;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = OpKind::LayerNorm;
  c->layernorm_op = std::move(op);
  return Status::Ok();
}

Status bind_swiglu(Grid& g, int z, int y, int x, int l, swiglu::Layer op) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("bind", z, y, x, l);
  }
  c->meta = op.meta;
  c->meta.type = LayerType::SwiGLU;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = OpKind::SwiGLU;
  c->swiglu_op = std::move(op);
  return Status::Ok();
}

Status bind_softmax(Grid& g, int z, int y, int x, int l, softmax::Layer op) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("bind", z, y, x, l);
  }
  c->meta = op.meta;
  c->meta.type = LayerType::Softmax;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = OpKind::Softmax;
  c->softmax_op = std::move(op);
  return Status::Ok();
}

Status bind_embedding(Grid& g, int z, int y, int x, int l, embedding::Layer op) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("bind", z, y, x, l);
  }
  c->meta = op.meta;
  c->meta.type = LayerType::Embedding;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = OpKind::Embedding;
  c->embedding_op = std::move(op);
  return Status::Ok();
}

Status bind_mha(Grid& g, int z, int y, int x, int l, mha::Layer op) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("bind", z, y, x, l);
  }
  c->meta = op.meta;
  c->meta.type = LayerType::MultiHeadAttention;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = OpKind::MHA;
  c->mha_op = std::move(op);
  return Status::Ok();
}

Status bind_cnn1(Grid& g, int z, int y, int x, int l, cnn1::Layer op) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("bind", z, y, x, l);
  }
  c->meta = op.meta;
  c->meta.type = LayerType::CNN1;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = OpKind::CNN1;
  c->cnn1_op = std::move(op);
  return Status::Ok();
}

Status bind_cnn2(Grid& g, int z, int y, int x, int l, cnn2::Layer op) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("bind", z, y, x, l);
  }
  c->meta = op.meta;
  c->meta.type = LayerType::CNN2;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = OpKind::CNN2;
  c->cnn2_op = std::move(op);
  return Status::Ok();
}

Status bind_cnn3(Grid& g, int z, int y, int x, int l, cnn3::Layer op) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("bind", z, y, x, l);
  }
  c->meta = op.meta;
  c->meta.type = LayerType::CNN3;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = OpKind::CNN3;
  c->cnn3_op = std::move(op);
  return Status::Ok();
}

Status bind_convt1(Grid& g, int z, int y, int x, int l, convt1::Layer op) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("bind", z, y, x, l);
  }
  c->meta = op.meta;
  c->meta.type = LayerType::ConvTransposed1D;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = OpKind::ConvT1;
  c->convt1_op = std::move(op);
  return Status::Ok();
}

Status bind_convt2(Grid& g, int z, int y, int x, int l, convt2::Layer op) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("bind", z, y, x, l);
  }
  c->meta = op.meta;
  c->meta.type = LayerType::ConvTransposed2D;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = OpKind::ConvT2;
  c->convt2_op = std::move(op);
  return Status::Ok();
}

Status bind_convt3(Grid& g, int z, int y, int x, int l, convt3::Layer op) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("bind", z, y, x, l);
  }
  c->meta = op.meta;
  c->meta.type = LayerType::ConvTransposed3D;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = OpKind::ConvT3;
  c->convt3_op = std::move(op);
  return Status::Ok();
}

Status bind_rnn(Grid& g, int z, int y, int x, int l, rnn::Layer op) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("bind", z, y, x, l);
  }
  c->meta = op.meta;
  c->meta.type = LayerType::RNN;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = OpKind::RNN;
  c->rnn_op = std::move(op);
  return Status::Ok();
}

Status bind_lstm(Grid& g, int z, int y, int x, int l, lstm::Layer op) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("bind", z, y, x, l);
  }
  c->meta = op.meta;
  c->meta.type = LayerType::LSTM;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = OpKind::LSTM;
  c->lstm_op = std::move(op);
  return Status::Ok();
}

Status bind_sequential(Grid& g, int z, int y, int x, int l, sequential::Layer op) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("bind", z, y, x, l);
  }
  c->meta = op.meta;
  c->meta.type = LayerType::Sequential;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = OpKind::Sequential;
  c->sequential_op = std::move(op);
  return Status::Ok();
}

Status bind_residual(Grid& g, int z, int y, int x, int l, residual::Layer op) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("bind", z, y, x, l);
  }
  c->meta = op.meta;
  c->meta.type = LayerType::Residual;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = OpKind::Residual;
  c->residual_op = std::move(op);
  return Status::Ok();
}

Status bind_mamba(Grid& g, int z, int y, int x, int l, mamba::Layer op) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("bind", z, y, x, l);
  }
  c->meta = op.meta;
  c->meta.type = LayerType::Mamba;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = OpKind::Mamba;
  c->mamba_op = std::move(op);
  return Status::Ok();
}

Status bind_stack(Grid& g, int z, int y, int x, int l, stack::Layer op) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("bind", z, y, x, l);
  }
  c->meta = op.meta;
  c->meta.type = LayerType::Stack;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = OpKind::Stack;
  c->stack_op = std::move(op);
  return Status::Ok();
}

Status bind_metacognition(Grid& g, int z, int y, int x, int l, metacognition::Layer op) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("bind", z, y, x, l);
  }
  c->meta = op.meta;
  c->meta.type = LayerType::Metacognition;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = OpKind::Metacognition;
  c->metacognition_op = std::move(op);
  return Status::Ok();
}

Status bind_gdn(Grid& g, int z, int y, int x, int l, gdn::Layer op) {
  Cell* c = at(g, z, y, x, l);
  if (!c) {
    return bad_coord("bind", z, y, x, l);
  }
  c->meta = op.meta;
  c->meta.type = LayerType::GDN;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = OpKind::GDN;
  c->gdn_op = std::move(op);
  return Status::Ok();
}

} // namespace welvet::architecture
