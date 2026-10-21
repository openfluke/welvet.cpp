#pragma once

#include "welvet/cnn1.hpp"
#include "welvet/cnn2.hpp"
#include "welvet/cnn3.hpp"
#include "welvet/convt1.hpp"
#include "welvet/convt2.hpp"
#include "welvet/convt3.hpp"
#include "welvet/dense.hpp"
#include "welvet/embedding.hpp"
#include "welvet/gdn.hpp"
#include "welvet/kmeans.hpp"
#include "welvet/layer.hpp"
#include "welvet/layernorm.hpp"
#include "welvet/lstm.hpp"
#include "welvet/mamba.hpp"
#include "welvet/metacognition.hpp"
#include "welvet/mha.hpp"
#include "welvet/parallel.hpp"
#include "welvet/rmsnorm.hpp"
#include "welvet/residual.hpp"
#include "welvet/rnn.hpp"
#include "welvet/sequential.hpp"
#include "welvet/softmax.hpp"
#include "welvet/stack.hpp"
#include "welvet/status.hpp"
#include "welvet/swiglu.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace welvet::architecture {

// Volumetric address: Z depth, Y row, X col, L layer-in-cell.
struct Coord {
  int z = 0, y = 0, x = 0, l = 0;
  bool operator==(const Coord& o) const {
    return z == o.z && y == o.y && x == o.x && l == o.l;
  }
};

enum class OpKind : int32_t {
  None = 0,
  Dense = 1,
  RMSNorm = 2,
  LayerNorm = 3,
  SwiGLU = 4,
  Softmax = 5,
  Embedding = 6,
  MHA = 7,
  CNN1 = 8,
  CNN2 = 9,
  CNN3 = 10,
  ConvT1 = 11,
  ConvT2 = 12,
  ConvT3 = 13,
  RNN = 14,
  LSTM = 15,
  Sequential = 16,
  Residual = 17,
  Mamba = 18,
  Stack = 19, // thin Stack host (Line/StepBP children; volume = full-chain NormalBP)
  Metacognition = 20,
  GDN = 21,
  KMeans = 22,
  Parallel = 23, // Dense-branch Parallel host (volume = Forward+Backward+SGD / family credit)
};

// One processing unit in the grid (Go architecture.Cell topology subset).
struct Cell {
  LayerMeta meta;
  OpKind op_kind = OpKind::None;
  dense::Layer dense_op;         // valid when op_kind == Dense
  rmsnorm::Layer rmsnorm_op;     // valid when op_kind == RMSNorm
  layernorm::Layer layernorm_op; // valid when op_kind == LayerNorm
  swiglu::Layer swiglu_op;       // valid when op_kind == SwiGLU
  softmax::Layer softmax_op;     // valid when op_kind == Softmax
  embedding::Layer embedding_op; // valid when op_kind == Embedding
  mha::Layer mha_op;             // valid when op_kind == MHA
  cnn1::Layer cnn1_op;           // valid when op_kind == CNN1
  cnn2::Layer cnn2_op;           // valid when op_kind == CNN2
  cnn3::Layer cnn3_op;           // valid when op_kind == CNN3
  convt1::Layer convt1_op;       // valid when op_kind == ConvT1
  convt2::Layer convt2_op;       // valid when op_kind == ConvT2
  convt3::Layer convt3_op;       // valid when op_kind == ConvT3
  rnn::Layer rnn_op;             // valid when op_kind == RNN
  lstm::Layer lstm_op;           // valid when op_kind == LSTM
  sequential::Layer sequential_op; // valid when op_kind == Sequential
  residual::Layer residual_op;     // valid when op_kind == Residual
  mamba::Layer mamba_op;           // valid when op_kind == Mamba
  stack::Layer stack_op;           // valid when op_kind == Stack
  metacognition::Layer metacognition_op; // valid when op_kind == Metacognition
  gdn::Layer gdn_op;               // valid when op_kind == GDN
  kmeans::Layer kmeans_op;         // valid when op_kind == KMeans
  parallel::Layer parallel_op;     // valid when op_kind == Parallel

  bool is_remote_link = false;
  int target_z = 0, target_y = 0, target_x = 0, target_l = 0;

  std::string combine_mode; // "concat", "add", "avg", … (branches gap? for now)
};

// Depth × Rows × Cols × LayersPerCell volumetric container.
struct Grid {
  int depth = 1;
  int rows = 1;
  int cols = 1;
  int layers_per_cell = 1;
  std::vector<Cell> cells;
  ExecConfig exec{};
  bool native_exact = true;
  stack::SplitAccel accel{}; // LinearCache / HeadProxyAsync / Sparse
};

using VolumetricNetwork = Grid;

// Allocate grid; stamps default Dense+ReLU+Float32 cell metadata (no Op payloads).
Grid new_grid(int depth, int rows, int cols, int layers_per_cell);

inline Grid new_volumetric_network(int depth, int rows, int cols, int layers_per_cell) {
  return new_grid(depth, rows, cols, layers_per_cell);
}

int index(const Grid& g, int z, int y, int x, int l);
Cell* at(Grid& g, int z, int y, int x, int l);
const Cell* at(const Grid& g, int z, int y, int x, int l);

inline int stack_layer_count(const Grid& g) {
  return g.depth * g.rows * g.cols * g.layers_per_cell;
}

// Traversal order z→y→x→l.
std::vector<Coord> hop_order(const Grid& g);

Status set_remote_link(Grid& g, int z, int y, int x, int l, int tz, int ty, int tx, int tl);
void clear_remote_link(Grid& g, int z, int y, int x, int l);

// Effective source cell for a hop (remote target or self).
Cell* resolve_hop(Grid& g, Cell* c);
const Cell* resolve_hop(const Grid& g, const Cell* c);

// Bind a Dense op at coordinates (syncs meta type/geometry from the dense layer).
Status bind_dense(Grid& g, int z, int y, int x, int l, dense::Layer op);
Status bind_rmsnorm(Grid& g, int z, int y, int x, int l, rmsnorm::Layer op);
Status bind_layernorm(Grid& g, int z, int y, int x, int l, layernorm::Layer op);
Status bind_swiglu(Grid& g, int z, int y, int x, int l, swiglu::Layer op);
Status bind_softmax(Grid& g, int z, int y, int x, int l, softmax::Layer op);
Status bind_embedding(Grid& g, int z, int y, int x, int l, embedding::Layer op);
Status bind_mha(Grid& g, int z, int y, int x, int l, mha::Layer op);
Status bind_cnn1(Grid& g, int z, int y, int x, int l, cnn1::Layer op);
Status bind_cnn2(Grid& g, int z, int y, int x, int l, cnn2::Layer op);
Status bind_cnn3(Grid& g, int z, int y, int x, int l, cnn3::Layer op);
Status bind_convt1(Grid& g, int z, int y, int x, int l, convt1::Layer op);
Status bind_convt2(Grid& g, int z, int y, int x, int l, convt2::Layer op);
Status bind_convt3(Grid& g, int z, int y, int x, int l, convt3::Layer op);
Status bind_rnn(Grid& g, int z, int y, int x, int l, rnn::Layer op);
Status bind_lstm(Grid& g, int z, int y, int x, int l, lstm::Layer op);
Status bind_sequential(Grid& g, int z, int y, int x, int l, sequential::Layer op);
Status bind_residual(Grid& g, int z, int y, int x, int l, residual::Layer op);
Status bind_mamba(Grid& g, int z, int y, int x, int l, mamba::Layer op);
Status bind_stack(Grid& g, int z, int y, int x, int l, stack::Layer op);
Status bind_metacognition(Grid& g, int z, int y, int x, int l, metacognition::Layer op);
Status bind_gdn(Grid& g, int z, int y, int x, int l, gdn::Layer op);
Status bind_kmeans(Grid& g, int z, int y, int x, int l, kmeans::Layer op);
Status bind_parallel(Grid& g, int z, int y, int x, int l, parallel::Layer op);

} // namespace welvet::architecture
