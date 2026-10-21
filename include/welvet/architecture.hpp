#pragma once

#include "welvet/dense.hpp"
#include "welvet/layer.hpp"
#include "welvet/layernorm.hpp"
#include "welvet/rmsnorm.hpp"
#include "welvet/status.hpp"

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
  // gap? other layer ops
};

// One processing unit in the grid (Go architecture.Cell topology subset).
struct Cell {
  LayerMeta meta;
  OpKind op_kind = OpKind::None;
  dense::Layer dense_op;         // valid when op_kind == Dense
  rmsnorm::Layer rmsnorm_op;     // valid when op_kind == RMSNorm
  layernorm::Layer layernorm_op; // valid when op_kind == LayerNorm

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

} // namespace welvet::architecture
