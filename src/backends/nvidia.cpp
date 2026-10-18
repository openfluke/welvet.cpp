#include "welvet/backends.hpp"

namespace welvet::backends {

Status nvidia_load() {
  return Status::Gap("nvidia CUDA device discovery + dense GEMV/GEMM kernels");
}
Status nvidia_unload() { return Status::Gap("nvidia CUDA unload"); }

} // namespace welvet::backends
