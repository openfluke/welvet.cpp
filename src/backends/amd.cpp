#include "welvet/backends.hpp"

namespace welvet::backends {

Status amd_load() { return Status::Gap("amd ROCm/HIP device discovery + dense kernels"); }
Status amd_unload() { return Status::Gap("amd ROCm unload"); }

} // namespace welvet::backends
