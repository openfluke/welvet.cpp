#include "welvet/backends.hpp"

namespace welvet::backends {

Status intel_load() {
  return Status::Gap("intel GPU/NPU (oneAPI / OpenVINO) load + dense kernels");
}
Status intel_unload() { return Status::Gap("intel unload"); }

} // namespace welvet::backends
