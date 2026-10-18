#include "welvet/backends.hpp"

namespace welvet::backends {

Status qualcomm_load() {
  return Status::Gap("qualcomm Adreno GPU / QNN NPU load + dense kernels");
}
Status qualcomm_unload() { return Status::Gap("qualcomm unload"); }

} // namespace welvet::backends
