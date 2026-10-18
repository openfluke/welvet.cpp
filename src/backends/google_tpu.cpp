#include "welvet/backends.hpp"

namespace welvet::backends {

Status tpu_load() { return Status::Gap("google TPU / PJRT load + dense kernels"); }
Status tpu_unload() { return Status::Gap("google TPU unload"); }

} // namespace welvet::backends
