#include "welvet/backends.hpp"

namespace welvet::backends {

Status apple_load() {
  return Status::Gap("apple Metal / Neural Engine load + dense kernels");
}
Status apple_unload() { return Status::Gap("apple unload"); }

} // namespace welvet::backends
