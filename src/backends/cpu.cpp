#include "welvet/backends.hpp"

namespace welvet::backends {

Status cpu_load() { return Status::Ok(); }
Status cpu_unload() { return Status::Ok(); }

} // namespace welvet::backends
