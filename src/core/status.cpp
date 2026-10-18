#include "welvet/status.hpp"

// Status helpers are header-inline; this TU keeps the library linked.
namespace welvet {
namespace {
volatile int status_tu_anchor = 0;
}
void status_force_link() { status_tu_anchor = 1; }
} // namespace welvet
