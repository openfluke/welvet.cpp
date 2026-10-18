#include "welvet/backends.hpp"
#include "welvet/dense.hpp"
#include "welvet/gaps.hpp"

#include <cstdio>
#include <iostream>

int main() {
  const float init[] = {1, 0, 0, 0, 1, 0};
  welvet::dense::Layer layer;
  auto st = welvet::dense::new_layer(3, 2, welvet::Activation::Linear, welvet::DType::Float32, init,
                                     &layer);
  if (!st) {
    std::fprintf(stderr, "%s\n", st.message.c_str());
    return 1;
  }
  layer.weights.bias = {0.5, -0.5};

  welvet::TensorF32 x;
  x.shape = {1, 3};
  x.data = {2, 3, 9};

  welvet::TensorF32 pre, post;
  st = welvet::dense::forward_f32(layer, x, &pre, &post);
  if (!st) {
    std::fprintf(stderr, "%s\n", st.message.c_str());
    return 1;
  }

  std::printf("pre = [%g, %g]  (want 2.5, 2.5)\n", pre.data[0], pre.data[1]);
  std::printf("post = [%g, %g]\n", post.data[0], post.data[1]);

  std::cout << "\nbackends:\n";
  for (const auto& b : welvet::backends::list()) {
    std::cout << "  " << b.name << " available=" << (b.available ? "yes" : "no") << " — " << b.note
              << "\n";
  }

  std::cout << "\n";
  welvet::gaps::print_census(std::cout);
  return (pre.data[0] == 2.5f && pre.data[1] == 2.5f) ? 0 : 1;
}
