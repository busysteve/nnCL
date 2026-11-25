#include "check.hpp"

#include "nn/dtype.hpp"
#include "nn/tensor.hpp"

#include <cstdio>
#include <vector>

namespace {

void expect_value(const std::vector<float>& got, float want, const char* name) {
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] == want) continue;
    std::fprintf(stderr, "%s[%zu] got %.8g want %.8g\n", name, i, static_cast<double>(got[i]),
                 static_cast<double>(want));
    CHECK(false);
  }
}

}  // namespace

int test_fill() {
  nn::Context ctx;
  ctx.set_activation_dtype(nn::DType::Float32);

  // 300 values cross a 256-wide work group.
  nn::Tensor values(ctx, {300});
  values.fill(3.5f);
  expect_value(values.to_host(), 3.5f, "fill");
  values.zero();
  expect_value(values.to_host(), 0.f, "zero");
  values.fill(0.f);
  expect_value(values.to_host(), 0.f, "fill-zero");

  // Three BFloat16 values are 6 bytes. That length is not a multiple of 4.
  nn::Tensor half(ctx, {3}, nn::DType::BFloat16);
  half.fill(1.f);
  expect_value(half.to_host(), 1.f, "bf16-fill");
  half.zero();
  expect_value(half.to_host(), 0.f, "bf16-zero");

  std::printf("fill ok\n");
  return 0;
}
