#include "nn/layers/global_avg_pool2d.hpp"

#include "ensure.hpp"
#include "nn/error.hpp"
#include "ocl.hpp"

namespace nn {

struct GlobalAvgPool2d::Impl {};

GlobalAvgPool2d::GlobalAvgPool2d(Context& ctx) : ctx_(&ctx), impl_(std::make_unique<Impl>()) {}

GlobalAvgPool2d::~GlobalAvgPool2d() = default;

std::vector<NamedParam> GlobalAvgPool2d::parameters() const { return {}; }

void GlobalAvgPool2d::zero_grad() {}

const Tensor& GlobalAvgPool2d::forward(const Tensor& input) {
  if (input.rank() != 4) throw Error("GlobalAvgPool2d input rank");
  const int n = input.shape()[0];
  const int c = input.shape()[1];
  const int h = input.shape()[2];
  const int w = input.shape()[3];
  ensure_tensor(cache_x_, *ctx_, input.shape());
  ensure_tensor(output_, *ctx_, {n, c, 1, 1});
  ensure_tensor(cache_y_, *ctx_, output_->shape());
  cache_x_->copy_from(input);
  launch_global_avg_forward(*ctx_, output_->data(), input.data(), n, c, h, w);
  cache_y_->copy_from(*output_);
  return *output_;
}

const Tensor& GlobalAvgPool2d::backward(const Tensor& grad_output) {
  if (!cache_x_ || !cache_y_ || grad_output.shape() != cache_y_->shape()) {
    throw Error("GlobalAvgPool2d backward shape");
  }
  ensure_tensor(grad_input_, *ctx_, cache_x_->shape());
  const int n = cache_x_->shape()[0];
  const int c = cache_x_->shape()[1];
  const int h = cache_x_->shape()[2];
  const int w = cache_x_->shape()[3];
  launch_global_avg_backward(*ctx_, grad_input_->data(), grad_output.data(), n, c, h, w);
  return *grad_input_;
}

}  // namespace nn
