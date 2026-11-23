#include "nn/layers/relu.hpp"

#include "activate.hpp"
#include "ensure.hpp"
#include "nn/error.hpp"
#include "ocl.hpp"

namespace nn {

struct ReLU::Impl {};

ReLU::ReLU(Context& ctx) : ctx_(&ctx), impl_(std::make_unique<Impl>()) {}

ReLU::~ReLU() = default;

std::vector<NamedParam> ReLU::parameters() const { return {}; }

void ReLU::zero_grad() {}

namespace {

void require_rank(const std::vector<int>& shape) {
  if (shape.size() != 2 && shape.size() != 3 && shape.size() != 4) throw Error("ReLU rank");
}

}  // namespace

const Tensor& ReLU::forward(const Tensor& input) {
  require_rank(input.shape());
  ensure_activation(cache_x_, *ctx_, input.shape());
  ensure_activation(output_, *ctx_, input.shape());
  if (output_->dtype() == DType::BFloat16) {
    arena_reset(*ctx_);
    float* x = stage_f32(*ctx_, input);
    float* y = arena_f32(*ctx_, input.numel());
    launch_relu(*ctx_, y, x, input.numel());
    keep_f32(*ctx_, *cache_x_, x);
    keep_f32(*ctx_, *output_, y);
    return *output_;
  }
  cache_x_->copy_from(input);
  launch_relu(*ctx_, output_->data(), cache_x_->data(), cache_x_->numel());
  return *output_;
}

const Tensor& ReLU::backward(const Tensor& grad_output) {
  if (!cache_x_ || !output_ || grad_output.shape() != output_->shape()) {
    throw Error("ReLU backward shape");
  }
  ensure_activation(grad_input_, *ctx_, grad_output.shape());
  if (grad_input_->dtype() == DType::BFloat16) {
    arena_reset(*ctx_);
    float* x = stage_f32(*ctx_, *cache_x_);
    float* dy = stage_f32(*ctx_, grad_output);
    float* dx = arena_f32(*ctx_, cache_x_->numel());
    launch_relu_backward(*ctx_, dx, x, dy, cache_x_->numel());
    keep_f32(*ctx_, *grad_input_, dx);
    return *grad_input_;
  }
  launch_relu_backward(*ctx_, grad_input_->data(), cache_x_->data(), grad_output.data(),
                       cache_x_->numel());
  return *grad_input_;
}

}  // namespace nn
