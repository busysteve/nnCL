#include "nn/layers/relu.hpp"

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
  ensure_tensor(cache_x_, *ctx_, input.shape());
  ensure_tensor(output_, *ctx_, input.shape());
  cache_x_->copy_from(input);
  launch_relu(*ctx_, output_->data(), cache_x_->data(), cache_x_->numel());
  return *output_;
}

const Tensor& ReLU::backward(const Tensor& grad_output) {
  if (!cache_x_ || !output_ || grad_output.shape() != output_->shape()) {
    throw Error("ReLU backward shape");
  }
  ensure_tensor(grad_input_, *ctx_, grad_output.shape());
  launch_relu_backward(*ctx_, grad_input_->data(), cache_x_->data(), grad_output.data(),
                       cache_x_->numel());
  return *grad_input_;
}

}  // namespace nn
