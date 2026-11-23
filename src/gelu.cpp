#include "nn/layers/gelu.hpp"

#include "activate.hpp"
#include "ensure.hpp"
#include "nn/error.hpp"
#include "ocl.hpp"

namespace nn {

GELU::GELU(Context& ctx) : ctx_(&ctx) {}

std::vector<NamedParam> GELU::parameters() const { return {}; }

void GELU::zero_grad() {}

const Tensor& GELU::forward(const Tensor& input) {
  const std::vector<int>& shape = input.shape();
  if (shape.size() != 2 && shape.size() != 3 && shape.size() != 4) throw Error("GELU rank");
  ensure_activation(cache_x_, *ctx_, shape);
  ensure_activation(output_, *ctx_, shape);
  if (output_->dtype() == DType::BFloat16) {
    arena_reset(*ctx_);
    float* x = stage_f32(*ctx_, input);
    float* y = arena_f32(*ctx_, input.numel());
    launch_gelu(*ctx_, y, x, input.numel());
    keep_f32(*ctx_, *cache_x_, x);
    keep_f32(*ctx_, *output_, y);
    return *output_;
  }
  cache_x_->copy_from(input);
  launch_gelu(*ctx_, output_->data(), cache_x_->data(), cache_x_->numel());
  return *output_;
}

const Tensor& GELU::backward(const Tensor& grad_output) {
  if (!cache_x_ || grad_output.shape() != cache_x_->shape()) throw Error("GELU backward shape");
  ensure_activation(grad_input_, *ctx_, cache_x_->shape());
  if (grad_input_->dtype() == DType::BFloat16) {
    arena_reset(*ctx_);
    float* x = stage_f32(*ctx_, *cache_x_);
    float* dy = stage_f32(*ctx_, grad_output);
    float* dx = arena_f32(*ctx_, cache_x_->numel());
    launch_gelu_backward(*ctx_, dx, x, dy, cache_x_->numel());
    keep_f32(*ctx_, *grad_input_, dx);
    return *grad_input_;
  }
  launch_gelu_backward(*ctx_, grad_input_->data(), cache_x_->data(), grad_output.data(),
                       cache_x_->numel());
  return *grad_input_;
}

}  // namespace nn
