#include "nn/layers/layer_norm.hpp"

#include "ensure.hpp"
#include "nn/error.hpp"
#include "ocl.hpp"

#include <cmath>
#include <vector>

namespace nn {
namespace {

int normalized_rows(const Tensor& input, int normalized_shape) {
  const std::vector<int>& shape = input.shape();
  if ((shape.size() != 2 && shape.size() != 3) || shape.back() != normalized_shape) {
    throw Error("LayerNorm input shape");
  }
  return static_cast<int>(input.numel() / normalized_shape);
}

Context& require_layer_norm(Context& ctx, int normalized_shape, float eps) {
  if (normalized_shape < 1) throw Error("LayerNorm size");
  if (!(eps > 0.f) || !std::isfinite(eps)) throw Error("LayerNorm eps");
  return ctx;
}

}  // namespace

LayerNorm::LayerNorm(Context& ctx, int normalized_shape, float eps)
    : ctx_(&ctx),
      normalized_shape_(normalized_shape),
      eps_(eps),
      weight_(require_layer_norm(ctx, normalized_shape, eps), {normalized_shape}),
      grad_weight_(ctx, {normalized_shape}),
      bias_(ctx, {normalized_shape}),
      grad_bias_(ctx, {normalized_shape}) {
  weight_.fill(1.f);
  bias_.zero();
  grad_weight_.zero();
  grad_bias_.zero();
}

std::vector<NamedParam> LayerNorm::parameters() const {
  return {NamedParam{"weight", const_cast<Tensor*>(&weight_), const_cast<Tensor*>(&grad_weight_)},
          NamedParam{"bias", const_cast<Tensor*>(&bias_), const_cast<Tensor*>(&grad_bias_)}};
}

void LayerNorm::zero_grad() {
  grad_weight_.zero();
  grad_bias_.zero();
}

const Tensor& LayerNorm::forward(const Tensor& input) {
  const int rows = normalized_rows(input, normalized_shape_);
  const std::vector<int>& shape = input.shape();
  ensure_tensor(cache_x_, *ctx_, shape);
  ensure_tensor(output_, *ctx_, shape);
  ensure_tensor(mean_, *ctx_, std::vector<int>{rows});
  ensure_tensor(inv_std_, *ctx_, std::vector<int>{rows});
  cache_x_->copy_from(input);
  launch_layer_norm_forward(*ctx_, output_->data(), mean_->data(), inv_std_->data(), cache_x_->data(),
                            weight_.data(), bias_.data(), rows, normalized_shape_, eps_);
  return *output_;
}

const Tensor& LayerNorm::backward(const Tensor& grad_output) {
  if (!cache_x_ || !mean_ || !inv_std_ || grad_output.shape() != cache_x_->shape()) {
    throw Error("LayerNorm backward shape");
  }
  const int rows = static_cast<int>(mean_->numel());
  ensure_tensor(grad_input_, *ctx_, cache_x_->shape());
  launch_layer_norm_backward_input(*ctx_, grad_input_->data(), grad_output.data(), cache_x_->data(),
                                   mean_->data(), inv_std_->data(), weight_.data(), rows,
                                   normalized_shape_);
  launch_layer_norm_param_grad(*ctx_, grad_output.data(), cache_x_->data(), mean_->data(),
                               inv_std_->data(), grad_weight_.data(), grad_bias_.data(), rows,
                               normalized_shape_);
  return *grad_input_;
}

}  // namespace nn
