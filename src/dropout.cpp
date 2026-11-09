#include "nn/layers/dropout.hpp"

#include "ensure.hpp"
#include "nn/error.hpp"
#include "ocl.hpp"

#include <cmath>
#include <vector>

namespace nn {

Dropout::Dropout(Context& ctx, float p) : ctx_(&ctx), p_(p) {
  if (!(p >= 0.f) || !(p < 1.f) || !std::isfinite(p)) throw Error("Dropout p");
  rng_.seed(std::random_device{}());
}

std::vector<NamedParam> Dropout::parameters() const { return {}; }

void Dropout::zero_grad() {}

void Dropout::set_training(bool training) { training_ = training; }

const Tensor& Dropout::forward(const Tensor& input) {
  ensure_tensor(output_, *ctx_, input.shape());
  const bool drop = training_ && p_ > 0.f;
  if (!drop) {
    output_->copy_from(input);
    dropped_ = false;
    has_forward_ = true;
    return *output_;
  }

  ensure_tensor(mask_, *ctx_, input.shape());
  {
    std::vector<float> host(static_cast<size_t>(input.numel()));
    std::bernoulli_distribution keep(static_cast<double>(1.f - p_));
    const float scale = 1.f / (1.f - p_);
    for (float& value : host) value = keep(rng_) ? scale : 0.f;
    mask_->copy_from_host(host);
    // copy_from_host is async. The host mask must stay alive until the upload finishes.
    ctx_->synchronize();
  }
  launch_mul(*ctx_, output_->data(), input.data(), mask_->data(), input.numel());
  dropped_ = true;
  has_forward_ = true;
  return *output_;
}

const Tensor& Dropout::backward(const Tensor& grad_output) {
  if (!has_forward_ || !output_ || grad_output.shape() != output_->shape()) {
    throw Error("Dropout backward shape");
  }
  ensure_tensor(grad_input_, *ctx_, grad_output.shape());
  if (!dropped_) {
    grad_input_->copy_from(grad_output);
    return *grad_input_;
  }
  launch_mul(*ctx_, grad_input_->data(), grad_output.data(), mask_->data(), grad_output.numel());
  return *grad_input_;
}

}  // namespace nn
