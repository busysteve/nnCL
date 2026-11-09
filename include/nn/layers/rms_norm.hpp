#pragma once

#include "nn/layer.hpp"

#include <optional>

namespace nn {

class RMSNorm : public Layer {
 public:
  RMSNorm(Context& ctx, int normalized_shape, float eps = 1e-5f);
  const Tensor& forward(const Tensor& input) override;
  const Tensor& backward(const Tensor& grad_output) override;
  std::vector<NamedParam> parameters() const override;
  void zero_grad() override;

 private:
  Context* ctx_;
  int normalized_shape_;
  float eps_;
  Tensor weight_;
  Tensor grad_weight_;
  std::optional<Tensor> cache_x_;
  std::optional<Tensor> inv_rms_;
  std::optional<Tensor> output_;
  std::optional<Tensor> grad_input_;
};

}  // namespace nn
