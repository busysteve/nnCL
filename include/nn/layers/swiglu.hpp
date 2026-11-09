#pragma once

#include "nn/layer.hpp"

#include <optional>

namespace nn {

class SwiGLU : public Layer {
 public:
  explicit SwiGLU(Context& ctx);
  const Tensor& forward(const Tensor& input) override;
  const Tensor& backward(const Tensor& grad_output) override;
  std::vector<NamedParam> parameters() const override;
  void zero_grad() override;

 private:
  Context* ctx_;
  std::optional<Tensor> cache_x_;
  std::optional<Tensor> output_;
  std::optional<Tensor> grad_input_;
};

}  // namespace nn
