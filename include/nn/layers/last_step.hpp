#pragma once

#include "nn/layer.hpp"

#include <optional>

namespace nn {

class LastStep : public Layer {
 public:
  const Tensor& forward(const Tensor& input) override;
  const Tensor& backward(const Tensor& grad_output) override;
  std::vector<NamedParam> parameters() const override;
  void zero_grad() override;

 private:
  int n_ = 0;
  int t_ = 0;
  int f_ = 0;
  std::optional<Tensor> output_;
  std::optional<Tensor> grad_input_;
};

}  // namespace nn
