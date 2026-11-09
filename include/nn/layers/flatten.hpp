#pragma once

#include "nn/layer.hpp"

#include <optional>

namespace nn {

class Flatten : public Layer {
 public:
  const Tensor& forward(const Tensor& input) override;
  const Tensor& backward(const Tensor& grad_output) override;
  std::vector<NamedParam> parameters() const override;
  void zero_grad() override;

 private:
  int n_ = 0;
  int c_ = 0;
  int h_ = 0;
  int w_ = 0;
  std::optional<Tensor> output_;
  std::optional<Tensor> grad_input_;
};

}  // namespace nn
