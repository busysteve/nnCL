#pragma once

#include "nn/layer.hpp"

#include <memory>
#include <optional>

namespace nn {

class MaxPool2d : public Layer {
 public:
  MaxPool2d(Context& ctx, int kernel, int stride);
  ~MaxPool2d() override;
  const Tensor& forward(const Tensor& input) override;
  const Tensor& backward(const Tensor& grad_output) override;
  std::vector<NamedParam> parameters() const override;
  void zero_grad() override;

 private:
  struct Impl;
  Context* ctx_;
  int kernel_;
  int stride_;
  std::unique_ptr<Impl> impl_;
  std::optional<Tensor> cache_x_;
  std::optional<Tensor> cache_y_;
  std::optional<Tensor> output_;
  std::optional<Tensor> grad_input_;
};

}  // namespace nn
