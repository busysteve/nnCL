#pragma once

#include "nn/layer.hpp"

#include <memory>
#include <optional>

namespace nn {

class ReLU : public Layer {
 public:
  explicit ReLU(Context& ctx);
  ~ReLU() override;
  const Tensor& forward(const Tensor& input) override;
  const Tensor& backward(const Tensor& grad_output) override;
  std::vector<NamedParam> parameters() const override;
  void zero_grad() override;

 private:
  struct Impl;
  Context* ctx_;
  std::unique_ptr<Impl> impl_;
  std::optional<Tensor> cache_x_;
  std::optional<Tensor> output_;
  std::optional<Tensor> grad_input_;
};

}  // namespace nn
