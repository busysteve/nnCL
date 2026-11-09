#pragma once

#include "nn/layer.hpp"

#include <optional>
#include <random>

namespace nn {

class Dropout : public Layer {
 public:
  Dropout(Context& ctx, float p);
  const Tensor& forward(const Tensor& input) override;
  const Tensor& backward(const Tensor& grad_output) override;
  std::vector<NamedParam> parameters() const override;
  void zero_grad() override;
  void set_training(bool training) override;

 private:
  Context* ctx_;
  float p_;
  bool training_ = true;
  bool has_forward_ = false;
  bool dropped_ = false;
  std::mt19937 rng_;
  std::optional<Tensor> mask_;
  std::optional<Tensor> output_;
  std::optional<Tensor> grad_input_;
};

}  // namespace nn
