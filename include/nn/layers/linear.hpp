#pragma once

#include "nn/layer.hpp"

#include <optional>
#include <random>

namespace nn {

class Linear : public Layer {
 public:
  Linear(Context& ctx, int in_features, int out_features, std::mt19937& rng);
  Linear(Context& ctx, int in_features, int out_features, bool use_bias, std::mt19937& rng);
  const Tensor& forward(const Tensor& input) override;
  const Tensor& backward(const Tensor& grad_output) override;
  std::vector<NamedParam> parameters() const override;
  void zero_grad() override;

 private:
  Context* ctx_;
  int in_features_;
  int out_features_;
  bool use_bias_;
  Tensor weight_;
  Tensor grad_weight_;
  std::optional<Tensor> bias_;
  std::optional<Tensor> grad_bias_;
  std::optional<Tensor> cache_x_;
  std::optional<Tensor> output_;
  std::optional<Tensor> grad_input_;
};

}  // namespace nn
