#pragma once

#include "nn/layer.hpp"

#include <optional>
#include <random>

namespace nn {

class RNN : public Layer {
 public:
  RNN(Context& ctx, int input_size, int hidden_size, std::mt19937& rng);
  const Tensor& forward(const Tensor& input) override;
  const Tensor& backward(const Tensor& grad_output) override;
  std::vector<NamedParam> parameters() const override;
  void zero_grad() override;

 private:
  Context* ctx_;
  int input_size_;
  int hidden_size_;
  Tensor weight_ih_;
  Tensor grad_weight_ih_;
  Tensor weight_hh_;
  Tensor grad_weight_hh_;
  Tensor bias_ih_;
  Tensor grad_bias_ih_;
  Tensor bias_hh_;
  Tensor grad_bias_hh_;
  std::optional<Tensor> cache_x_;
  std::optional<Tensor> output_;
  std::optional<Tensor> grad_input_;
  std::optional<Tensor> x_t_;
  std::optional<Tensor> h_t_;
  std::optional<Tensor> h_prev_;
  std::optional<Tensor> pre_;
  std::optional<Tensor> grad_a_;
  std::optional<Tensor> grad_h_;
  std::optional<Tensor> grad_y_t_;
  std::optional<Tensor> grad_h_next_;
  std::optional<Tensor> grad_x_t_;
};

}  // namespace nn
