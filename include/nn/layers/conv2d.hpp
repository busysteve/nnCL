#pragma once

#include "nn/layer.hpp"

#include <memory>
#include <optional>
#include <random>

namespace nn {

class Conv2d : public Layer {
 public:
  Conv2d(Context& ctx, int in_channels, int out_channels, int kernel, int stride, int pad,
         bool use_bias, std::mt19937& rng);
  ~Conv2d() override;
  const Tensor& forward(const Tensor& input) override;
  const Tensor& backward(const Tensor& grad_output) override;
  std::vector<NamedParam> parameters() const override;
  void zero_grad() override;

 private:
  struct Impl;
  void prepare(int n, int h, int w);

  Context* ctx_;
  int in_channels_;
  int out_channels_;
  int kernel_;
  int stride_;
  int pad_;
  bool use_bias_;
  Tensor weight_;
  Tensor grad_weight_;
  std::optional<Tensor> bias_;
  std::optional<Tensor> grad_bias_;
  std::unique_ptr<Impl> impl_;
  std::optional<Tensor> cache_x_;
  std::optional<Tensor> output_;
  std::optional<Tensor> grad_input_;
  void* workspace_ = nullptr;
  size_t workspace_bytes_ = 0;
};

}  // namespace nn
