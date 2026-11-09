#pragma once

#include "nn/layer.hpp"

#include <memory>
#include <optional>
#include <random>

namespace nn {

class Conv2d;
class BatchNorm2d;
class ReLU;

class ResidualBlock : public Layer {
 public:
  ResidualBlock(Context& ctx, int in_channels, int out_channels, int stride, std::mt19937& rng);
  ~ResidualBlock() override;
  const Tensor& forward(const Tensor& input) override;
  const Tensor& backward(const Tensor& grad_output) override;
  std::vector<NamedParam> parameters() const override;
  std::vector<NamedBuffer> buffers() const override;
  void zero_grad() override;
  void set_training(bool training) override;

 private:
  struct Impl;
  Context* ctx_;
  bool project_;
  std::unique_ptr<Conv2d> conv1_;
  std::unique_ptr<BatchNorm2d> bn1_;
  std::unique_ptr<ReLU> relu1_;
  std::unique_ptr<Conv2d> conv2_;
  std::unique_ptr<BatchNorm2d> bn2_;
  std::unique_ptr<Conv2d> proj_;
  std::unique_ptr<BatchNorm2d> proj_bn_;
  std::unique_ptr<ReLU> relu_out_;
  std::unique_ptr<Impl> impl_;
  std::optional<Tensor> sum_;
  std::optional<Tensor> grad_input_;
};

}  // namespace nn
