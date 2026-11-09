#pragma once

#include "nn/layer.hpp"

#include <cstdint>
#include <memory>
#include <optional>

namespace nn {

class BatchNorm2d : public Layer {
 public:
  BatchNorm2d(Context& ctx, int channels);
  ~BatchNorm2d() override;
  const Tensor& forward(const Tensor& input) override;
  const Tensor& backward(const Tensor& grad_output) override;
  std::vector<NamedParam> parameters() const override;
  std::vector<NamedBuffer> buffers() const override;
  void zero_grad() override;
  void set_training(bool training) override;

 private:
  struct Impl;
  void bind(const Tensor& input);

  Context* ctx_;
  int channels_;
  bool training_ = true;
  uint64_t batches_ = 0;
  Tensor weight_;
  Tensor grad_weight_;
  Tensor bias_;
  Tensor grad_bias_;
  Tensor running_mean_;
  Tensor running_var_;
  Tensor saved_mean_;
  Tensor saved_inv_variance_;
  std::unique_ptr<Impl> impl_;
  std::optional<Tensor> cache_x_;
  std::optional<Tensor> output_;
  std::optional<Tensor> grad_input_;
};

}  // namespace nn
