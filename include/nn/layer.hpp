#pragma once

#include "nn/tensor.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace nn {

struct NamedParam {
  std::string name;
  Tensor* value = nullptr;
  Tensor* grad = nullptr;
};

struct NamedBuffer {
  std::string name;
  Tensor* value = nullptr;
  uint64_t* batches = nullptr;
};

class Layer {
 public:
  virtual ~Layer() = default;
  virtual const Tensor& forward(const Tensor& input) = 0;
  virtual const Tensor& backward(const Tensor& grad_output) = 0;
  virtual std::vector<NamedParam> parameters() const = 0;
  virtual std::vector<NamedBuffer> buffers() const { return {}; }
  virtual void zero_grad() = 0;
  virtual void set_training(bool) {}

  const std::string& name() const { return name_; }
  void set_name(std::string name) { name_ = std::move(name); }

 private:
  std::string name_;
};

}  // namespace nn
