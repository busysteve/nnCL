#pragma once

#include "nn/context.hpp"

#include <cstdint>
#include <vector>

namespace nn {

class Tensor {
 public:
  Tensor(Context& ctx, std::vector<int> shape);
  ~Tensor();
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&& other) noexcept;
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  Context& context() const { return *ctx_; }
  int rank() const { return static_cast<int>(shape_.size()); }
  const std::vector<int>& shape() const { return shape_; }
  int64_t numel() const { return numel_; }
  float* data() { return data_; }
  const float* data() const { return data_; }

  void zero();
  void fill(float value);
  void scale(float factor);
  void copy_from(const Tensor& src);
  void copy_from_host(const float* src, int64_t n);
  void copy_from_host(const std::vector<float>& src);
  std::vector<float> to_host() const;

  Tensor reshape(const std::vector<int>& shape) const;

 private:
  void release() noexcept;

  Context* ctx_ = nullptr;
  std::vector<int> shape_;
  int64_t numel_ = 0;
  float* data_ = nullptr;
  size_t bytes_ = 0;
};

}  // namespace nn
