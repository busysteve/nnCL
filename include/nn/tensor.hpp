#pragma once

#include "nn/context.hpp"

#include <cstdint>
#include <vector>

namespace nn {

class Tensor {
 public:
  Tensor(Context& ctx, std::vector<int> shape);
  Tensor(Context& ctx, std::vector<int> shape, DType dtype);
  ~Tensor();
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&& other) noexcept;
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  Context& context() const { return *ctx_; }
  int rank() const { return static_cast<int>(shape_.size()); }
  const std::vector<int>& shape() const { return shape_; }
  int64_t numel() const { return numel_; }
  DType dtype() const { return dtype_; }
  size_t bytes() const { return bytes_; }
  // Float32 storage. Throws tensor dtype when the tensor is BFloat16.
  float* data();
  const float* data() const;
  // BFloat16 storage. Throws tensor dtype when the tensor is Float32.
  uint16_t* bf16_data();
  const uint16_t* bf16_data() const;

  void zero();
  void fill(float value);
  void scale(float factor);
  void copy_from(const Tensor& src);
  void copy_from_host(const float* src, int64_t n);
  void copy_from_host(const std::vector<float>& src);
  // Widens BFloat16 values to float.
  std::vector<float> to_host() const;

  Tensor reshape(const std::vector<int>& shape) const;

 private:
  void release() noexcept;
  void allocate();

  Context* ctx_ = nullptr;
  std::vector<int> shape_;
  int64_t numel_ = 0;
  DType dtype_ = DType::Float32;
  void* storage_ = nullptr;
  size_t bytes_ = 0;
};

}  // namespace nn
