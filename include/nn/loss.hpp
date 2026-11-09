#pragma once

#include "nn/context.hpp"
#include "nn/index_tensor.hpp"
#include "nn/tensor.hpp"

#include <cstdint>

namespace nn {

class DeviceIndices {
 public:
  DeviceIndices(Context& ctx, int64_t n);
  ~DeviceIndices();
  DeviceIndices(DeviceIndices&&) noexcept;
  DeviceIndices& operator=(DeviceIndices&&) noexcept;
  DeviceIndices(const DeviceIndices&) = delete;
  DeviceIndices& operator=(const DeviceIndices&) = delete;

  void copy_from_host(const int64_t* src, int64_t n);
  const int64_t* data() const { return data_; }
  int64_t size() const { return n_; }
  Context& context() const { return *ctx_; }

 private:
  Context* ctx_ = nullptr;
  int64_t* data_ = nullptr;
  int64_t n_ = 0;
};

struct LossResult {
  float loss = 0.f;
  Tensor grad_logits;
};

LossResult softmax_cross_entropy(Context& ctx, const Tensor& logits, const DeviceIndices& labels);

LossResult sequence_cross_entropy(Context& ctx, const Tensor& logits, const IndexTensor& labels,
                                  int64_t ignore_index = -100);

}  // namespace nn
