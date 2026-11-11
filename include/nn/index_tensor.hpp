#pragma once

#include "nn/context.hpp"

#include <cstdint>
#include <vector>

namespace nn {

class IndexTensor {
 public:
  IndexTensor(Context& ctx, std::vector<int> shape);
  ~IndexTensor();
  IndexTensor(IndexTensor&& other) noexcept;
  IndexTensor& operator=(IndexTensor&& other) noexcept;
  IndexTensor(const IndexTensor&) = delete;
  IndexTensor& operator=(const IndexTensor&) = delete;

  void copy_from_host(const int64_t* src, int64_t n);
  std::vector<int64_t> to_host() const;
  const int64_t* data() const { return data_; }
  const std::vector<int>& shape() const { return shape_; }
  int64_t numel() const { return numel_; }
  Context& context() const { return *ctx_; }

 private:
  void release() noexcept;

  Context* ctx_ = nullptr;
  std::vector<int> shape_;
  int64_t numel_ = 0;
  int64_t* data_ = nullptr;
};

}  // namespace nn
