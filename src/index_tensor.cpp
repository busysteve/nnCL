#include "nn/index_tensor.hpp"

#include "nn/error.hpp"
#include "ocl.hpp"

#include <climits>
#include <utility>

namespace nn {
namespace {

int64_t checked_numel(const std::vector<int>& shape) {
  if (shape.size() != 1 && shape.size() != 2) throw Error("index tensor rank out of range");
  int64_t n = 1;
  for (int dim : shape) {
    if (dim < 1) throw Error("index tensor dimension out of range");
    if (n > static_cast<int64_t>(INT32_MAX) / dim) throw Error("index tensor numel out of range");
    n *= dim;
  }
  return n;
}

}  // namespace

IndexTensor::IndexTensor(Context& ctx, std::vector<int> shape)
    : ctx_(&ctx), shape_(std::move(shape)) {
  numel_ = checked_numel(shape_);
  data_ = as_i64(alloc_mem(*ctx_, static_cast<size_t>(numel_) * sizeof(int64_t)));
}

void IndexTensor::release() noexcept {
  if (!data_) return;
  if (ctx_) {
    cl_command_queue queue = ContextAccess::get(*ctx_).queue;
    if (queue) clFinish(queue);
  }
  free_mem(mem_of(data_));
  data_ = nullptr;
  numel_ = 0;
}

IndexTensor::~IndexTensor() { release(); }

IndexTensor::IndexTensor(IndexTensor&& other) noexcept
    : ctx_(other.ctx_), shape_(std::move(other.shape_)), numel_(other.numel_), data_(other.data_) {
  other.ctx_ = nullptr;
  other.numel_ = 0;
  other.data_ = nullptr;
}

IndexTensor& IndexTensor::operator=(IndexTensor&& other) noexcept {
  if (this == &other) return *this;
  release();
  ctx_ = other.ctx_;
  shape_ = std::move(other.shape_);
  numel_ = other.numel_;
  data_ = other.data_;
  other.ctx_ = nullptr;
  other.numel_ = 0;
  other.data_ = nullptr;
  return *this;
}

void IndexTensor::copy_from_host(const int64_t* src, int64_t n) {
  if (n != numel_) throw Error("index tensor host copy numel mismatch");
  write_mem(*ctx_, mem_of(data_), src, static_cast<size_t>(n) * sizeof(int64_t));
}

std::vector<int64_t> IndexTensor::to_host() const {
  ctx_->synchronize();
  std::vector<int64_t> host(static_cast<size_t>(numel_));
  read_mem(*ctx_, mem_of(data_), host.data(), host.size() * sizeof(int64_t));
  return host;
}

}  // namespace nn
