#include "nn/tensor.hpp"

#include "nn/error.hpp"
#include "ocl.hpp"

#include <climits>
#include <utility>

namespace nn {
namespace {

int64_t checked_numel(const std::vector<int>& shape) {
  if (shape.empty() || shape.size() > 4) throw Error("tensor rank out of range");
  int64_t n = 1;
  for (int dim : shape) {
    if (dim < 1) throw Error("tensor dimension out of range");
    if (n > static_cast<int64_t>(INT32_MAX) / dim) throw Error("tensor numel out of range");
    n *= dim;
  }
  return n;
}

}  // namespace

Tensor::Tensor(Context& ctx, std::vector<int> shape) : ctx_(&ctx), shape_(std::move(shape)) {
  numel_ = checked_numel(shape_);
  bytes_ = static_cast<size_t>(numel_) * sizeof(float);
  data_ = as_float(alloc_mem(*ctx_, bytes_));
}

void Tensor::release() noexcept {
  if (!data_) return;
  if (ctx_) {
    cl_command_queue queue = ContextAccess::get(*ctx_).queue;
    if (queue) clFinish(queue);
  }
  free_mem(mem_of(data_));
  data_ = nullptr;
  bytes_ = 0;
  numel_ = 0;
}

Tensor::~Tensor() { release(); }

Tensor::Tensor(Tensor&& other) noexcept
    : ctx_(other.ctx_),
      shape_(std::move(other.shape_)),
      numel_(other.numel_),
      data_(other.data_),
      bytes_(other.bytes_) {
  other.ctx_ = nullptr;
  other.numel_ = 0;
  other.data_ = nullptr;
  other.bytes_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this == &other) return *this;
  release();
  ctx_ = other.ctx_;
  shape_ = std::move(other.shape_);
  numel_ = other.numel_;
  data_ = other.data_;
  bytes_ = other.bytes_;
  other.ctx_ = nullptr;
  other.numel_ = 0;
  other.data_ = nullptr;
  other.bytes_ = 0;
  return *this;
}

void Tensor::fill(float value) { fill_mem_f(*ctx_, mem_of(data_), value, static_cast<size_t>(numel_)); }

void Tensor::zero() { fill(0.f); }

void Tensor::scale(float factor) { launch_scale(*ctx_, data_, numel_, factor); }

void Tensor::copy_from(const Tensor& src) {
  if (numel_ != src.numel_) throw Error("tensor copy numel mismatch");
  if (src.data_ == data_) return;
  if (&src.context() != ctx_) {
    const std::vector<float> host = src.to_host();
    copy_from_host(host);
    return;
  }
  copy_mem(*ctx_, mem_of(data_), mem_of(src.data_), bytes_);
}

void Tensor::copy_from_host(const float* src, int64_t n) {
  if (n != numel_) throw Error("tensor host copy numel mismatch");
  write_mem(*ctx_, mem_of(data_), src, bytes_);
}

void Tensor::copy_from_host(const std::vector<float>& src) {
  copy_from_host(src.data(), static_cast<int64_t>(src.size()));
}

std::vector<float> Tensor::to_host() const {
  ctx_->synchronize();
  std::vector<float> host(static_cast<size_t>(numel_));
  read_mem(*ctx_, mem_of(data_), host.data(), bytes_);
  return host;
}

Tensor Tensor::reshape(const std::vector<int>& shape) const {
  if (checked_numel(shape) != numel_) throw Error("tensor reshape numel mismatch");
  Tensor out(*ctx_, shape);
  out.copy_from(*this);
  return out;
}

}  // namespace nn
