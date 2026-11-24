#include "nn/tensor.hpp"

#include "activate.hpp"
#include "nn/error.hpp"
#include "ocl.hpp"

#include <climits>
#include <utility>
#include <vector>

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

Tensor::Tensor(Context& ctx, std::vector<int> shape) : Tensor(ctx, std::move(shape), DType::Float32) {}

Tensor::Tensor(Context& ctx, std::vector<int> shape, DType dtype)
    : ctx_(&ctx), shape_(std::move(shape)), dtype_(dtype) {
  if (dtype_ != DType::Float32 && dtype_ != DType::BFloat16) throw Error("dtype");
  numel_ = checked_numel(shape_);
  bytes_ = static_cast<size_t>(numel_) * static_cast<size_t>(dtype_size(dtype_));
  storage_ = alloc_mem(*ctx_, bytes_);
}

void Tensor::allocate() {}

void Tensor::release() noexcept {
  if (!storage_) return;
  if (ctx_) {
    cl_command_queue queue = ContextAccess::get(*ctx_).queue;
    if (queue) clFinish(queue);
  }
  free_mem(static_cast<cl_mem>(storage_));
  storage_ = nullptr;
  bytes_ = 0;
  numel_ = 0;
}

Tensor::~Tensor() { release(); }

Tensor::Tensor(Tensor&& other) noexcept
    : ctx_(other.ctx_),
      shape_(std::move(other.shape_)),
      numel_(other.numel_),
      dtype_(other.dtype_),
      storage_(other.storage_),
      bytes_(other.bytes_) {
  other.ctx_ = nullptr;
  other.numel_ = 0;
  other.storage_ = nullptr;
  other.bytes_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this == &other) return *this;
  release();
  ctx_ = other.ctx_;
  shape_ = std::move(other.shape_);
  numel_ = other.numel_;
  dtype_ = other.dtype_;
  storage_ = other.storage_;
  bytes_ = other.bytes_;
  other.ctx_ = nullptr;
  other.numel_ = 0;
  other.storage_ = nullptr;
  other.bytes_ = 0;
  return *this;
}

float* Tensor::data() {
  if (dtype_ != DType::Float32) throw Error("tensor dtype");
  return static_cast<float*>(storage_);
}

const float* Tensor::data() const {
  if (dtype_ != DType::Float32) throw Error("tensor dtype");
  return static_cast<const float*>(storage_);
}

uint16_t* Tensor::bf16_data() {
  if (dtype_ != DType::BFloat16) throw Error("tensor dtype");
  return static_cast<uint16_t*>(storage_);
}

const uint16_t* Tensor::bf16_data() const {
  if (dtype_ != DType::BFloat16) throw Error("tensor dtype");
  return static_cast<const uint16_t*>(storage_);
}

void Tensor::fill(float value) {
  if (dtype_ == DType::Float32) {
    fill_mem_f(*ctx_, static_cast<cl_mem>(storage_), value, static_cast<size_t>(numel_));
    return;
  }
  if (value == 0.f) {
    zero();
    return;
  }
  std::vector<uint16_t> host(static_cast<size_t>(numel_), f32_to_bf16(value));
  write_mem(*ctx_, static_cast<cl_mem>(storage_), host.data(), bytes_);
  ctx_->synchronize();
}

void Tensor::zero() { zero_mem(*ctx_, static_cast<cl_mem>(storage_), bytes_); }

void Tensor::scale(float factor) {
  if (dtype_ == DType::Float32) {
    launch_scale(*ctx_, data(), numel_, factor);
    return;
  }
  arena_reset(*ctx_);
  float* tmp = arena_f32(*ctx_, numel_);
  cast_bf16_to_f32(*ctx_, tmp, bf16_data(), numel_);
  launch_scale(*ctx_, tmp, numel_, factor);
  cast_f32_to_bf16(*ctx_, bf16_data(), tmp, numel_);
}

void Tensor::copy_from(const Tensor& src) {
  if (numel_ != src.numel_) throw Error("tensor copy numel mismatch");
  if (storage_ == src.storage_) return;
  if (&src.context() != ctx_) {
    const std::vector<float> host = src.to_host();
    copy_from_host(host);
    return;
  }
  if (dtype_ == src.dtype_) {
    copy_mem(*ctx_, static_cast<cl_mem>(storage_), static_cast<cl_mem>(src.storage_), bytes_);
    return;
  }
  if (dtype_ == DType::BFloat16 && src.dtype_ == DType::Float32) {
    cast_f32_to_bf16(*ctx_, bf16_data(), src.data(), numel_);
    return;
  }
  if (dtype_ == DType::Float32 && src.dtype_ == DType::BFloat16) {
    cast_bf16_to_f32(*ctx_, data(), src.bf16_data(), numel_);
    return;
  }
  throw Error("tensor dtype");
}

void Tensor::copy_from_host(const float* src, int64_t n) {
  if (n != numel_) throw Error("tensor host copy numel mismatch");
  if (dtype_ == DType::Float32) {
    write_mem(*ctx_, static_cast<cl_mem>(storage_), src, bytes_);
    return;
  }
  std::vector<uint16_t> host(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) host[static_cast<size_t>(i)] = f32_to_bf16(src[i]);
  write_mem(*ctx_, static_cast<cl_mem>(storage_), host.data(), bytes_);
  ctx_->synchronize();
}

void Tensor::copy_from_host(const std::vector<float>& src) {
  copy_from_host(src.data(), static_cast<int64_t>(src.size()));
}

std::vector<float> Tensor::to_host() const {
  ctx_->synchronize();
  if (dtype_ == DType::Float32) {
    std::vector<float> host(static_cast<size_t>(numel_));
    read_mem(*ctx_, static_cast<cl_mem>(storage_), host.data(), bytes_);
    return host;
  }
  std::vector<uint16_t> bits(static_cast<size_t>(numel_));
  read_mem(*ctx_, static_cast<cl_mem>(storage_), bits.data(), bytes_);
  std::vector<float> host(bits.size());
  for (size_t i = 0; i < bits.size(); ++i) host[i] = bf16_to_f32(bits[i]);
  return host;
}

Tensor Tensor::reshape(const std::vector<int>& shape) const {
  if (checked_numel(shape) != numel_) throw Error("tensor reshape numel mismatch");
  Tensor out(*ctx_, shape, dtype_);
  out.copy_from(*this);
  return out;
}

}  // namespace nn
