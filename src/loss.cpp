#include "nn/loss.hpp"

#include "nn/error.hpp"
#include "ocl.hpp"

namespace nn {
namespace {

struct Mem {
  cl_mem raw = nullptr;
  ~Mem() { free_mem(raw); }
};

}  // namespace

DeviceIndices::DeviceIndices(Context& ctx, int64_t n) : ctx_(&ctx), n_(n) {
  if (n <= 0) throw Error("empty labels");
  data_ = as_i64(alloc_mem(ctx, static_cast<size_t>(n) * sizeof(int64_t)));
}

DeviceIndices::~DeviceIndices() {
  if (!data_) return;
  if (ctx_) {
    cl_command_queue queue = ContextAccess::get(*ctx_).queue;
    if (queue) clFinish(queue);
  }
  free_mem(mem_of(data_));
  data_ = nullptr;
}

DeviceIndices::DeviceIndices(DeviceIndices&& other) noexcept
    : ctx_(other.ctx_), data_(other.data_), n_(other.n_) {
  other.ctx_ = nullptr;
  other.data_ = nullptr;
  other.n_ = 0;
}

DeviceIndices& DeviceIndices::operator=(DeviceIndices&& other) noexcept {
  if (this == &other) return *this;
  if (data_) {
    if (ctx_) {
      cl_command_queue queue = ContextAccess::get(*ctx_).queue;
      if (queue) clFinish(queue);
    }
    free_mem(mem_of(data_));
  }
  ctx_ = other.ctx_;
  data_ = other.data_;
  n_ = other.n_;
  other.ctx_ = nullptr;
  other.data_ = nullptr;
  other.n_ = 0;
  return *this;
}

void DeviceIndices::copy_from_host(const int64_t* src, int64_t n) {
  if (n != n_) throw Error("label count mismatch");
  write_mem(*ctx_, mem_of(data_), src, static_cast<size_t>(n) * sizeof(int64_t));
}

LossResult softmax_cross_entropy(Context& ctx, const Tensor& logits, const DeviceIndices& labels) {
  if (logits.rank() != 2) throw Error("logits rank");
  const int n = logits.shape()[0];
  const int classes = logits.shape()[1];
  if (n <= 0 || classes <= 1) throw Error("logits shape");
  if (labels.size() != n) throw Error("label count mismatch");

  LossResult result{0.f, Tensor(ctx, {n, classes})};
  Mem per_sample;
  Mem bad;
  per_sample.raw = alloc_mem(ctx, static_cast<size_t>(n) * sizeof(float));
  bad.raw = alloc_mem(ctx, sizeof(int));
  fill_mem_i(ctx, bad.raw, 0, 1);
  launch_softmax_ce(ctx, logits.data(), labels.data(), result.grad_logits.data(),
                    as_float(per_sample.raw), n, classes, as_i32(bad.raw));
  const float sum = launch_reduce_sum(ctx, as_float(per_sample.raw), n);
  int bad_host = 0;
  read_mem(ctx, bad.raw, &bad_host, sizeof(bad_host));
  if (bad_host) throw Error("label out of range");
  result.loss = sum / static_cast<float>(n);
  return result;
}

LossResult sequence_cross_entropy(Context& ctx, const Tensor& logits, const IndexTensor& labels,
                                  int64_t ignore_index) {
  int classes = 0;
  int rows = 0;
  if (logits.rank() == 2) {
    rows = logits.shape()[0];
    classes = logits.shape()[1];
    if (labels.shape().size() != 1 || labels.shape()[0] != rows) throw Error("label shape");
  } else if (logits.rank() == 3) {
    if (labels.shape().size() != 2 || labels.shape()[0] != logits.shape()[0] ||
        labels.shape()[1] != logits.shape()[1]) {
      throw Error("label shape");
    }
    classes = logits.shape()[2];
    rows = static_cast<int>(static_cast<int64_t>(logits.shape()[0]) * logits.shape()[1]);
  } else {
    throw Error("logits rank");
  }
  if (classes <= 1) throw Error("logits shape");

  LossResult result{0.f, Tensor(ctx, logits.shape())};
  Mem bad;
  Mem n_valid_mem;
  bad.raw = alloc_mem(ctx, sizeof(int));
  n_valid_mem.raw = alloc_mem(ctx, sizeof(int));
  fill_mem_i(ctx, bad.raw, 0, 1);
  fill_mem_i(ctx, n_valid_mem.raw, 0, 1);
  launch_sequence_label_count(ctx, labels.data(), rows, classes, ignore_index, as_i32(n_valid_mem.raw),
                              as_i32(bad.raw));
  int bad_host = 0;
  int n_valid = 0;
  read_mem(ctx, bad.raw, &bad_host, sizeof(bad_host));
  read_mem(ctx, n_valid_mem.raw, &n_valid, sizeof(n_valid));
  if (bad_host) throw Error("label out of range");
  if (n_valid == 0) {
    result.grad_logits.zero();
    ctx.synchronize();
    return result;
  }

  Mem per_sample;
  per_sample.raw = alloc_mem(ctx, static_cast<size_t>(rows) * sizeof(float));
  launch_sequence_ce(ctx, logits.data(), labels.data(), result.grad_logits.data(),
                     as_float(per_sample.raw), rows, classes, ignore_index, n_valid, as_i32(bad.raw));
  const float sum = launch_reduce_sum(ctx, as_float(per_sample.raw), rows);
  read_mem(ctx, bad.raw, &bad_host, sizeof(bad_host));
  if (bad_host) throw Error("label out of range");
  result.loss = sum / static_cast<float>(n_valid);
  return result;
}

}  // namespace nn
