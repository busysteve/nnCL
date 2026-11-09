#include "nn/layers/swiglu.hpp"

#include "ensure.hpp"
#include "nn/error.hpp"
#include "ocl.hpp"

#include <cstdint>

namespace nn {
namespace {

struct Split {
  int rows = 0;
  int64_t inner = 0;
  std::vector<int> output_shape;
};

Split split_features(const std::vector<int>& shape) {
  if (shape.size() != 2 && shape.size() != 3 && shape.size() != 4) throw Error("SwiGLU rank");
  Split split;
  split.output_shape = shape;
  if (shape.size() == 3) {
    // Last dimension, so each (n, t) row is [gate | value]. Rank 2/4 stay on axis 1.
    if ((shape.back() & 1) != 0) throw Error("SwiGLU size");
    split.output_shape.back() /= 2;
    split.rows = static_cast<int>(static_cast<int64_t>(shape[0]) * shape[1]);
    split.inner = split.output_shape.back();
    return split;
  }
  if ((shape[1] & 1) != 0) throw Error("SwiGLU size");
  split.rows = shape[0];
  split.output_shape[1] /= 2;
  split.inner = 1;
  for (size_t axis = 1; axis < split.output_shape.size(); ++axis) {
    split.inner *= split.output_shape[axis];
  }
  return split;
}

}  // namespace

SwiGLU::SwiGLU(Context& ctx) : ctx_(&ctx) {}

std::vector<NamedParam> SwiGLU::parameters() const { return {}; }

void SwiGLU::zero_grad() {}

const Tensor& SwiGLU::forward(const Tensor& input) {
  const Split split = split_features(input.shape());
  ensure_tensor(cache_x_, *ctx_, input.shape());
  ensure_tensor(output_, *ctx_, split.output_shape);
  cache_x_->copy_from(input);
  launch_swiglu(*ctx_, output_->data(), cache_x_->data(), split.rows, split.inner);
  return *output_;
}

const Tensor& SwiGLU::backward(const Tensor& grad_output) {
  if (!cache_x_ || !output_ || grad_output.shape() != output_->shape()) {
    throw Error("SwiGLU backward shape");
  }
  const Split split = split_features(cache_x_->shape());
  ensure_tensor(grad_input_, *ctx_, cache_x_->shape());
  launch_swiglu_backward(*ctx_, grad_input_->data(), cache_x_->data(), grad_output.data(),
                         split.rows, split.inner);
  return *grad_input_;
}

}  // namespace nn
