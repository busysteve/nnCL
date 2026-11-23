#include "nn/layers/max_pool2d.hpp"

#include "activate.hpp"
#include "ensure.hpp"
#include "nn/config.hpp"
#include "nn/error.hpp"
#include "ocl.hpp"

namespace nn {

struct MaxPool2d::Impl {
  cl_mem index = nullptr;
  int64_t count = 0;
};

MaxPool2d::MaxPool2d(Context& ctx, int kernel, int stride)
    : ctx_(&ctx), kernel_(kernel), stride_(stride), impl_(std::make_unique<Impl>()) {
  conv_out_size(kernel, kernel, stride, 0);
}

MaxPool2d::~MaxPool2d() {
  if (impl_->index != nullptr) {
    clFinish(ContextAccess::get(*ctx_).queue);
    free_mem(impl_->index);
    impl_->index = nullptr;
  }
}

std::vector<NamedParam> MaxPool2d::parameters() const { return {}; }

void MaxPool2d::zero_grad() {}

const Tensor& MaxPool2d::forward(const Tensor& input) {
  if (input.rank() != 4) throw Error("MaxPool2d input rank");
  const int n = input.shape()[0];
  const int c = input.shape()[1];
  const int h = input.shape()[2];
  const int w = input.shape()[3];
  const int ho = conv_out_size(h, kernel_, stride_, 0);
  const int wo = conv_out_size(w, kernel_, stride_, 0);
  ensure_activation(cache_x_, *ctx_, input.shape());
  ensure_activation(output_, *ctx_, {n, c, ho, wo});
  ensure_activation(cache_y_, *ctx_, output_->shape());
  const int64_t count = output_->numel();
  if (impl_->index == nullptr || impl_->count != count) {
    if (impl_->index != nullptr) {
      clFinish(ContextAccess::get(*ctx_).queue);
      free_mem(impl_->index);
      impl_->index = nullptr;
    }
    impl_->index = alloc_mem(*ctx_, static_cast<size_t>(count) * sizeof(int));
    impl_->count = count;
  }
  if (output_->dtype() == DType::BFloat16) {
    arena_reset(*ctx_);
    float* x = stage_f32(*ctx_, input);
    float* y = arena_f32(*ctx_, count);
    launch_max_pool_forward(*ctx_, y, as_i32(impl_->index), x, n, c, h, w, kernel_, ho, wo, stride_);
    keep_f32(*ctx_, *cache_x_, x);
    keep_f32(*ctx_, *output_, y);
    cache_y_->copy_from(*output_);
    return *output_;
  }
  cache_x_->copy_from(input);
  launch_max_pool_forward(*ctx_, output_->data(), as_i32(impl_->index), input.data(), n, c, h, w,
                          kernel_, ho, wo, stride_);
  cache_y_->copy_from(*output_);
  return *output_;
}

const Tensor& MaxPool2d::backward(const Tensor& grad_output) {
  if (!cache_x_ || !cache_y_ || grad_output.shape() != cache_y_->shape()) {
    throw Error("MaxPool2d backward shape");
  }
  ensure_activation(grad_input_, *ctx_, cache_x_->shape());
  if (grad_input_->dtype() == DType::BFloat16) {
    arena_reset(*ctx_);
    float* dy = stage_f32(*ctx_, grad_output);
    float* dx = arena_f32(*ctx_, grad_input_->numel());
    zero_f32(*ctx_, dx, grad_input_->numel());
    launch_max_pool_backward(*ctx_, dx, dy, as_i32(impl_->index), cache_x_->shape()[0],
                             cache_x_->shape()[1], cache_x_->shape()[2], cache_x_->shape()[3],
                             cache_y_->shape()[2], cache_y_->shape()[3]);
    keep_f32(*ctx_, *grad_input_, dx);
    return *grad_input_;
  }
  grad_input_->zero();
  launch_max_pool_backward(*ctx_, grad_input_->data(), grad_output.data(), as_i32(impl_->index),
                           cache_x_->shape()[0], cache_x_->shape()[1], cache_x_->shape()[2],
                           cache_x_->shape()[3], cache_y_->shape()[2], cache_y_->shape()[3]);
  return *grad_input_;
}

}  // namespace nn
