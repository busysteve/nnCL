#include "nn/layers/batch_norm2d.hpp"

#include "ensure.hpp"
#include "nn/error.hpp"
#include "ocl.hpp"

namespace nn {

struct BatchNorm2d::Impl {};

BatchNorm2d::BatchNorm2d(Context& ctx, int channels)
    : ctx_(&ctx),
      channels_(channels),
      weight_(ctx, {channels}),
      grad_weight_(ctx, {channels}),
      bias_(ctx, {channels}),
      grad_bias_(ctx, {channels}),
      running_mean_(ctx, {channels}),
      running_var_(ctx, {channels}),
      saved_mean_(ctx, {channels}),
      saved_inv_variance_(ctx, {channels}),
      impl_(std::make_unique<Impl>()) {
  if (channels < 1) throw Error("BatchNorm2d channels");
  weight_.fill(1.f);
  bias_.zero();
  grad_weight_.zero();
  grad_bias_.zero();
  running_mean_.zero();
  running_var_.fill(1.f);
  saved_mean_.zero();
  saved_inv_variance_.zero();
}

BatchNorm2d::~BatchNorm2d() = default;

std::vector<NamedParam> BatchNorm2d::parameters() const {
  return {NamedParam{"weight", const_cast<Tensor*>(&weight_), const_cast<Tensor*>(&grad_weight_)},
          NamedParam{"bias", const_cast<Tensor*>(&bias_), const_cast<Tensor*>(&grad_bias_)}};
}

std::vector<NamedBuffer> BatchNorm2d::buffers() const {
  auto* self = const_cast<BatchNorm2d*>(this);
  return {NamedBuffer{"running_mean", &self->running_mean_, &self->batches_},
          NamedBuffer{"running_var", &self->running_var_, &self->batches_}};
}

void BatchNorm2d::zero_grad() {
  grad_weight_.zero();
  grad_bias_.zero();
}

void BatchNorm2d::set_training(bool training) { training_ = training; }

void BatchNorm2d::bind(const Tensor& input) {
  if (input.rank() != 4 || input.shape()[1] != channels_) throw Error("BatchNorm2d input shape");
  ensure_tensor(output_, *ctx_, input.shape());
  ensure_tensor(grad_input_, *ctx_, input.shape());
}

const Tensor& BatchNorm2d::forward(const Tensor& input) {
  bind(input);
  ensure_tensor(cache_x_, *ctx_, input.shape());
  cache_x_->copy_from(input);
  const float eps = 1e-5f;
  const int n = input.shape()[0];
  const int h = input.shape()[2];
  const int w = input.shape()[3];
  if (training_) {
    const int64_t count =
        static_cast<int64_t>(input.shape()[0]) * input.shape()[2] * input.shape()[3];
    if (count <= 1) throw Error("BatchNorm train mode needs more than one value per channel");
    ++batches_;
    const float factor = static_cast<float>(1.0 / static_cast<double>(batches_));
    launch_batch_norm_train(*ctx_, output_->data(), input.data(), weight_.data(), bias_.data(),
                            running_mean_.data(), running_var_.data(), saved_mean_.data(),
                            saved_inv_variance_.data(), n, channels_, h, w, eps, factor);
  } else {
    launch_batch_norm_eval(*ctx_, output_->data(), input.data(), weight_.data(), bias_.data(),
                           running_mean_.data(), running_var_.data(), n, channels_, h, w, eps);
  }
  return *output_;
}

const Tensor& BatchNorm2d::backward(const Tensor& grad_output) {
  if (!training_) throw Error("BatchNorm backward requires train mode");
  if (!cache_x_ || grad_output.shape() != cache_x_->shape()) throw Error("BatchNorm2d backward shape");
  const int n = cache_x_->shape()[0];
  const int h = cache_x_->shape()[2];
  const int w = cache_x_->shape()[3];
  launch_batch_norm_backward(*ctx_, grad_input_->data(), grad_output.data(), cache_x_->data(),
                             weight_.data(), saved_mean_.data(), saved_inv_variance_.data(),
                             grad_weight_.data(), grad_bias_.data(), n, channels_, h, w);
  return *grad_input_;
}

}  // namespace nn
