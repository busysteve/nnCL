#include "nn/layers/conv2d.hpp"

#include "ensure.hpp"
#include "nn/config.hpp"
#include "nn/error.hpp"
#include "nn/init.hpp"
#include "ocl.hpp"

namespace nn {

struct Conv2d::Impl {};

Conv2d::Conv2d(Context& ctx, int in_channels, int out_channels, int kernel, int stride, int pad,
               bool use_bias, std::mt19937& rng)
    : ctx_(&ctx),
      in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_(kernel),
      stride_(stride),
      pad_(pad),
      use_bias_(use_bias),
      weight_(ctx, {out_channels, in_channels, kernel, kernel}),
      grad_weight_(ctx, {out_channels, in_channels, kernel, kernel}),
      impl_(std::make_unique<Impl>()) {
  if (in_channels < 1 || out_channels < 1) throw Error("Conv2d channels");
  conv_out_size(1 + kernel, kernel, stride, pad);
  kaiming_normal(weight_, in_channels * kernel * kernel, rng);
  grad_weight_.zero();
  if (use_bias_) {
    bias_.emplace(ctx, std::vector<int>{out_channels});
    grad_bias_.emplace(ctx, std::vector<int>{out_channels});
    bias_->zero();
    grad_bias_->zero();
  }
}

Conv2d::~Conv2d() = default;

std::vector<NamedParam> Conv2d::parameters() const {
  std::vector<NamedParam> params;
  params.push_back(NamedParam{"weight", const_cast<Tensor*>(&weight_), const_cast<Tensor*>(&grad_weight_)});
  if (bias_) {
    params.push_back(NamedParam{"bias", const_cast<Tensor*>(&*bias_), const_cast<Tensor*>(&*grad_bias_)});
  }
  return params;
}

void Conv2d::zero_grad() {
  grad_weight_.zero();
  if (grad_bias_) grad_bias_->zero();
}

void Conv2d::prepare(int n, int h, int w) {
  const int ho = conv_out_size(h, kernel_, stride_, pad_);
  const int wo = conv_out_size(w, kernel_, stride_, pad_);
  ensure_tensor(output_, *ctx_, {n, out_channels_, ho, wo});
  ensure_tensor(grad_input_, *ctx_, {n, in_channels_, h, w});
}

const Tensor& Conv2d::forward(const Tensor& input) {
  if (input.rank() != 4 || input.shape()[1] != in_channels_) throw Error("Conv2d input shape");
  const int n = input.shape()[0];
  const int h = input.shape()[2];
  const int w = input.shape()[3];
  prepare(n, h, w);
  ensure_tensor(cache_x_, *ctx_, input.shape());
  cache_x_->copy_from(input);
  const float* bias = use_bias_ ? bias_->data() : nullptr;
  const int has_bias = use_bias_ ? 1 : 0;
  launch_conv2d_forward(*ctx_, output_->data(), input.data(), weight_.data(), bias, has_bias, n,
                        in_channels_, out_channels_, h, w, kernel_, output_->shape()[2],
                        output_->shape()[3], stride_, pad_);
  return *output_;
}

const Tensor& Conv2d::backward(const Tensor& grad_output) {
  if (!cache_x_ || !output_ || grad_output.shape() != output_->shape()) {
    throw Error("Conv2d backward shape");
  }
  const int n = cache_x_->shape()[0];
  const int h = cache_x_->shape()[2];
  const int w = cache_x_->shape()[3];
  const int ho = output_->shape()[2];
  const int wo = output_->shape()[3];
  launch_conv2d_backward_data(*ctx_, grad_input_->data(), grad_output.data(), weight_.data(), n,
                              in_channels_, out_channels_, h, w, kernel_, ho, wo, stride_, pad_);
  launch_conv2d_backward_filter(*ctx_, grad_weight_.data(), cache_x_->data(), grad_output.data(), n,
                                in_channels_, out_channels_, h, w, kernel_, ho, wo, stride_, pad_);
  if (grad_bias_) {
    launch_conv_bias_grad(*ctx_, grad_output.data(), grad_bias_->data(), n, out_channels_, ho, wo);
  }
  return *grad_input_;
}

}  // namespace nn
