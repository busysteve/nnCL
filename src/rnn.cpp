#include "nn/layers/rnn.hpp"

#include "ensure.hpp"
#include "nn/error.hpp"
#include "nn/init.hpp"
#include "ocl.hpp"

namespace nn {
namespace {

Context& require_rnn_size(Context& ctx, int input_size, int hidden_size) {
  if (input_size < 1 || hidden_size < 1) throw Error("RNN size");
  return ctx;
}

void gemm_xh(Context& ctx, const Tensor& weight, const Tensor& x, Tensor& y, float beta) {
  const int out = weight.shape()[0];
  const int in = weight.shape()[1];
  const int n = x.shape()[0];
  gemm_nt(ctx, y.data(), x.data(), weight.data(), n, out, in, 1, out, in, in,
          static_cast<int64_t>(n) * out, static_cast<int64_t>(n) * in,
          static_cast<int64_t>(out) * in, 1.f, beta);
}

void gemm_grad_weight(Context& ctx, const Tensor& x, const Tensor& grad_y, Tensor& grad_w) {
  const int n = x.shape()[0];
  const int in = x.shape()[1];
  const int out = grad_y.shape()[1];
  gemm_tn(ctx, grad_w.data(), grad_y.data(), x.data(), out, in, n, 1, in, out, in,
          static_cast<int64_t>(out) * in, static_cast<int64_t>(n) * out,
          static_cast<int64_t>(n) * in, 1.f, 1.f);
}

void gemm_grad_input(Context& ctx, const Tensor& weight, const Tensor& grad_y, Tensor& grad_x) {
  const int out = weight.shape()[0];
  const int in = weight.shape()[1];
  const int n = grad_y.shape()[0];
  gemm_nn(ctx, grad_x.data(), grad_y.data(), weight.data(), n, in, out, 1, in, out, in,
          static_cast<int64_t>(n) * in, static_cast<int64_t>(n) * out,
          static_cast<int64_t>(out) * in, 1.f, 0.f);
}

}  // namespace

RNN::RNN(Context& ctx, int input_size, int hidden_size, std::mt19937& rng)
    : ctx_(&ctx),
      input_size_(input_size),
      hidden_size_(hidden_size),
      weight_ih_(require_rnn_size(ctx, input_size, hidden_size), {hidden_size, input_size}),
      grad_weight_ih_(ctx, {hidden_size, input_size}),
      weight_hh_(ctx, {hidden_size, hidden_size}),
      grad_weight_hh_(ctx, {hidden_size, hidden_size}),
      bias_ih_(ctx, {hidden_size}),
      grad_bias_ih_(ctx, {hidden_size}),
      bias_hh_(ctx, {hidden_size}),
      grad_bias_hh_(ctx, {hidden_size}) {
  kaiming_normal(weight_ih_, input_size_, rng);
  kaiming_normal(weight_hh_, hidden_size_, rng);
  bias_ih_.zero();
  bias_hh_.zero();
  grad_weight_ih_.zero();
  grad_weight_hh_.zero();
  grad_bias_ih_.zero();
  grad_bias_hh_.zero();
}

std::vector<NamedParam> RNN::parameters() const {
  return {NamedParam{"weight_ih", const_cast<Tensor*>(&weight_ih_), const_cast<Tensor*>(&grad_weight_ih_)},
          NamedParam{"weight_hh", const_cast<Tensor*>(&weight_hh_), const_cast<Tensor*>(&grad_weight_hh_)},
          NamedParam{"bias_ih", const_cast<Tensor*>(&bias_ih_), const_cast<Tensor*>(&grad_bias_ih_)},
          NamedParam{"bias_hh", const_cast<Tensor*>(&bias_hh_), const_cast<Tensor*>(&grad_bias_hh_)}};
}

void RNN::zero_grad() {
  grad_weight_ih_.zero();
  grad_weight_hh_.zero();
  grad_bias_ih_.zero();
  grad_bias_hh_.zero();
}

const Tensor& RNN::forward(const Tensor& input) {
  if (input.rank() != 3 || input.shape()[2] != input_size_) throw Error("RNN input shape");
  const int n = input.shape()[0];
  const int time = input.shape()[1];
  ensure_tensor(cache_x_, *ctx_, input.shape());
  ensure_tensor(output_, *ctx_, {n, time, hidden_size_});
  ensure_tensor(x_t_, *ctx_, {n, input_size_});
  ensure_tensor(h_t_, *ctx_, {n, hidden_size_});
  ensure_tensor(h_prev_, *ctx_, {n, hidden_size_});
  ensure_tensor(pre_, *ctx_, {n, hidden_size_});
  cache_x_->copy_from(input);
  h_prev_->zero();

  const int64_t hidden_count = static_cast<int64_t>(n) * hidden_size_;
  for (int t = 0; t < time; ++t) {
    launch_gather_time(*ctx_, x_t_->data(), cache_x_->data(), n, time, input_size_, t);
    gemm_xh(*ctx_, weight_ih_, *x_t_, *pre_, 0.f);
    launch_add_bias(*ctx_, pre_->data(), bias_ih_.data(), n, hidden_size_);
    gemm_xh(*ctx_, weight_hh_, *h_prev_, *pre_, 1.f);
    launch_add_bias(*ctx_, pre_->data(), bias_hh_.data(), n, hidden_size_);
    launch_tanh(*ctx_, h_t_->data(), pre_->data(), hidden_count);
    launch_scatter_time(*ctx_, output_->data(), h_t_->data(), n, time, hidden_size_, t);
    h_prev_->copy_from(*h_t_);
  }
  return *output_;
}

const Tensor& RNN::backward(const Tensor& grad_output) {
  if (!cache_x_ || !output_ || grad_output.shape() != output_->shape()) {
    throw Error("RNN backward shape");
  }
  const int n = cache_x_->shape()[0];
  const int time = cache_x_->shape()[1];
  ensure_tensor(grad_input_, *ctx_, cache_x_->shape());
  ensure_tensor(x_t_, *ctx_, {n, input_size_});
  ensure_tensor(h_t_, *ctx_, {n, hidden_size_});
  ensure_tensor(h_prev_, *ctx_, {n, hidden_size_});
  ensure_tensor(grad_a_, *ctx_, {n, hidden_size_});
  ensure_tensor(grad_h_, *ctx_, {n, hidden_size_});
  ensure_tensor(grad_y_t_, *ctx_, {n, hidden_size_});
  ensure_tensor(grad_h_next_, *ctx_, {n, hidden_size_});
  ensure_tensor(grad_x_t_, *ctx_, {n, input_size_});
  grad_h_next_->zero();

  const int64_t hidden_count = static_cast<int64_t>(n) * hidden_size_;
  for (int t = time - 1; t >= 0; --t) {
    launch_gather_time(*ctx_, grad_y_t_->data(), grad_output.data(), n, time, hidden_size_, t);
    launch_add2(*ctx_, grad_h_->data(), grad_y_t_->data(), grad_h_next_->data(), hidden_count);
    launch_gather_time(*ctx_, h_t_->data(), output_->data(), n, time, hidden_size_, t);
    launch_tanh_backward(*ctx_, grad_a_->data(), h_t_->data(), grad_h_->data(), hidden_count);
    launch_gather_time(*ctx_, x_t_->data(), cache_x_->data(), n, time, input_size_, t);
    if (t == 0) {
      h_prev_->zero();
    } else {
      launch_gather_time(*ctx_, h_prev_->data(), output_->data(), n, time, hidden_size_, t - 1);
    }
    gemm_grad_weight(*ctx_, *x_t_, *grad_a_, grad_weight_ih_);
    gemm_grad_weight(*ctx_, *h_prev_, *grad_a_, grad_weight_hh_);
    launch_bias_grad(*ctx_, grad_a_->data(), grad_bias_ih_.data(), n, hidden_size_);
    launch_bias_grad(*ctx_, grad_a_->data(), grad_bias_hh_.data(), n, hidden_size_);
    gemm_grad_input(*ctx_, weight_ih_, *grad_a_, *grad_x_t_);
    launch_scatter_time(*ctx_, grad_input_->data(), grad_x_t_->data(), n, time, input_size_, t);
    gemm_grad_input(*ctx_, weight_hh_, *grad_a_, *grad_h_next_);
  }
  return *grad_input_;
}

}  // namespace nn
