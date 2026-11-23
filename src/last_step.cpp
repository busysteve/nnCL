#include "nn/layers/last_step.hpp"

#include "activate.hpp"
#include "ensure.hpp"
#include "nn/error.hpp"
#include "ocl.hpp"

namespace nn {

std::vector<NamedParam> LastStep::parameters() const { return {}; }

void LastStep::zero_grad() {}

const Tensor& LastStep::forward(const Tensor& input) {
  if (input.rank() != 3) throw Error("LastStep input rank");
  n_ = input.shape()[0];
  t_ = input.shape()[1];
  f_ = input.shape()[2];
  ensure_activation(output_, input.context(), {n_, f_});
  if (output_->dtype() == DType::BFloat16) {
    Context& ctx = input.context();
    arena_reset(ctx);
    float* x = stage_f32(ctx, input);
    float* y = arena_f32(ctx, output_->numel());
    launch_gather_time(ctx, y, x, n_, t_, f_, t_ - 1);
    keep_f32(ctx, *output_, y);
    return *output_;
  }
  launch_gather_time(input.context(), output_->data(), input.data(), n_, t_, f_, t_ - 1);
  return *output_;
}

const Tensor& LastStep::backward(const Tensor& grad_output) {
  if (t_ == 0) throw Error("LastStep backward without forward");
  if (grad_output.rank() != 2 || grad_output.shape()[0] != n_ || grad_output.shape()[1] != f_) {
    throw Error("LastStep backward shape");
  }
  ensure_activation(grad_input_, grad_output.context(), {n_, t_, f_});
  if (grad_input_->dtype() == DType::BFloat16) {
    Context& ctx = grad_output.context();
    arena_reset(ctx);
    float* dy = stage_f32(ctx, grad_output);
    float* dx = arena_f32(ctx, grad_input_->numel());
    // The scatter kernel writes one time step and leaves the rest untouched.
    zero_f32(ctx, dx, grad_input_->numel());
    launch_scatter_time(ctx, dx, dy, n_, t_, f_, t_ - 1);
    keep_f32(ctx, *grad_input_, dx);
    return *grad_input_;
  }
  grad_input_->zero();
  launch_scatter_time(grad_output.context(), grad_input_->data(), grad_output.data(), n_, t_, f_,
                      t_ - 1);
  return *grad_input_;
}

}  // namespace nn
