#include "nn/layers/flatten.hpp"

#include "ensure.hpp"
#include "nn/error.hpp"

#include <cstdint>

namespace nn {

std::vector<NamedParam> Flatten::parameters() const { return {}; }

void Flatten::zero_grad() {}

const Tensor& Flatten::forward(const Tensor& input) {
  if (input.rank() != 4) throw Error("Flatten input rank");
  n_ = input.shape()[0];
  c_ = input.shape()[1];
  h_ = input.shape()[2];
  w_ = input.shape()[3];
  const int64_t features = static_cast<int64_t>(c_) * h_ * w_;
  if (features > INT32_MAX) throw Error("Flatten features out of range");
  ensure_tensor(output_, input.context(), {n_, static_cast<int>(features)});
  output_->copy_from(input);
  return *output_;
}

const Tensor& Flatten::backward(const Tensor& grad_output) {
  if (n_ == 0) throw Error("Flatten backward without forward");
  const int features = c_ * h_ * w_;
  if (grad_output.rank() != 2 || grad_output.shape()[0] != n_ || grad_output.shape()[1] != features) {
    throw Error("Flatten backward shape");
  }
  ensure_tensor(grad_input_, grad_output.context(), {n_, c_, h_, w_});
  grad_input_->copy_from(grad_output);
  return *grad_input_;
}

}  // namespace nn
