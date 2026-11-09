#include "nn/layers/transformer_block.hpp"

#include "ensure.hpp"
#include "nn/error.hpp"
#include "nn/layers/dropout.hpp"
#include "nn/layers/gelu.hpp"
#include "nn/layers/layer_norm.hpp"
#include "nn/layers/linear.hpp"
#include "nn/layers/rms_norm.hpp"
#include "nn/layers/swiglu.hpp"
#include "ocl.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace nn {
namespace {

void append_params(std::vector<NamedParam>& out, const Layer& layer, const std::string& prefix) {
  for (NamedParam param : layer.parameters()) {
    param.name = prefix + param.name;
    out.push_back(param);
  }
}

void require_block_spec(const TransformerBlockSpec& spec) {
  if (spec.d_model < 1 || spec.num_heads < 1 || spec.d_ff < 1 ||
      spec.d_model % spec.num_heads != 0) {
    throw Error("TransformerBlock size");
  }
  if (!(spec.dropout >= 0.f) || !(spec.dropout < 1.f) || !std::isfinite(spec.dropout)) {
    throw Error("TransformerBlock dropout");
  }
  if (spec.activation != TransformerActivation::Gelu &&
      spec.activation != TransformerActivation::SwiGLU) {
    throw Error("TransformerBlock activation");
  }
  if (spec.activation == TransformerActivation::SwiGLU &&
      spec.d_ff > std::numeric_limits<int>::max() / 2) {
    throw Error("TransformerBlock size");
  }
}

}  // namespace

TransformerBlock::TransformerBlock(Context& ctx, const TransformerBlockSpec& spec, std::mt19937& rng)
    : ctx_(&ctx), d_model_(spec.d_model) {
  require_block_spec(spec);
  const bool swiglu = spec.activation == TransformerActivation::SwiGLU;
  const int fc1_out = swiglu ? spec.d_ff * 2 : spec.d_ff;
  if (spec.use_rope) {
    ln1_ = std::make_unique<RMSNorm>(ctx, spec.d_model);
  } else {
    ln1_ = std::make_unique<LayerNorm>(ctx, spec.d_model);
  }
  mha_ = std::make_unique<MultiHeadAttention>(ctx, spec.d_model, spec.num_heads, true, spec.bias,
                                              spec.use_rope, rng);
  drop1_ = std::make_unique<Dropout>(ctx, spec.dropout);
  if (spec.use_rope) {
    ln2_ = std::make_unique<RMSNorm>(ctx, spec.d_model);
  } else {
    ln2_ = std::make_unique<LayerNorm>(ctx, spec.d_model);
  }
  fc1_ = std::make_unique<Linear>(ctx, spec.d_model, fc1_out, spec.bias, rng);
  if (swiglu) {
    act_ = std::make_unique<SwiGLU>(ctx);
  } else {
    act_ = std::make_unique<GELU>(ctx);
  }
  fc2_ = std::make_unique<Linear>(ctx, spec.d_ff, spec.d_model, spec.bias, rng);
  drop2_ = std::make_unique<Dropout>(ctx, spec.dropout);
}

TransformerBlock::~TransformerBlock() = default;

std::vector<NamedParam> TransformerBlock::parameters() const {
  std::vector<NamedParam> params;
  append_params(params, *ln1_, "ln1.");
  append_params(params, *mha_, "attn.");
  append_params(params, *drop1_, "drop1.");
  append_params(params, *ln2_, "ln2.");
  append_params(params, *fc1_, "fc1.");
  append_params(params, *fc2_, "fc2.");
  append_params(params, *drop2_, "drop2.");
  return params;
}

void TransformerBlock::zero_grad() {
  ln1_->zero_grad();
  mha_->zero_grad();
  drop1_->zero_grad();
  ln2_->zero_grad();
  fc1_->zero_grad();
  act_->zero_grad();
  fc2_->zero_grad();
  drop2_->zero_grad();
}

void TransformerBlock::set_training(bool training) {
  ln1_->set_training(training);
  mha_->set_training(training);
  drop1_->set_training(training);
  ln2_->set_training(training);
  fc1_->set_training(training);
  act_->set_training(training);
  fc2_->set_training(training);
  drop2_->set_training(training);
}

const Tensor& TransformerBlock::forward(const Tensor& input) {
  const std::vector<int>& shape = input.shape();
  if (shape.size() != 3 || shape[2] != d_model_) throw Error("TransformerBlock input shape");

  const Tensor& dropped_attn = drop1_->forward(mha_->forward(ln1_->forward(input)));
  ensure_tensor(hidden_, *ctx_, shape);
  launch_add2(*ctx_, hidden_->data(), input.data(), dropped_attn.data(), input.numel());

  const Tensor& dropped_ff =
      drop2_->forward(fc2_->forward(act_->forward(fc1_->forward(ln2_->forward(*hidden_)))));
  ensure_tensor(output_, *ctx_, shape);
  launch_add2(*ctx_, output_->data(), hidden_->data(), dropped_ff.data(), input.numel());
  return *output_;
}

const Tensor& TransformerBlock::forward_cached(const Tensor& input, MultiHeadAttention::Cache& cache) {
  const std::vector<int>& shape = input.shape();
  if (shape.size() != 3 || shape[2] != d_model_) throw Error("TransformerBlock input shape");

  const Tensor& dropped_attn = drop1_->forward(mha_->forward_cached(ln1_->forward(input), cache));
  ensure_tensor(hidden_, *ctx_, shape);
  launch_add2(*ctx_, hidden_->data(), input.data(), dropped_attn.data(), input.numel());

  const Tensor& dropped_ff =
      drop2_->forward(fc2_->forward(act_->forward(fc1_->forward(ln2_->forward(*hidden_)))));
  ensure_tensor(output_, *ctx_, shape);
  launch_add2(*ctx_, output_->data(), hidden_->data(), dropped_ff.data(), input.numel());
  return *output_;
}

const Tensor& TransformerBlock::backward(const Tensor& grad_output) {
  if (!hidden_ || !output_ || grad_output.shape() != output_->shape()) {
    throw Error("TransformerBlock backward shape");
  }
  const Tensor& g_ffn =
      ln2_->backward(fc1_->backward(act_->backward(fc2_->backward(drop2_->backward(grad_output)))));
  ensure_tensor(grad_hidden_, *ctx_, hidden_->shape());
  launch_add2(*ctx_, grad_hidden_->data(), grad_output.data(), g_ffn.data(), grad_output.numel());

  const Tensor& g_attn = ln1_->backward(mha_->backward(drop1_->backward(*grad_hidden_)));
  ensure_tensor(grad_input_, *ctx_, hidden_->shape());
  launch_add2(*ctx_, grad_input_->data(), grad_hidden_->data(), g_attn.data(), grad_output.numel());
  return *grad_input_;
}

}  // namespace nn
