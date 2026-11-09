#include "nn/layers/residual_block.hpp"

#include "ensure.hpp"
#include "nn/error.hpp"
#include "nn/layers/batch_norm2d.hpp"
#include "nn/layers/conv2d.hpp"
#include "nn/layers/relu.hpp"
#include "ocl.hpp"

namespace nn {

struct ResidualBlock::Impl {};

namespace {

void append_params(std::vector<NamedParam>& out, const Layer& layer, const std::string& prefix) {
  for (NamedParam param : layer.parameters()) {
    param.name = prefix + param.name;
    out.push_back(param);
  }
}

void append_buffers(std::vector<NamedBuffer>& out, const Layer& layer, const std::string& prefix) {
  for (NamedBuffer buffer : layer.buffers()) {
    buffer.name = prefix + buffer.name;
    out.push_back(buffer);
  }
}

}  // namespace

ResidualBlock::ResidualBlock(Context& ctx, int in_channels, int out_channels, int stride,
                             std::mt19937& rng)
    : ctx_(&ctx), project_(stride != 1 || in_channels != out_channels), impl_(std::make_unique<Impl>()) {
  if (in_channels < 1 || out_channels < 1 || stride < 1) throw Error("ResidualBlock size");
  conv1_ = std::make_unique<Conv2d>(ctx, in_channels, out_channels, 3, stride, 1, false, rng);
  bn1_ = std::make_unique<BatchNorm2d>(ctx, out_channels);
  relu1_ = std::make_unique<ReLU>(ctx);
  conv2_ = std::make_unique<Conv2d>(ctx, out_channels, out_channels, 3, 1, 1, false, rng);
  bn2_ = std::make_unique<BatchNorm2d>(ctx, out_channels);
  if (project_) {
    proj_ = std::make_unique<Conv2d>(ctx, in_channels, out_channels, 1, stride, 0, false, rng);
    proj_bn_ = std::make_unique<BatchNorm2d>(ctx, out_channels);
  }
  relu_out_ = std::make_unique<ReLU>(ctx);
}

ResidualBlock::~ResidualBlock() = default;

std::vector<NamedParam> ResidualBlock::parameters() const {
  std::vector<NamedParam> params;
  append_params(params, *conv1_, "conv1.");
  append_params(params, *bn1_, "bn1.");
  append_params(params, *conv2_, "conv2.");
  append_params(params, *bn2_, "bn2.");
  if (project_) {
    append_params(params, *proj_, "proj.");
    append_params(params, *proj_bn_, "proj_bn.");
  }
  return params;
}

std::vector<NamedBuffer> ResidualBlock::buffers() const {
  std::vector<NamedBuffer> buffers;
  append_buffers(buffers, *bn1_, "bn1.");
  append_buffers(buffers, *bn2_, "bn2.");
  if (project_) append_buffers(buffers, *proj_bn_, "proj_bn.");
  return buffers;
}

void ResidualBlock::zero_grad() {
  conv1_->zero_grad();
  bn1_->zero_grad();
  conv2_->zero_grad();
  bn2_->zero_grad();
  if (project_) {
    proj_->zero_grad();
    proj_bn_->zero_grad();
  }
}

void ResidualBlock::set_training(bool training) {
  conv1_->set_training(training);
  bn1_->set_training(training);
  relu1_->set_training(training);
  conv2_->set_training(training);
  bn2_->set_training(training);
  if (project_) {
    proj_->set_training(training);
    proj_bn_->set_training(training);
  }
  relu_out_->set_training(training);
}

const Tensor& ResidualBlock::forward(const Tensor& input) {
  const Tensor& hidden = relu1_->forward(bn1_->forward(conv1_->forward(input)));
  const Tensor& residual = bn2_->forward(conv2_->forward(hidden));
  const Tensor* skip = &input;
  if (project_) skip = &proj_bn_->forward(proj_->forward(input));
  if (skip->shape() != residual.shape()) throw Error("ResidualBlock skip shape");
  ensure_tensor(sum_, *ctx_, residual.shape());
  launch_add2(*ctx_, sum_->data(), residual.data(), skip->data(), residual.numel());
  return relu_out_->forward(*sum_);
}

const Tensor& ResidualBlock::backward(const Tensor& grad_output) {
  // Main branch first. Both branches then receive the full pre-activation gradient.
  const Tensor& g_sum = relu_out_->backward(grad_output);
  const Tensor& g_conv2 = bn2_->backward(g_sum);
  const Tensor& g_hidden = conv2_->backward(g_conv2);
  const Tensor& g_bn1 = relu1_->backward(g_hidden);
  const Tensor& g_conv1 = bn1_->backward(g_bn1);
  const Tensor& g_main = conv1_->backward(g_conv1);
  const Tensor* g_skip = &g_sum;
  if (project_) {
    const Tensor& g_proj_bn = proj_bn_->backward(g_sum);
    g_skip = &proj_->backward(g_proj_bn);
  }
  if (g_main.shape() != g_skip->shape()) throw Error("ResidualBlock grad shape");
  ensure_tensor(grad_input_, *ctx_, g_main.shape());
  launch_add2(*ctx_, grad_input_->data(), g_main.data(), g_skip->data(), g_main.numel());
  return *grad_input_;
}

}  // namespace nn
