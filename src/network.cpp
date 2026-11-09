#include "nn/network.hpp"

#include "nn/error.hpp"

namespace nn {
namespace {

std::string join_name(const std::string& prefix, const std::string& name) {
  if (prefix.empty()) return name;
  return prefix + "." + name;
}

}  // namespace

Network::Network(Context& ctx, Family family, std::vector<std::unique_ptr<Layer>> layers,
                 std::variant<MlpConfig, ConvConfig, ResNetConfig, ConvResNetConfig, RnnConfig,
                              TransformerConfig> config)
    : ctx_(&ctx), family_(family), layers_(std::move(layers)), config_(std::move(config)) {}

const Tensor& Network::forward(const Tensor& input) {
  if (layers_.empty()) throw Error("empty network");
  const Tensor* current = &input;
  for (const std::unique_ptr<Layer>& layer : layers_) current = &layer->forward(*current);
  return *current;
}

const Tensor& Network::backward(const Tensor& grad_output) {
  if (layers_.empty()) throw Error("empty network");
  const Tensor* grad = &grad_output;
  for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) grad = &(*it)->backward(*grad);
  return *grad;
}

void Network::zero_grad() {
  for (const std::unique_ptr<Layer>& layer : layers_) layer->zero_grad();
}

void Network::train() {
  training_ = true;
  for (const std::unique_ptr<Layer>& layer : layers_) layer->set_training(true);
}

void Network::eval() {
  training_ = false;
  for (const std::unique_ptr<Layer>& layer : layers_) layer->set_training(false);
}

std::vector<NamedParam> Network::parameters() const {
  std::vector<NamedParam> params;
  for (const std::unique_ptr<Layer>& layer : layers_) {
    for (NamedParam param : layer->parameters()) {
      param.name = join_name(layer->name(), param.name);
      params.push_back(param);
    }
  }
  return params;
}

std::vector<NamedBuffer> Network::buffers() const {
  std::vector<NamedBuffer> buffers;
  for (const std::unique_ptr<Layer>& layer : layers_) {
    for (NamedBuffer buffer : layer->buffers()) {
      buffer.name = join_name(layer->name(), buffer.name);
      buffers.push_back(buffer);
    }
  }
  return buffers;
}

}  // namespace nn
