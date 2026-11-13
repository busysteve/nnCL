#include "check.hpp"

#include "nn/index_tensor.hpp"
#include "nn/layers/batch_norm2d.hpp"
#include "nn/layers/conv2d.hpp"
#include "nn/layers/dropout.hpp"
#include "nn/layers/embedding.hpp"
#include "nn/layers/gelu.hpp"
#include "nn/layers/last_step.hpp"
#include "nn/layers/layer_norm.hpp"
#include "nn/layers/linear.hpp"
#include "nn/layers/max_pool2d.hpp"
#include "nn/layers/multi_head_attention.hpp"
#include "nn/layers/relu.hpp"
#include "nn/layers/residual_block.hpp"
#include "nn/layers/rms_norm.hpp"
#include "nn/layers/rnn.hpp"
#include "nn/layers/swiglu.hpp"
#include "nn/layers/transformer_block.hpp"
#include "nn/loss.hpp"
#include "nn/optim.hpp"
#include "nn/transformer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kEps = 1e-3;

void check_close(double analytic, double numeric, double rel, double abs_floor, const char* name,
                 size_t index) {
  const double diff = std::abs(analytic - numeric);
  const double scale = std::max(1.0, std::max(std::abs(analytic), std::abs(numeric)));
  if (diff / scale <= rel || diff <= abs_floor) return;
  std::fprintf(stderr, "grad mismatch %s[%zu] analytic=%.8g numeric=%.8g diff=%.8g rel=%.4g\n", name,
               index, analytic, numeric, diff, diff / scale);
  std::exit(1);
}

double squared_loss(const nn::Tensor& output) {
  const std::vector<float> values = output.to_host();
  double sum = 0.0;
  for (float value : values) sum += 0.5 * static_cast<double>(value) * static_cast<double>(value);
  return sum;
}

void finite_difference(nn::Tensor& value, const std::vector<float>& analytic,
                       const std::function<double()>& loss, const char* name, double rel = 1e-2) {
  const std::vector<float> original = value.to_host();
  CHECK(original.size() == analytic.size());
  std::vector<float> current = original;
  for (size_t i = 0; i < current.size(); ++i) {
    current[i] = original[i] + static_cast<float>(kEps);
    value.copy_from_host(current);
    const double loss_plus = loss();
    current[i] = original[i] - static_cast<float>(kEps);
    value.copy_from_host(current);
    const double loss_minus = loss();
    current[i] = original[i];
    value.copy_from_host(current);
    const double numeric = (loss_plus - loss_minus) / (2.0 * kEps);
    check_close(static_cast<double>(analytic[i]), numeric, rel, 1e-4, name, i);
  }
}

void check_layer(nn::Layer& layer, nn::Tensor& input, double rel = 1e-2) {
  layer.zero_grad();
  const nn::Tensor& output = layer.forward(input);
  const std::vector<float> output_host = output.to_host();
  nn::Tensor grad_output(input.context(), output.shape());
  grad_output.copy_from_host(output_host);
  const std::vector<float> grad_output_before = grad_output.to_host();
  const nn::Tensor& grad_input = layer.backward(grad_output);
  const std::vector<float> analytic_input = grad_input.to_host();
  std::vector<std::pair<nn::Tensor*, std::vector<float>>> analytic_params;
  std::vector<std::string> names;
  for (const nn::NamedParam& param : layer.parameters()) {
    names.push_back(param.name);
    analytic_params.push_back({param.value, param.grad->to_host()});
  }
  const std::vector<float> grad_output_after = grad_output.to_host();
  CHECK(std::memcmp(grad_output_before.data(), grad_output_after.data(),
                    grad_output_before.size() * sizeof(float)) == 0);

  const auto loss = [&]() { return squared_loss(layer.forward(input)); };
  finite_difference(input, analytic_input, loss, "input", rel);
  for (size_t i = 0; i < analytic_params.size(); ++i) {
    finite_difference(*analytic_params[i].first, analytic_params[i].second, loss, names[i].c_str(), rel);
  }
}

std::vector<float> uniform(int64_t n, std::mt19937& rng, float low, float high) {
  std::uniform_real_distribution<float> dist(low, high);
  std::vector<float> values(static_cast<size_t>(n));
  for (float& value : values) value = dist(rng);
  return values;
}

size_t argmax_abs(const std::vector<float>& values, size_t begin, size_t end) {
  CHECK(begin < end);
  CHECK(end <= values.size());
  size_t best = begin;
  float best_abs = std::fabs(values[begin]);
  for (size_t i = begin + 1; i < end; ++i) {
    const float magnitude = std::fabs(values[i]);
    if (magnitude > best_abs) {
      best_abs = magnitude;
      best = i;
    }
  }
  CHECK(best_abs > 0.f);
  return best;
}

void finite_difference_at(nn::Tensor& value, float analytic, size_t index,
                          const std::function<double()>& loss, const char* name) {
  const std::vector<float> original = value.to_host();
  CHECK(index < original.size());
  std::vector<float> current = original;
  current[index] = original[index] + static_cast<float>(kEps);
  value.copy_from_host(current);
  const double loss_plus = loss();
  current[index] = original[index] - static_cast<float>(kEps);
  value.copy_from_host(current);
  const double loss_minus = loss();
  current[index] = original[index];
  value.copy_from_host(current);
  const double numeric = (loss_plus - loss_minus) / (2.0 * kEps);
  check_close(static_cast<double>(analytic), numeric, 1e-2, 1e-4, name, index);
}

void check_embedding(nn::Embedding& layer, nn::IndexTensor& indices) {
  layer.zero_grad();
  const nn::Tensor& output = layer.forward_indices(indices);
  const std::vector<float> output_host = output.to_host();
  nn::Tensor grad_output(output.context(), output.shape());
  grad_output.copy_from_host(output_host);
  layer.backward(grad_output);
  const std::vector<nn::NamedParam> params = layer.parameters();
  CHECK(params.size() == 1);
  CHECK(params[0].name == "weight");
  const std::vector<float> analytic = params[0].grad->to_host();
  const auto loss = [&]() { return squared_loss(layer.forward_indices(indices)); };
  finite_difference(*params[0].value, analytic, loss, "weight");
}

void condition_bn(nn::Layer& layer) {
  // Keep every ReLU input away from zero so a 1e-3 probe stays on one side of the kink,
  // and keep the outputs moderate so float32 central differences stay inside 1e-2.
  for (nn::NamedParam param : layer.parameters()) {
    const std::string& name = param.name;
    if (name.find("bn") == std::string::npos) continue;
    if (name.size() >= 7 && name.compare(name.size() - 7, 7, ".weight") == 0) param.value->fill(0.05f);
    if (name.size() >= 5 && name.compare(name.size() - 5, 5, ".bias") == 0) param.value->fill(1.f);
  }
}

}  // namespace

int test_gradients() {
  nn::Context ctx;
  std::mt19937 rng(1);

  std::printf("grad linear\n");
  {
    nn::Linear layer(ctx, 5, 3, rng);
    nn::Tensor input(ctx, {4, 5});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad linear no bias\n");
  {
    nn::Linear layer(ctx, 5, 3, false, rng);
    nn::Tensor input(ctx, {4, 5});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad linear rank3\n");
  {
    nn::Linear layer(ctx, 4, 3, true, rng);
    nn::Tensor input(ctx, {2, 3, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad linear rank3 no bias\n");
  {
    nn::Linear layer(ctx, 4, 3, false, rng);
    nn::Tensor input(ctx, {2, 3, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad conv\n");
  {
    nn::Conv2d layer(ctx, 2, 2, 3, 1, 1, true, rng);
    nn::Tensor input(ctx, {2, 2, 5, 5});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad batchnorm\n");
  {
    nn::BatchNorm2d layer(ctx, 2);
    nn::Tensor input(ctx, {4, 2, 3, 3});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad relu\n");
  {
    nn::ReLU layer(ctx);
    nn::Tensor input(ctx, {2, 2, 2, 2});
    std::vector<float> values(static_cast<size_t>(input.numel()));
    std::uniform_real_distribution<float> dist(0.2f, 1.f);
    for (size_t i = 0; i < values.size(); ++i) {
      const float magnitude = dist(rng);
      values[i] = (i % 2 == 0) ? magnitude : -magnitude;
    }
    input.copy_from_host(values);
    check_layer(layer, input);
  }

  std::printf("grad relu rank3\n");
  {
    nn::ReLU layer(ctx);
    nn::Tensor input(ctx, {2, 3, 4});
    std::vector<float> values(static_cast<size_t>(input.numel()));
    std::uniform_real_distribution<float> dist(0.2f, 1.f);
    for (size_t i = 0; i < values.size(); ++i) {
      const float magnitude = dist(rng);
      values[i] = (i % 2 == 0) ? magnitude : -magnitude;
    }
    input.copy_from_host(values);
    check_layer(layer, input);
  }

  std::printf("grad gelu\n");
  {
    nn::GELU layer(ctx);
    nn::Tensor input(ctx, {2, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.8f, 0.8f));
    check_layer(layer, input);
  }

  std::printf("grad gelu rank3\n");
  {
    nn::GELU layer(ctx);
    nn::Tensor input(ctx, {2, 3, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.8f, 0.8f));
    check_layer(layer, input);
  }

  std::printf("grad layernorm\n");
  {
    nn::LayerNorm layer(ctx, 5);
    nn::Tensor input(ctx, {4, 5});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad layernorm rank3\n");
  {
    nn::LayerNorm layer(ctx, 4);
    std::vector<nn::NamedParam> params = layer.parameters();
    params[0].value->copy_from_host(std::vector<float>{0.5f, -1.f, 1.5f, 0.25f});
    params[1].value->copy_from_host(std::vector<float>{0.1f, -0.2f, 0.3f, -0.4f});
    nn::Tensor input(ctx, {2, 3, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad rmsnorm\n");
  {
    std::mt19937 local(11);
    nn::RMSNorm layer(ctx, 5);
    nn::Tensor input(ctx, {4, 5});
    input.copy_from_host(uniform(input.numel(), local, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad rmsnorm rank3\n");
  {
    std::mt19937 local(12);
    nn::RMSNorm layer(ctx, 4);
    std::vector<nn::NamedParam> params = layer.parameters();
    params[0].value->copy_from_host(std::vector<float>{0.5f, -1.f, 1.5f, 0.25f});
    nn::Tensor input(ctx, {2, 3, 4});
    input.copy_from_host(uniform(input.numel(), local, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad dropout p0\n");
  {
    nn::Dropout layer(ctx, 0.f);
    nn::Tensor input(ctx, {4, 5});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad dropout p0 rank3\n");
  {
    nn::Dropout layer(ctx, 0.f);
    nn::Tensor input(ctx, {2, 3, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad dropout eval\n");
  {
    nn::Dropout layer(ctx, 0.5f);
    layer.set_training(false);
    nn::Tensor input(ctx, {4, 5});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad dropout eval rank3\n");
  {
    nn::Dropout layer(ctx, 0.4f);
    layer.set_training(false);
    nn::Tensor input(ctx, {2, 3, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad swiglu\n");
  {
    nn::SwiGLU layer(ctx);
    nn::Tensor input(ctx, {2, 6});
    input.copy_from_host(uniform(input.numel(), rng, -0.8f, 0.8f));
    check_layer(layer, input);
  }

  std::printf("grad swiglu rank3\n");
  {
    nn::SwiGLU layer(ctx);
    nn::Tensor input(ctx, {2, 3, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.8f, 0.8f));
    check_layer(layer, input);
  }

  std::printf("grad swiglu nchw\n");
  {
    nn::SwiGLU layer(ctx);
    nn::Tensor input(ctx, {2, 2, 2, 2});
    input.copy_from_host(uniform(input.numel(), rng, -0.8f, 0.8f));
    check_layer(layer, input);
  }

  std::printf("grad last step\n");
  {
    nn::LastStep layer;
    nn::Tensor input(ctx, {2, 3, 2});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad rnn\n");
  {
    nn::RNN layer(ctx, 3, 2, rng);
    for (nn::NamedParam param : layer.parameters()) {
      if (param.name.find("weight") != std::string::npos) param.value->fill(0.2f);
    }
    nn::Tensor input(ctx, {2, 3, 3});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad maxpool\n");
  {
    nn::MaxPool2d layer(ctx, 2, 2);
    nn::Tensor input(ctx, {2, 1, 4, 4});
    std::vector<float> values(static_cast<size_t>(input.numel()), 0.f);
    const float window[4] = {0.05f, 0.15f, 0.25f, 0.80f};
    for (int n = 0; n < 2; ++n) {
      for (int by = 0; by < 2; ++by) {
        for (int bx = 0; bx < 2; ++bx) {
          for (int ky = 0; ky < 2; ++ky) {
            for (int kx = 0; kx < 2; ++kx) {
              const int y = by * 2 + ky;
              const int x = bx * 2 + kx;
              const size_t index = static_cast<size_t>(((n * 4) + y) * 4 + x);
              values[index] = window[ky * 2 + kx] + 0.01f * static_cast<float>(by * 2 + bx);
            }
          }
        }
      }
    }
    input.copy_from_host(values);
    check_layer(layer, input);
  }

  std::printf("grad residual identity\n");
  {
    nn::ResidualBlock layer(ctx, 2, 2, 1, rng);
    condition_bn(layer);
    nn::Tensor input(ctx, {2, 2, 6, 6});
    input.copy_from_host(uniform(input.numel(), rng, 0.2f, 0.8f));
    check_layer(layer, input);
  }

  std::printf("grad residual projection\n");
  {
    nn::ResidualBlock layer(ctx, 2, 4, 2, rng);
    condition_bn(layer);
    nn::Tensor input(ctx, {2, 2, 6, 6});
    input.copy_from_host(uniform(input.numel(), rng, 0.2f, 0.8f));
    check_layer(layer, input);
  }

  std::printf("grad mha causal\n");
  {
    nn::MultiHeadAttention layer(ctx, 4, 2, true, true, false, rng);
    nn::Tensor input(ctx, {1, 3, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad mha causal no bias\n");
  {
    nn::MultiHeadAttention layer(ctx, 4, 2, true, false, false, rng);
    nn::Tensor input(ctx, {1, 3, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad mha no causal\n");
  {
    nn::MultiHeadAttention layer(ctx, 4, 2, false, false, false, rng);
    nn::Tensor input(ctx, {1, 3, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad mha rope\n");
  {
    std::mt19937 local(13);
    nn::MultiHeadAttention layer(ctx, 4, 2, true, false, true, local);
    nn::Tensor input(ctx, {1, 3, 4});
    input.copy_from_host(uniform(input.numel(), local, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad transformer swiglu\n");
  {
    nn::TransformerBlockSpec spec;
    spec.d_model = 8;
    spec.num_heads = 2;
    spec.d_ff = 16;
    spec.dropout = 0.f;
    spec.bias = false;
    spec.activation = nn::TransformerActivation::SwiGLU;
    nn::TransformerBlock layer(ctx, spec, rng);
    nn::Tensor input(ctx, {1, 3, 8});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad transformer rope\n");
  {
    std::mt19937 local(14);
    nn::TransformerBlockSpec spec;
    spec.d_model = 8;
    spec.num_heads = 2;
    spec.d_ff = 16;
    spec.dropout = 0.f;
    spec.bias = false;
    spec.use_rope = true;
    spec.activation = nn::TransformerActivation::SwiGLU;
    nn::TransformerBlock layer(ctx, spec, local);
    nn::Tensor input(ctx, {1, 3, 8});
    input.copy_from_host(uniform(input.numel(), local, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad mha qwen\n");
  {
    std::mt19937 local(15);
    nn::MultiHeadAttention layer(ctx, 8, 4, true, false, true, local, 4, 2, true, 1000000.f, 1e-6f);
    nn::Tensor input(ctx, {1, 3, 8});
    input.copy_from_host(uniform(input.numel(), local, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad transformer qwen\n");
  {
    std::mt19937 local(16);
    nn::TransformerBlockSpec spec;
    spec.d_model = 8;
    spec.num_heads = 4;
    spec.d_ff = 16;
    spec.dropout = 0.f;
    spec.bias = false;
    spec.use_rope = true;
    spec.activation = nn::TransformerActivation::SwiGLU;
    spec.head_dim = 4;
    spec.num_kv_heads = 2;
    spec.qk_norm = true;
    spec.rope_theta = 1000000.f;
    spec.rms_eps = 1e-6f;
    nn::TransformerBlock layer(ctx, spec, local);
    nn::Tensor input(ctx, {1, 3, 8});
    input.copy_from_host(uniform(input.numel(), local, -0.5f, 0.5f));
    // QK-norm plus the block RMSNorm leaves one K weight just outside 1e-2.
    check_layer(layer, input, 2e-2);
  }

  std::printf("grad embedding\n");
  {
    nn::Embedding layer(ctx, 5, 3, rng);
    nn::IndexTensor indices(ctx, {6});
    // Index 1 is repeated so scatter-add has to accumulate.
    const std::vector<int64_t> host = {1, 0, 1, 4, 0, 1};
    indices.copy_from_host(host.data(), static_cast<int64_t>(host.size()));
    check_embedding(layer, indices);
  }

  std::printf("grad embedding rank2\n");
  {
    nn::Embedding layer(ctx, 4, 2, rng);
    nn::IndexTensor indices(ctx, {2, 3});
    const std::vector<int64_t> host = {2, 1, 2, 0, 1, 3};
    indices.copy_from_host(host.data(), static_cast<int64_t>(host.size()));
    check_embedding(layer, indices);
  }

  std::printf("grad softmax\n");
  {
    constexpr int n = 4;
    constexpr int classes = 5;
    nn::Tensor logits(ctx, {n, classes});
    const std::vector<float> host_logits = uniform(logits.numel(), rng, -1.f, 1.f);
    logits.copy_from_host(host_logits);
    const std::vector<int64_t> host_labels = {0, 2, 4, 1};
    nn::DeviceIndices labels(ctx, n);
    labels.copy_from_host(host_labels.data(), n);
    nn::LossResult result = nn::softmax_cross_entropy(ctx, logits, labels);

    double host_loss = 0.0;
    std::vector<double> host_grad(static_cast<size_t>(n * classes), 0.0);
    for (int row = 0; row < n; ++row) {
      double max_logit = -1e300;
      for (int c = 0; c < classes; ++c) {
        max_logit = std::max(max_logit, static_cast<double>(host_logits[row * classes + c]));
      }
      double sum = 0.0;
      for (int c = 0; c < classes; ++c) {
        sum += std::exp(static_cast<double>(host_logits[row * classes + c]) - max_logit);
      }
      const double log_z = max_logit + std::log(sum);
      host_loss += -(static_cast<double>(host_logits[row * classes + host_labels[row]]) - log_z);
      for (int c = 0; c < classes; ++c) {
        const double softmax =
            std::exp(static_cast<double>(host_logits[row * classes + c]) - max_logit) / sum;
        const double indicator = (c == host_labels[row]) ? 1.0 : 0.0;
        host_grad[row * classes + c] = (softmax - indicator) / static_cast<double>(n);
      }
    }
    host_loss /= static_cast<double>(n);
    check_close(host_loss, static_cast<double>(result.loss), 1e-5, -1.0, "loss", 0);
    const std::vector<float> device_grad = result.grad_logits.to_host();
    CHECK(device_grad.size() == host_grad.size());
    for (size_t i = 0; i < host_grad.size(); ++i) {
      check_close(host_grad[i], static_cast<double>(device_grad[i]), 1e-4, -1.0, "grad_logits", i);
    }
  }

  std::printf("grad transformer facade\n");
  {
    nn::TransformerConfig config;
    config.vocab_size = 10;
    config.d_model = 8;
    config.num_heads = 2;
    config.num_layers = 1;
    config.d_ff = 16;
    config.max_seq_len = 4;
    config.dropout = 0.f;
    config.bias = false;
    config.activation = nn::TransformerActivation::SwiGLU;
    config.positional = nn::PositionalEncoding::LearnedAbsolute;
    nn::Transformer model(ctx, config, 1);
    nn::IndexTensor tokens(ctx, {1, 3});
    const std::vector<int64_t> ids = {1, 0, 1};
    tokens.copy_from_host(ids.data(), static_cast<int64_t>(ids.size()));

    model.zero_grad();
    const nn::Tensor& logits = model.forward(tokens);
    nn::Tensor grad(logits.context(), logits.shape());
    grad.fill(1.f);
    model.backward(grad);

    const std::vector<nn::NamedParam> params = model.parameters();
    CHECK(params[0].name == "tok_emb.weight");
    CHECK(params[1].name == "pos_emb.weight");
    const std::vector<float> tok_grad = params[0].grad->to_host();
    const std::vector<float> pos_grad = params[1].grad->to_host();
    const size_t tok_index = argmax_abs(tok_grad, 8, 16);
    const size_t pos_index = argmax_abs(pos_grad, 0, 24);
    const auto loss = [&]() {
      const std::vector<float> values = model.forward(tokens).to_host();
      double sum = 0.0;
      for (float value : values) sum += static_cast<double>(value);
      return sum;
    };
    finite_difference_at(*params[0].value, tok_grad[tok_index], tok_index, loss, "tok_emb.weight");
    finite_difference_at(*params[1].value, pos_grad[pos_index], pos_index, loss, "pos_emb.weight");
  }

  std::printf("grad sequence softmax\n");
  {
    constexpr int n = 2;
    constexpr int t = 3;
    constexpr int classes = 5;
    constexpr int rows = n * t;
    const std::vector<float> host_logits = uniform(static_cast<int64_t>(rows) * classes, rng, -1.f, 1.f);
    const std::vector<int64_t> host_labels = {0, 2, 4, 1, 3, 0};

    nn::Tensor logits(ctx, {n, t, classes});
    logits.copy_from_host(host_logits);
    nn::IndexTensor labels(ctx, {n, t});
    labels.copy_from_host(host_labels.data(), rows);
    nn::LossResult seq = nn::sequence_cross_entropy(ctx, logits, labels);

    nn::Tensor flat_logits(ctx, {rows, classes});
    flat_logits.copy_from_host(host_logits);
    nn::DeviceIndices flat_labels(ctx, rows);
    flat_labels.copy_from_host(host_labels.data(), rows);
    nn::LossResult flat = nn::softmax_cross_entropy(ctx, flat_logits, flat_labels);

    check_close(static_cast<double>(flat.loss), static_cast<double>(seq.loss), 1e-5, -1.0, "seq loss",
                0);
    const std::vector<float> seq_grad = seq.grad_logits.to_host();
    const std::vector<float> flat_grad = flat.grad_logits.to_host();
    CHECK(seq_grad.size() == flat_grad.size());
    for (size_t i = 0; i < flat_grad.size(); ++i) {
      check_close(static_cast<double>(flat_grad[i]), static_cast<double>(seq_grad[i]), 1e-4, -1.0,
                  "seq grad_logits", i);
    }

    nn::Tensor row_logits(ctx, {rows, classes});
    row_logits.copy_from_host(host_logits);
    nn::IndexTensor row_labels(ctx, {rows});
    row_labels.copy_from_host(host_labels.data(), rows);
    nn::LossResult rows_result = nn::sequence_cross_entropy(ctx, row_logits, row_labels);
    check_close(static_cast<double>(flat.loss), static_cast<double>(rows_result.loss), 1e-5, -1.0,
                "seq rows loss", 0);
    const std::vector<float> row_grad = rows_result.grad_logits.to_host();
    CHECK(row_grad.size() == flat_grad.size());
    for (size_t i = 0; i < flat_grad.size(); ++i) {
      check_close(static_cast<double>(flat_grad[i]), static_cast<double>(row_grad[i]), 1e-4, -1.0,
                  "seq rows grad_logits", i);
    }

    // Ignored rows are exact zeros. Kept rows match softmax over the kept set, scaled by 1/n_valid.
    const std::vector<int64_t> mixed_labels = {-100, 2, 4, -100, 3, 0};
    nn::IndexTensor mixed(ctx, {n, t});
    mixed.copy_from_host(mixed_labels.data(), rows);
    nn::LossResult mixed_result = nn::sequence_cross_entropy(ctx, logits, mixed);
    std::vector<float> kept_logits;
    std::vector<int64_t> kept_labels;
    for (int row = 0; row < rows; ++row) {
      if (mixed_labels[static_cast<size_t>(row)] == -100) continue;
      kept_labels.push_back(mixed_labels[static_cast<size_t>(row)]);
      const size_t base = static_cast<size_t>(row) * classes;
      for (int c = 0; c < classes; ++c) kept_logits.push_back(host_logits[base + static_cast<size_t>(c)]);
    }
    nn::Tensor kept(ctx, {static_cast<int>(kept_labels.size()), classes});
    kept.copy_from_host(kept_logits);
    nn::DeviceIndices kept_device(ctx, static_cast<int64_t>(kept_labels.size()));
    kept_device.copy_from_host(kept_labels.data(), static_cast<int64_t>(kept_labels.size()));
    nn::LossResult kept_loss = nn::softmax_cross_entropy(ctx, kept, kept_device);
    check_close(static_cast<double>(kept_loss.loss), static_cast<double>(mixed_result.loss), 1e-5, -1.0,
                "seq mixed loss", 0);
    const std::vector<float> mixed_grad = mixed_result.grad_logits.to_host();
    const std::vector<float> kept_grad = kept_loss.grad_logits.to_host();
    int kept_row = 0;
    for (int row = 0; row < rows; ++row) {
      const size_t base = static_cast<size_t>(row) * classes;
      if (mixed_labels[static_cast<size_t>(row)] == -100) {
        for (int c = 0; c < classes; ++c) CHECK(mixed_grad[base + static_cast<size_t>(c)] == 0.f);
        continue;
      }
      const size_t src = static_cast<size_t>(kept_row) * classes;
      for (int c = 0; c < classes; ++c) {
        check_close(static_cast<double>(kept_grad[src + static_cast<size_t>(c)]),
                    static_cast<double>(mixed_grad[base + static_cast<size_t>(c)]), 1e-4, -1.0,
                    "seq mixed grad", base + static_cast<size_t>(c));
      }
      ++kept_row;
    }

    nn::IndexTensor ignored(ctx, {n, t});
    const std::vector<int64_t> all_ignored(static_cast<size_t>(rows), -100);
    ignored.copy_from_host(all_ignored.data(), rows);
    nn::LossResult zero_result = nn::sequence_cross_entropy(ctx, logits, ignored);
    CHECK(zero_result.loss == 0.f);
    for (float value : zero_result.grad_logits.to_host()) CHECK(value == 0.f);
  }

  std::printf("adamw step\n");
  {
    auto expect_invalid = [](const std::function<void()>& call) {
      bool threw = false;
      try {
        call();
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      CHECK(threw);
    };
    expect_invalid([] { nn::AdamW(std::vector<nn::AdamWGroup>{}, 0.1f, 0.9f, 0.999f, 1e-8f); });
    nn::AdamWGroup empty;
    expect_invalid([&] { nn::AdamW(std::vector<nn::AdamWGroup>{empty}, 0.1f, 0.9f, 0.999f, 1e-8f); });
    expect_invalid([] {
      nn::AdamW(std::vector<nn::NamedParam>{}, 0.1f, 0.9f, 0.999f, 1e-8f, 0.f);
    });

    nn::Tensor weight(ctx, {2});
    nn::Tensor grad_weight(ctx, {2});
    nn::Tensor bias(ctx, {1});
    nn::Tensor grad_bias(ctx, {1});
    const std::vector<float> w0{1.5f, -0.25f};
    const std::vector<float> gw{0.1f, -0.2f};
    const std::vector<float> b0{0.5f};
    const std::vector<float> gb{0.3f};
    weight.copy_from_host(w0);
    grad_weight.copy_from_host(gw);
    bias.copy_from_host(b0);
    grad_bias.copy_from_host(gb);
    nn::NamedParam weight_param{"weight", &weight, &grad_weight};
    nn::NamedParam bias_param{"bias", &bias, &grad_bias};
    expect_invalid([&] {
      nn::NamedParam missing{"weight", &weight, nullptr};
      nn::AdamW({missing}, 0.1f, 0.9f, 0.999f, 1e-8f, 0.f);
    });
    expect_invalid([&] {
      nn::Tensor wide(ctx, {3});
      nn::NamedParam mismatch{"weight", &weight, &wide};
      nn::AdamW({mismatch}, 0.1f, 0.9f, 0.999f, 1e-8f, 0.f);
    });
    expect_invalid([&] {
      nn::AdamWGroup twice;
      twice.params = {weight_param, weight_param};
      nn::AdamW(std::vector<nn::AdamWGroup>{twice}, 0.1f, 0.9f, 0.999f, 1e-8f);
    });
    expect_invalid([&] {
      nn::AdamW({weight_param}, std::numeric_limits<float>::quiet_NaN(), 0.9f, 0.999f, 1e-8f, 0.f);
    });
    expect_invalid([&] {
      nn::AdamWGroup group;
      group.params = {weight_param};
      group.weight_decay = std::numeric_limits<float>::infinity();
      nn::AdamW(std::vector<nn::AdamWGroup>{group}, 0.1f, 0.9f, 0.999f, 1e-8f);
    });

    nn::AdamWGroup decay;
    decay.weight_decay = 0.1f;
    decay.params = {weight_param};
    nn::AdamWGroup plain;
    plain.weight_decay = 0.f;
    plain.params = {bias_param};
    const float lr = 0.01f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    nn::AdamW opt(std::vector<nn::AdamWGroup>{decay, plain}, lr, beta1, beta2, eps);
    CHECK(opt.step_count() == 0);
    std::vector<float> w = w0;
    std::vector<float> b = b0;
    std::vector<float> mw(w.size(), 0.f);
    std::vector<float> vw(w.size(), 0.f);
    std::vector<float> mb(b.size(), 0.f);
    std::vector<float> vb(b.size(), 0.f);
    auto reference = [](std::vector<float>& param, std::vector<float>& m, std::vector<float>& v,
                        const std::vector<float>& grad, float step_lr, float step_beta1, float step_beta2,
                        float step_eps, float wd, float bc1, float bc2) {
      for (size_t i = 0; i < param.size(); ++i) {
        const float g = grad[i];
        const float m_new = step_beta1 * m[i] + (1.f - step_beta1) * g;
        const float v_new = step_beta2 * v[i] + (1.f - step_beta2) * g * g;
        m[i] = m_new;
        v[i] = v_new;
        const float m_hat = m_new / bc1;
        const float v_hat = v_new / bc2;
        const float update = m_hat / (std::sqrt(v_hat) + step_eps) + wd * param[i];
        param[i] = param[i] - step_lr * update;
      }
    };
    for (int step = 1; step <= 2; ++step) {
      opt.step();
      const float bc1 =
          1.f - static_cast<float>(std::pow(static_cast<double>(beta1), static_cast<double>(step)));
      const float bc2 =
          1.f - static_cast<float>(std::pow(static_cast<double>(beta2), static_cast<double>(step)));
      reference(w, mw, vw, gw, lr, beta1, beta2, eps, 0.1f, bc1, bc2);
      reference(b, mb, vb, gb, lr, beta1, beta2, eps, 0.f, bc1, bc2);
    }
    CHECK(opt.step_count() == 2);
    CHECK(opt.learning_rate() == lr);
    CHECK(opt.beta1() == beta1);
    CHECK(opt.beta2() == beta2);
    CHECK(opt.eps() == eps);
    const std::vector<float> got_w = weight.to_host();
    const std::vector<float> got_b = bias.to_host();
    const std::vector<float> got_mw = opt.m()[0].to_host();
    const std::vector<float> got_mb = opt.m()[1].to_host();
    const std::vector<float> got_vw = opt.v()[0].to_host();
    const std::vector<float> got_vb = opt.v()[1].to_host();
    CHECK(got_w.size() == w.size());
    CHECK(got_b.size() == b.size());
    for (size_t i = 0; i < w.size(); ++i) {
      check_close(w[i], got_w[i], 1e-5, 1e-6, "adamw weight", i);
      check_close(mw[i], got_mw[i], 1e-5, 1e-6, "adamw m", i);
      check_close(vw[i], got_vw[i], 1e-5, 1e-6, "adamw v", i);
    }
    check_close(b[0], got_b[0], 1e-5, 1e-6, "adamw bias", 0);
    check_close(mb[0], got_mb[0], 1e-5, 1e-6, "adamw m bias", 0);
    check_close(vb[0], got_vb[0], 1e-5, 1e-6, "adamw v bias", 0);
    CHECK(got_w[0] != w0[0]);
    opt.set_learning_rate(0.02f);
    CHECK(opt.learning_rate() == 0.02f);
    expect_invalid([&] { opt.set_learning_rate(std::numeric_limits<float>::quiet_NaN()); });
  }

  std::printf("clip grad norm\n");
  {
    CHECK(nn::clip_grad_norm({}, 1.f) == 0.f);
    nn::Tensor grad(ctx, {2});
    nn::Tensor skipped(ctx, {1});
    grad.copy_from_host({3.f, 4.f});
    skipped.copy_from_host({12.f});
    nn::NamedParam kept{"grad", &grad, &grad};
    nn::NamedParam ignored{"skip", &skipped, nullptr};
    const float open = nn::clip_grad_norm({kept, ignored}, 10.f);
    CHECK(open == 5.f);
    const std::vector<float> unchanged = grad.to_host();
    CHECK(unchanged.size() == 2);
    CHECK(unchanged[0] == 3.f);
    CHECK(unchanged[1] == 4.f);
    CHECK(skipped.to_host()[0] == 12.f);

    nn::Tensor left(ctx, {1});
    nn::Tensor right(ctx, {1});
    left.copy_from_host({3.f});
    right.copy_from_host({4.f});
    nn::NamedParam p0{"left", &left, &left};
    nn::NamedParam p1{"right", &right, &right};
    const float max_norm = 2.5f;
    const float clipped = nn::clip_grad_norm({p0, ignored, p1}, max_norm);
    CHECK(clipped == 5.f);
    const float scale = std::min(1.f, max_norm / (5.f + 1e-6f));
    CHECK(left.to_host()[0] == 3.f * scale);
    CHECK(right.to_host()[0] == 4.f * scale);
    CHECK(skipped.to_host()[0] == 12.f);

    bool threw = false;
    try {
      nn::clip_grad_norm({kept}, 0.f);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
      nn::clip_grad_norm({kept}, -1.f);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
  }

  std::printf("grads ok\n");
  return 0;
}
