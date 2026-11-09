#include "check.hpp"

#include "nn/builders.hpp"
#include "nn/conv_resnet.hpp"
#include "nn/error.hpp"
#include "nn/index_tensor.hpp"
#include "nn/init.hpp"
#include "nn/layers/conv2d.hpp"
#include "nn/layers/dropout.hpp"
#include "nn/layers/embedding.hpp"
#include "nn/layers/gelu.hpp"
#include "nn/layers/last_step.hpp"
#include "nn/layers/layer_norm.hpp"
#include "nn/layers/linear.hpp"
#include "nn/layers/multi_head_attention.hpp"
#include "nn/layers/relu.hpp"
#include "nn/layers/residual_block.hpp"
#include "nn/layers/rms_norm.hpp"
#include "nn/layers/rnn.hpp"
#include "nn/layers/swiglu.hpp"
#include "nn/layers/transformer_block.hpp"
#include "nn/loss.hpp"
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
#include <utility>
#include <vector>

namespace {

void expect_params(const nn::Network& net,
                   const std::vector<std::pair<std::string, std::vector<int>>>& expected) {
  const std::vector<nn::NamedParam> params = net.parameters();
  if (params.size() != expected.size()) {
    std::fprintf(stderr, "parameter count %zu, expected %zu\n", params.size(), expected.size());
    CHECK(false);
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (params[i].name != expected[i].first || params[i].value->shape() != expected[i].second) {
      std::fprintf(stderr, "param %zu got %s rank %d, expected %s\n", i, params[i].name.c_str(),
                   params[i].value->rank(), expected[i].first.c_str());
      CHECK(params[i].name == expected[i].first);
      CHECK(params[i].value->shape() == expected[i].second);
    }
  }
}

bool has_param(const nn::Network& net, const std::string& name) {
  for (const nn::NamedParam& param : net.parameters()) {
    if (param.name == name) return true;
  }
  return false;
}

nn::Tensor* find_param(nn::Network& net, const std::string& name) {
  for (nn::NamedParam& param : net.parameters()) {
    if (param.name == name) return param.value;
  }
  return nullptr;
}

void check_io(nn::Context& ctx, nn::Network& net, const std::vector<int>& input_shape,
              const std::vector<int>& output_shape) {
  net.zero_grad();
  nn::Tensor input(ctx, input_shape);
  input.fill(0.1f);
  const nn::Tensor& output = net.forward(input);
  CHECK(output.shape() == output_shape);
  nn::Tensor grad_output(ctx, output.shape());
  grad_output.fill(1.f);
  const nn::Tensor& grad_input = net.backward(grad_output);
  CHECK(grad_input.shape() == input_shape);
}

void expect_error(const std::function<void()>& call, const std::string& message) {
  bool threw = false;
  try {
    call();
  } catch (const nn::Error& error) {
    threw = true;
    if (message != error.what()) {
      std::fprintf(stderr, "expected \"%s\", got \"%s\"\n", message.c_str(), error.what());
      CHECK(false);
    }
  }
  CHECK(threw);
}

void expect_invalid(const std::function<void()>& call, const std::string& message) {
  bool threw = false;
  try {
    call();
  } catch (const std::invalid_argument& error) {
    threw = true;
    if (message != error.what()) {
      std::fprintf(stderr, "expected \"%s\", got \"%s\"\n", message.c_str(), error.what());
      CHECK(false);
    }
  }
  CHECK(threw);
}

void expect_close(const std::vector<float>& got, const std::vector<float>& want, const char* name) {
  CHECK(got.size() == want.size());
  for (size_t i = 0; i < got.size(); ++i) {
    const double diff = std::abs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
    const double scale = std::max(1.0, std::max(std::abs(static_cast<double>(got[i])),
                                                std::abs(static_cast<double>(want[i]))));
    if (diff / scale <= 1e-5 || diff <= 1e-5) continue;
    std::fprintf(stderr, "%s[%zu] got %.8g want %.8g\n", name, i, static_cast<double>(got[i]),
                 static_cast<double>(want[i]));
    CHECK(false);
  }
}

void check_sequence_case(nn::Context& ctx, const std::vector<int>& logit_shape,
                         const std::vector<float>& host_logits, const std::vector<int64_t>& host_labels,
                         int64_t ignore_index) {
  CHECK(logit_shape.size() == 2 || logit_shape.size() == 3);
  const int classes = logit_shape.back();
  const int rows = logit_shape.size() == 2 ? logit_shape[0] : logit_shape[0] * logit_shape[1];
  CHECK(classes > 1);
  CHECK(static_cast<int>(host_logits.size()) == rows * classes);
  CHECK(static_cast<int>(host_labels.size()) == rows);

  nn::Tensor logits(ctx, logit_shape);
  logits.copy_from_host(host_logits);
  std::vector<int> label_shape;
  if (logit_shape.size() == 3) label_shape = {logit_shape[0], logit_shape[1]};
  else label_shape = {rows};
  nn::IndexTensor labels(ctx, label_shape);
  labels.copy_from_host(host_labels.data(), static_cast<int64_t>(host_labels.size()));

  nn::LossResult seq = nn::sequence_cross_entropy(ctx, logits, labels, ignore_index);
  CHECK(seq.grad_logits.shape() == logit_shape);
  const std::vector<float> seq_grad = seq.grad_logits.to_host();

  std::vector<float> valid_logits;
  std::vector<int64_t> valid_labels;
  for (int row = 0; row < rows; ++row) {
    if (host_labels[static_cast<size_t>(row)] == ignore_index) continue;
    valid_labels.push_back(host_labels[static_cast<size_t>(row)]);
    const size_t base = static_cast<size_t>(row) * static_cast<size_t>(classes);
    for (int c = 0; c < classes; ++c) valid_logits.push_back(host_logits[base + static_cast<size_t>(c)]);
  }

  if (valid_labels.empty()) {
    CHECK(seq.loss == 0.f);
    for (float value : seq_grad) CHECK(value == 0.f);
    return;
  }

  nn::Tensor flat(ctx, {static_cast<int>(valid_labels.size()), classes});
  flat.copy_from_host(valid_logits);
  nn::DeviceIndices device_labels(ctx, static_cast<int64_t>(valid_labels.size()));
  device_labels.copy_from_host(valid_labels.data(), static_cast<int64_t>(valid_labels.size()));
  nn::LossResult flat_loss = nn::softmax_cross_entropy(ctx, flat, device_labels);
  const std::vector<float> got_loss{seq.loss};
  const std::vector<float> want_loss{flat_loss.loss};
  expect_close(got_loss, want_loss, "sequence loss");

  const std::vector<float> flat_grad = flat_loss.grad_logits.to_host();
  std::vector<float> want(static_cast<size_t>(rows) * static_cast<size_t>(classes), 0.f);
  int valid_row = 0;
  for (int row = 0; row < rows; ++row) {
    const size_t base = static_cast<size_t>(row) * static_cast<size_t>(classes);
    if (host_labels[static_cast<size_t>(row)] == ignore_index) {
      for (int c = 0; c < classes; ++c) CHECK(seq_grad[base + static_cast<size_t>(c)] == 0.f);
      continue;
    }
    const size_t src = static_cast<size_t>(valid_row) * static_cast<size_t>(classes);
    for (int c = 0; c < classes; ++c) want[base + static_cast<size_t>(c)] = flat_grad[src + static_cast<size_t>(c)];
    ++valid_row;
  }
  expect_close(seq_grad, want, "sequence grad");
}

float gelu_ref(float x) {
  return 0.5f * x * (1.f + std::erff(x * 0.7071067811865476f));
}

float silu_ref(float x) {
  const float sigmoid = (x >= 0.f) ? 1.f / (1.f + std::exp(-x)) : std::exp(x) / (1.f + std::exp(x));
  return x * sigmoid;
}

struct LayerNormRef {
  std::vector<float> y;
  std::vector<float> dx;
  std::vector<float> dgamma;
  std::vector<float> dbeta;
};

LayerNormRef layer_norm_ref(const std::vector<float>& x, const std::vector<float>& gamma,
                            const std::vector<float>& beta, const std::vector<float>& dy, int rows,
                            int cols, float eps) {
  LayerNormRef ref;
  ref.y.resize(x.size());
  ref.dx.resize(x.size());
  ref.dgamma.assign(static_cast<size_t>(cols), 0.f);
  ref.dbeta.assign(static_cast<size_t>(cols), 0.f);
  std::vector<double> dgamma(static_cast<size_t>(cols), 0.0);
  std::vector<double> dbeta(static_cast<size_t>(cols), 0.0);
  for (int row = 0; row < rows; ++row) {
    const size_t base = static_cast<size_t>(row) * static_cast<size_t>(cols);
    double mean = 0.0;
    for (int col = 0; col < cols; ++col) mean += static_cast<double>(x[base + static_cast<size_t>(col)]);
    mean /= static_cast<double>(cols);
    double var = 0.0;
    for (int col = 0; col < cols; ++col) {
      const double diff = static_cast<double>(x[base + static_cast<size_t>(col)]) - mean;
      var += diff * diff;
    }
    var /= static_cast<double>(cols);
    const double inv = 1.0 / std::sqrt(var + static_cast<double>(eps));
    double sum_dhat = 0.0;
    double sum_dhat_xhat = 0.0;
    std::vector<double> xhat(static_cast<size_t>(cols));
    std::vector<double> dhat(static_cast<size_t>(cols));
    for (int col = 0; col < cols; ++col) {
      xhat[static_cast<size_t>(col)] =
          (static_cast<double>(x[base + static_cast<size_t>(col)]) - mean) * inv;
      ref.y[base + static_cast<size_t>(col)] = static_cast<float>(
          xhat[static_cast<size_t>(col)] * static_cast<double>(gamma[static_cast<size_t>(col)]) +
          static_cast<double>(beta[static_cast<size_t>(col)]));
      dhat[static_cast<size_t>(col)] = static_cast<double>(dy[base + static_cast<size_t>(col)]) *
                                       static_cast<double>(gamma[static_cast<size_t>(col)]);
      sum_dhat += dhat[static_cast<size_t>(col)];
      sum_dhat_xhat += dhat[static_cast<size_t>(col)] * xhat[static_cast<size_t>(col)];
    }
    const double mean_dhat = sum_dhat / static_cast<double>(cols);
    const double mean_dhat_xhat = sum_dhat_xhat / static_cast<double>(cols);
    for (int col = 0; col < cols; ++col) {
      ref.dx[base + static_cast<size_t>(col)] = static_cast<float>(
          inv * (dhat[static_cast<size_t>(col)] - mean_dhat -
                 xhat[static_cast<size_t>(col)] * mean_dhat_xhat));
      dgamma[static_cast<size_t>(col)] +=
          static_cast<double>(dy[base + static_cast<size_t>(col)]) * xhat[static_cast<size_t>(col)];
      dbeta[static_cast<size_t>(col)] += static_cast<double>(dy[base + static_cast<size_t>(col)]);
    }
  }
  for (int col = 0; col < cols; ++col) {
    ref.dgamma[static_cast<size_t>(col)] = static_cast<float>(dgamma[static_cast<size_t>(col)]);
    ref.dbeta[static_cast<size_t>(col)] = static_cast<float>(dbeta[static_cast<size_t>(col)]);
  }
  return ref;
}

std::vector<float> swiglu_ref(const std::vector<float>& input, int rows, int inner) {
  std::vector<float> output(static_cast<size_t>(rows) * static_cast<size_t>(inner));
  for (int row = 0; row < rows; ++row) {
    const float* gate = input.data() + static_cast<size_t>(row) * static_cast<size_t>(inner) * 2;
    const float* value = gate + inner;
    for (int col = 0; col < inner; ++col) {
      output[static_cast<size_t>(row) * static_cast<size_t>(inner) + static_cast<size_t>(col)] =
          silu_ref(gate[col]) * value[col];
    }
  }
  return output;
}

void check_block(nn::Context& ctx, int in_channels, int out_channels, int stride,
                 const std::vector<int>& input_shape, const std::vector<int>& output_shape,
                 bool expect_proj) {
  std::mt19937 rng(1);
  nn::ResidualBlock block(ctx, in_channels, out_channels, stride, rng);
  block.zero_grad();
  nn::Tensor input(ctx, input_shape);
  input.fill(0.1f);
  const nn::Tensor& output = block.forward(input);
  CHECK(output.shape() == output_shape);
  nn::Tensor grad_output(ctx, output.shape());
  grad_output.fill(1.f);
  const nn::Tensor& grad_input = block.backward(grad_output);
  CHECK(grad_input.shape() == input_shape);
  bool found = false;
  for (const nn::NamedParam& param : block.parameters()) {
    if (param.name == "proj.weight") found = true;
  }
  CHECK(found == expect_proj);
}

std::vector<float> linear_ref(const std::vector<float>& x, int rows, int in_f, int out_f,
                              const std::vector<float>& weight, const std::vector<float>* bias) {
  std::vector<float> y(static_cast<size_t>(rows) * static_cast<size_t>(out_f));
  for (int r = 0; r < rows; ++r) {
    for (int o = 0; o < out_f; ++o) {
      double acc = bias == nullptr ? 0.0 : static_cast<double>((*bias)[static_cast<size_t>(o)]);
      for (int c = 0; c < in_f; ++c) {
        acc += static_cast<double>(x[static_cast<size_t>(r * in_f + c)]) *
               static_cast<double>(weight[static_cast<size_t>(o * in_f + c)]);
      }
      y[static_cast<size_t>(r * out_f + o)] = static_cast<float>(acc);
    }
  }
  return y;
}

void rope_adjacent(std::vector<float>& values, int n, int time, int heads, int dh) {
  for (int batch = 0; batch < n; ++batch) {
    for (int t = 0; t < time; ++t) {
      for (int h = 0; h < heads; ++h) {
        for (int pair = 0; pair < dh / 2; ++pair) {
          const double theta = static_cast<double>(t) *
                               std::pow(10000.0, -2.0 * static_cast<double>(pair) / static_cast<double>(dh));
          const double c = std::cos(theta);
          const double s = std::sin(theta);
          const size_t i0 =
              static_cast<size_t>((((batch * time + t) * heads + h) * dh) + pair * 2);
          const double x0 = values[i0];
          const double x1 = values[i0 + 1];
          values[i0] = static_cast<float>(x0 * c - x1 * s);
          values[i0 + 1] = static_cast<float>(x0 * s + x1 * c);
        }
      }
    }
  }
}

std::vector<float> mha_ref(const std::vector<float>& x, int n, int time, int d_model, int heads,
                           bool causal, const std::vector<float>& wq, const std::vector<float>* bq,
                           const std::vector<float>& wk, const std::vector<float>* bk,
                           const std::vector<float>& wv, const std::vector<float>* bv,
                           const std::vector<float>& wo, const std::vector<float>* bo,
                           bool rope = false) {
  const int dh = d_model / heads;
  const int rows = n * time;
  std::vector<float> q = linear_ref(x, rows, d_model, d_model, wq, bq);
  std::vector<float> k = linear_ref(x, rows, d_model, d_model, wk, bk);
  const std::vector<float> v = linear_ref(x, rows, d_model, d_model, wv, bv);
  if (rope) {
    rope_adjacent(q, n, time, heads, dh);
    rope_adjacent(k, n, time, heads, dh);
  }
  const double scale = 1.0 / std::sqrt(static_cast<double>(dh));
  const double mask = static_cast<double>(-1.0e9f);
  std::vector<float> mixed(static_cast<size_t>(rows) * static_cast<size_t>(d_model), 0.f);
  for (int batch = 0; batch < n; ++batch) {
    for (int h = 0; h < heads; ++h) {
      std::vector<double> scores(static_cast<size_t>(time * time));
      for (int i = 0; i < time; ++i) {
        for (int j = 0; j < time; ++j) {
          double dot = 0.0;
          for (int e = 0; e < dh; ++e) {
            const size_t q_i = static_cast<size_t>((((batch * time + i) * heads + h) * dh) + e);
            const size_t k_i = static_cast<size_t>((((batch * time + j) * heads + h) * dh) + e);
            dot += static_cast<double>(q[q_i]) * static_cast<double>(k[k_i]);
          }
          double value = dot * scale;
          if (causal && j > i) value = mask;
          scores[static_cast<size_t>(i * time + j)] = value;
        }
      }
      for (int i = 0; i < time; ++i) {
        double max_s = scores[static_cast<size_t>(i * time)];
        for (int j = 1; j < time; ++j) {
          max_s = std::max(max_s, scores[static_cast<size_t>(i * time + j)]);
        }
        double sum = 0.0;
        for (int j = 0; j < time; ++j) {
          sum += std::exp(scores[static_cast<size_t>(i * time + j)] - max_s);
        }
        for (int e = 0; e < dh; ++e) {
          double acc = 0.0;
          for (int j = 0; j < time; ++j) {
            const double prob = std::exp(scores[static_cast<size_t>(i * time + j)] - max_s) / sum;
            const size_t v_i = static_cast<size_t>((((batch * time + j) * heads + h) * dh) + e);
            acc += prob * static_cast<double>(v[v_i]);
          }
          mixed[static_cast<size_t>((((batch * time + i) * heads + h) * dh) + e)] =
              static_cast<float>(acc);
        }
      }
    }
  }
  return linear_ref(mixed, rows, d_model, d_model, wo, bo);
}

std::vector<float> host_param(const nn::Layer& layer, const std::string& name) {
  for (const nn::NamedParam& param : layer.parameters()) {
    if (param.name == name) return param.value->to_host();
  }
  CHECK(false);
  return {};
}

const std::vector<float>* bias_of(const nn::Layer& layer, const std::string& name,
                                  std::vector<float>& storage) {
  for (const nn::NamedParam& param : layer.parameters()) {
    if (param.name == name) {
      storage = param.value->to_host();
      return &storage;
    }
  }
  return nullptr;
}

void set_param(nn::Layer& layer, const std::string& name, const std::vector<float>& values) {
  for (nn::NamedParam param : layer.parameters()) {
    if (param.name == name) {
      param.value->copy_from_host(values);
      return;
    }
  }
  CHECK(false);
}

void expect_mha_matches(nn::MultiHeadAttention& layer, const nn::Tensor& input, int n, int time,
                        int d_model, int heads, bool causal, const char* name, bool rope = false) {
  std::vector<float> bq;
  std::vector<float> bk;
  std::vector<float> bv;
  std::vector<float> bo;
  const std::vector<float> want = mha_ref(
      input.to_host(), n, time, d_model, heads, causal, host_param(layer, "q.weight"),
      bias_of(layer, "q.bias", bq), host_param(layer, "k.weight"), bias_of(layer, "k.bias", bk),
      host_param(layer, "v.weight"), bias_of(layer, "v.bias", bv), host_param(layer, "out.weight"),
      bias_of(layer, "out.bias", bo), rope);
  const nn::Tensor& output = layer.forward(input);
  CHECK(output.shape() == std::vector<int>({n, time, d_model}));
  expect_close(output.to_host(), want, name);
}

std::vector<float> eye(int d) {
  std::vector<float> weight(static_cast<size_t>(d * d), 0.f);
  for (int i = 0; i < d; ++i) weight[static_cast<size_t>(i * d + i)] = 1.f;
  return weight;
}

void check_multi_head_attention(nn::Context& ctx, std::mt19937& rng) {
  expect_invalid([&] { nn::MultiHeadAttention(ctx, 3, 1, true, false, true, rng); }, "use_rope");
  expect_error([&] { nn::MultiHeadAttention(ctx, 4, 3, true, false, false, rng); },
               "MultiHeadAttention size");
  expect_error([&] { nn::MultiHeadAttention(ctx, 4, 0, true, false, false, rng); },
               "MultiHeadAttention size");
  expect_error([&] { nn::MultiHeadAttention(ctx, 0, 1, false, false, false, rng); },
               "MultiHeadAttention size");
  expect_error([&] { nn::MultiHeadAttention(ctx, -2, 1, false, true, false, rng); },
               "MultiHeadAttention size");

  {
    nn::MultiHeadAttention layer(ctx, 4, 2, true, true, false, rng);
    const std::vector<nn::NamedParam> params = layer.parameters();
    const std::vector<std::string> names = {"q.weight", "q.bias", "k.weight", "k.bias",
                                            "v.weight", "v.bias", "out.weight", "out.bias"};
    CHECK(params.size() == names.size());
    for (size_t i = 0; i < names.size(); ++i) {
      CHECK(params[i].name == names[i]);
      if (names[i].find("weight") != std::string::npos) {
        CHECK(params[i].value->shape() == std::vector<int>({4, 4}));
      } else {
        CHECK(params[i].value->shape() == std::vector<int>({4}));
      }
    }
    nn::Tensor input(ctx, {2, 3, 4});
    input.copy_from_host(std::vector<float>{
        0.2f, -0.4f, 0.1f, 0.3f,  -0.2f, 0.5f,  0.0f,  -0.1f, 0.4f, 0.2f, -0.3f, 0.1f,
        0.1f, 0.0f,  -0.2f, 0.4f, 0.3f,  -0.1f, 0.2f, 0.0f,  -0.4f, 0.2f, 0.1f,  -0.3f});
    set_param(layer, "q.bias", {0.2f, -0.1f, 0.05f, 0.3f});
    set_param(layer, "k.bias", {0.1f, 0.0f, -0.2f, 0.4f});
    set_param(layer, "v.bias", {-0.3f, 0.2f, 0.1f, -0.1f});
    set_param(layer, "out.bias", {0.05f, -0.05f, 0.15f, -0.25f});
    expect_mha_matches(layer, input, 2, 3, 4, 2, true, "mha-causal-bias");
    nn::Tensor grad(ctx, input.shape());
    grad.fill(1.f);
    const std::vector<float> grad_before = grad.to_host();
    CHECK(layer.backward(grad).shape() == input.shape());
    CHECK(grad.to_host() == grad_before);
    bool nonzero = false;
    for (float value : params[0].grad->to_host()) {
      if (value != 0.f) nonzero = true;
    }
    CHECK(nonzero);
    layer.zero_grad();
    for (const nn::NamedParam& param : layer.parameters()) {
      for (float value : param.grad->to_host()) CHECK(value == 0.f);
    }
  }

  {
    nn::MultiHeadAttention layer(ctx, 4, 2, false, false, false, rng);
    const std::vector<nn::NamedParam> params = layer.parameters();
    CHECK(params.size() == 4);
    CHECK(params[0].name == "q.weight");
    CHECK(params[1].name == "k.weight");
    CHECK(params[2].name == "v.weight");
    CHECK(params[3].name == "out.weight");
    nn::Tensor input(ctx, {2, 3, 4});
    input.copy_from_host(std::vector<float>{
        0.2f, -0.4f, 0.1f, 0.3f,  -0.2f, 0.5f,  0.0f,  -0.1f, 0.4f, 0.2f, -0.3f, 0.1f,
        0.1f, 0.0f,  -0.2f, 0.4f, 0.3f,  -0.1f, 0.2f, 0.0f,  -0.4f, 0.2f, 0.1f,  -0.3f});
    expect_mha_matches(layer, input, 2, 3, 4, 2, false, "mha-plain");
    nn::Tensor bad_rank(ctx, {2, 4});
    expect_error([&] { layer.forward(bad_rank); }, "MultiHeadAttention input shape");
    nn::Tensor bad_last(ctx, {2, 3, 3});
    expect_error([&] { layer.forward(bad_last); }, "MultiHeadAttention input shape");
    nn::Tensor rank4(ctx, {1, 1, 3, 4});
    expect_error([&] { layer.forward(rank4); }, "MultiHeadAttention input shape");
    layer.forward(input);
    nn::Tensor bad_grad(ctx, {2, 2, 4});
    expect_error([&] { layer.backward(bad_grad); }, "MultiHeadAttention backward shape");
  }

  {
    nn::MultiHeadAttention fresh(ctx, 4, 2, true, false, false, rng);
    nn::Tensor grad(ctx, {1, 2, 4});
    expect_error([&] { fresh.backward(grad); }, "MultiHeadAttention backward shape");
  }

  // H = 1 and V = W_out = I make the output the post-softmax matrix when the input is I.
  {
    constexpr int d = 3;
    const std::vector<float> ident = eye(d);
    nn::MultiHeadAttention causal(ctx, d, 1, true, false, false, rng);
    for (const char* name : {"q.weight", "k.weight", "v.weight", "out.weight"}) {
      set_param(causal, name, ident);
    }
    nn::Tensor input(ctx, {1, d, d});
    input.copy_from_host(ident);
    expect_mha_matches(causal, input, 1, d, d, 1, true, "mha-causal-eye");
    const std::vector<float> y = causal.forward(input).to_host();
    CHECK(std::fabs(y[0] - 1.f) <= 1e-5f);
    for (int i = 0; i < d; ++i) {
      for (int j = i + 1; j < d; ++j) CHECK(std::fabs(y[static_cast<size_t>(i * d + j)]) <= 1e-5f);
    }
    CHECK(std::fabs(y[3]) > 1e-3f);

    nn::MultiHeadAttention plain(ctx, d, 1, false, false, false, rng);
    for (const char* name : {"q.weight", "k.weight", "v.weight", "out.weight"}) {
      set_param(plain, name, ident);
    }
    expect_mha_matches(plain, input, 1, d, d, 1, false, "mha-plain-eye");
    const std::vector<float> open = plain.forward(input).to_host();
    CHECK(std::fabs(open[1]) > 1e-3f);
  }

  {
    nn::MultiHeadAttention layer(ctx, 2, 2, true, false, false, rng);
    nn::Tensor input(ctx, {1, 1, 2});
    input.copy_from_host(std::vector<float>{0.25f, -0.5f});
    expect_mha_matches(layer, input, 1, 1, 2, 2, true, "mha-t1");
    CHECK(layer.forward(input).shape() == std::vector<int>({1, 1, 2}));
  }

  {
    nn::MultiHeadAttention layer(ctx, 4, 2, true, false, true, rng);
    nn::Tensor input(ctx, {1, 3, 4});
    input.copy_from_host(std::vector<float>{0.2f, -0.4f, 0.1f, 0.3f, -0.2f, 0.5f, -0.1f, 0.4f,
                                            0.3f, -0.3f, 0.2f, 0.1f});
    expect_mha_matches(layer, input, 1, 3, 4, 2, true, "mha-rope", true);
  }
}

void expect_named(const nn::Layer& layer,
                   const std::vector<std::pair<std::string, std::vector<int>>>& expected) {
  const std::vector<nn::NamedParam> params = layer.parameters();
  CHECK(params.size() == expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    CHECK(params[i].name == expected[i].first);
    CHECK(params[i].value->shape() == expected[i].second);
    CHECK(params[i].grad != nullptr);
    CHECK(params[i].grad->shape() == expected[i].second);
  }
}

nn::TransformerBlockSpec tiny_block_spec() {
  nn::TransformerBlockSpec spec;
  spec.d_model = 8;
  spec.num_heads = 2;
  spec.d_ff = 16;
  spec.dropout = 0.f;
  spec.bias = false;
  spec.use_rope = false;
  spec.activation = nn::TransformerActivation::SwiGLU;
  return spec;
}

void check_repeated(nn::TransformerBlock& block, nn::Tensor& input) {
  const std::vector<float> first = block.forward(input).to_host();
  const std::vector<float> second = block.forward(input).to_host();
  CHECK(first.size() == static_cast<size_t>(input.numel()));
  CHECK(first == second);
  for (float value : first) CHECK(std::isfinite(value));
}

void check_transformer_block(nn::Context& ctx, std::mt19937& rng) {
  {
    nn::TransformerBlockSpec spec = tiny_block_spec();
    spec.d_model = 0;
    expect_error([&] { nn::TransformerBlock(ctx, spec, rng); }, "TransformerBlock size");
    spec = tiny_block_spec();
    spec.d_model = -1;
    expect_error([&] { nn::TransformerBlock(ctx, spec, rng); }, "TransformerBlock size");
    spec = tiny_block_spec();
    spec.num_heads = 0;
    expect_error([&] { nn::TransformerBlock(ctx, spec, rng); }, "TransformerBlock size");
    spec = tiny_block_spec();
    spec.num_heads = -1;
    expect_error([&] { nn::TransformerBlock(ctx, spec, rng); }, "TransformerBlock size");
    spec = tiny_block_spec();
    spec.num_heads = 3;
    expect_error([&] { nn::TransformerBlock(ctx, spec, rng); }, "TransformerBlock size");
    spec = tiny_block_spec();
    spec.d_ff = 0;
    expect_error([&] { nn::TransformerBlock(ctx, spec, rng); }, "TransformerBlock size");
    spec = tiny_block_spec();
    spec.d_ff = -4;
    expect_error([&] { nn::TransformerBlock(ctx, spec, rng); }, "TransformerBlock size");
    spec = tiny_block_spec();
    spec.dropout = 1.f;
    expect_error([&] { nn::TransformerBlock(ctx, spec, rng); }, "TransformerBlock dropout");
    spec = tiny_block_spec();
    spec.dropout = -0.1f;
    expect_error([&] { nn::TransformerBlock(ctx, spec, rng); }, "TransformerBlock dropout");
    spec = tiny_block_spec();
    spec.dropout = std::nanf("");
    expect_error([&] { nn::TransformerBlock(ctx, spec, rng); }, "TransformerBlock dropout");
    spec = tiny_block_spec();
    spec.activation = static_cast<nn::TransformerActivation>(0);
    expect_error([&] { nn::TransformerBlock(ctx, spec, rng); }, "TransformerBlock activation");
    spec = tiny_block_spec();
    spec.d_model = 6;
    spec.num_heads = 2;
    spec.use_rope = true;
    expect_invalid([&] { nn::TransformerBlock(ctx, spec, rng); }, "use_rope");
  }

  nn::Tensor input(ctx, {1, 3, 8});
  input.fill(0.1f);
  {
    nn::TransformerBlock block(ctx, tiny_block_spec(), rng);
    const std::vector<std::pair<std::string, std::vector<int>>> names = {
        {"ln1.weight", {8}},
        {"ln1.bias", {8}},
        {"attn.q.weight", {8, 8}},
        {"attn.k.weight", {8, 8}},
        {"attn.v.weight", {8, 8}},
        {"attn.out.weight", {8, 8}},
        {"ln2.weight", {8}},
        {"ln2.bias", {8}},
        {"fc1.weight", {32, 8}},
        {"fc2.weight", {8, 16}},
    };
    expect_named(block, names);
    CHECK(block.attn().parameters()[0].name == "q.weight");
    CHECK(block.attn().parameters()[0].value == block.parameters()[2].value);
    const nn::TransformerBlock& cref = block;
    CHECK(&cref.attn() == &block.attn());
    const nn::Tensor& output = block.forward(input);
    CHECK(output.shape() == std::vector<int>({1, 3, 8}));
    for (float value : output.to_host()) CHECK(std::isfinite(value));
    check_repeated(block, input);
    nn::Tensor grad(ctx, output.shape());
    grad.fill(1.f);
    CHECK(block.backward(grad).shape() == input.shape());
    block.zero_grad();
    for (const nn::NamedParam& param : block.parameters()) {
      for (float value : param.grad->to_host()) CHECK(value == 0.f);
    }
    nn::Tensor bad_rank(ctx, {3, 8});
    expect_error([&] { block.forward(bad_rank); }, "TransformerBlock input shape");
    nn::Tensor bad_last(ctx, {1, 3, 4});
    expect_error([&] { block.forward(bad_last); }, "TransformerBlock input shape");
    block.forward(input);
    nn::Tensor bad_grad(ctx, {1, 2, 8});
    expect_error([&] { block.backward(bad_grad); }, "TransformerBlock backward shape");
    nn::TransformerBlock fresh(ctx, tiny_block_spec(), rng);
    expect_error([&] { fresh.backward(grad); }, "TransformerBlock backward shape");
  }

  {
    nn::TransformerBlockSpec spec = tiny_block_spec();
    spec.bias = true;
    nn::TransformerBlock block(ctx, spec, rng);
    expect_named(block, {{"ln1.weight", {8}},
                         {"ln1.bias", {8}},
                         {"attn.q.weight", {8, 8}},
                         {"attn.q.bias", {8}},
                         {"attn.k.weight", {8, 8}},
                         {"attn.k.bias", {8}},
                         {"attn.v.weight", {8, 8}},
                         {"attn.v.bias", {8}},
                         {"attn.out.weight", {8, 8}},
                         {"attn.out.bias", {8}},
                         {"ln2.weight", {8}},
                         {"ln2.bias", {8}},
                         {"fc1.weight", {32, 8}},
                         {"fc1.bias", {32}},
                         {"fc2.weight", {8, 16}},
                         {"fc2.bias", {8}}});
    CHECK(block.forward(input).shape() == std::vector<int>({1, 3, 8}));
  }

  {
    nn::TransformerBlockSpec spec = tiny_block_spec();
    spec.activation = nn::TransformerActivation::Gelu;
    nn::TransformerBlock block(ctx, spec, rng);
    expect_named(block, {{"ln1.weight", {8}},
                         {"ln1.bias", {8}},
                         {"attn.q.weight", {8, 8}},
                         {"attn.k.weight", {8, 8}},
                         {"attn.v.weight", {8, 8}},
                         {"attn.out.weight", {8, 8}},
                         {"ln2.weight", {8}},
                         {"ln2.bias", {8}},
                         {"fc1.weight", {16, 8}},
                         {"fc2.weight", {8, 16}}});
    CHECK(block.forward(input).shape() == std::vector<int>({1, 3, 8}));
    for (float value : block.forward(input).to_host()) CHECK(std::isfinite(value));
  }

  {
    nn::TransformerBlockSpec spec = tiny_block_spec();
    spec.use_rope = true;
    nn::TransformerBlock block(ctx, spec, rng);
    expect_named(block, {{"ln1.weight", {8}},
                         {"attn.q.weight", {8, 8}},
                         {"attn.k.weight", {8, 8}},
                         {"attn.v.weight", {8, 8}},
                         {"attn.out.weight", {8, 8}},
                         {"ln2.weight", {8}},
                         {"fc1.weight", {32, 8}},
                         {"fc2.weight", {8, 16}}});
    CHECK(block.parameters()[1].name == "attn.q.weight");
    CHECK(block.forward(input).shape() == std::vector<int>({1, 3, 8}));
    for (const nn::NamedParam& param : block.parameters()) {
      CHECK(param.name.find(".bias") == std::string::npos);
    }
  }

  {
    nn::TransformerBlockSpec spec = tiny_block_spec();
    spec.dropout = 0.5f;
    nn::TransformerBlock block(ctx, spec, rng);
    block.set_training(true);
    block.forward(input);
    block.set_training(false);
    check_repeated(block, input);
  }
}

std::vector<float> position_row(const std::vector<float>& logits, int batch, int time, int index,
                                int vocab) {
  const size_t offset =
      (static_cast<size_t>(batch) * static_cast<size_t>(time) + static_cast<size_t>(index)) *
      static_cast<size_t>(vocab);
  return std::vector<float>(logits.begin() + static_cast<std::ptrdiff_t>(offset),
                            logits.begin() + static_cast<std::ptrdiff_t>(offset + static_cast<size_t>(vocab)));
}

void check_cached_decode(nn::Context& ctx, std::mt19937& rng) {
  {
    nn::MultiHeadAttention layer(ctx, 8, 2, true, false, false, rng);
    nn::MultiHeadAttention::Cache cache{nn::Tensor(ctx, {1, 1, 1, 1}), nn::Tensor(ctx, {1, 1, 1, 1}), 0};
    expect_invalid([&] { layer.reset_cache(cache, 0, 4); }, "batch");
    expect_invalid([&] { layer.reset_cache(cache, 1, 0); }, "t_max");
    layer.reset_cache(cache, 1, 3);
    CHECK(cache.k.shape() == std::vector<int>({1, 2, 3, 4}));
    CHECK(cache.v.shape() == cache.k.shape());
    CHECK(cache.t == 0);

    nn::Tensor sequence(ctx, {1, 3, 8});
    std::mt19937 data_rng(3);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> host(static_cast<size_t>(sequence.numel()));
    for (float& value : host) value = dist(data_rng);
    sequence.copy_from_host(host);
    const std::vector<float> full = layer.forward(sequence).to_host();
    for (int t = 0; t < 3; ++t) {
      nn::Tensor step(ctx, {1, 1, 8});
      step.copy_from_host(std::vector<float>(host.begin() + t * 8, host.begin() + (t + 1) * 8));
      const std::vector<float> got = layer.forward_cached(step, cache).to_host();
      expect_close(got, position_row(full, 0, 3, t, 8), "mha-cache");
      CHECK(cache.t == t + 1);
    }
    nn::MultiHeadAttention rope(ctx, 8, 2, true, false, true, rng);
    nn::MultiHeadAttention::Cache rope_cache{nn::Tensor(ctx, {1, 1, 1, 1}),
                                             nn::Tensor(ctx, {1, 1, 1, 1}), 0};
    rope.reset_cache(rope_cache, 1, 3);
    const std::vector<float> rope_full = rope.forward(sequence).to_host();
    for (int t = 0; t < 3; ++t) {
      nn::Tensor step(ctx, {1, 1, 8});
      step.copy_from_host(std::vector<float>(host.begin() + t * 8, host.begin() + (t + 1) * 8));
      const std::vector<float> got = rope.forward_cached(step, rope_cache).to_host();
      expect_close(got, position_row(rope_full, 0, 3, t, 8), "mha-rope-cache");
      CHECK(rope_cache.t == t + 1);
    }
    nn::Tensor extra(ctx, {1, 1, 8});
    extra.fill(0.1f);
    expect_invalid([&] { layer.forward_cached(extra, cache); }, "cache");
    nn::Tensor bad_time(ctx, {1, 2, 8});
    expect_invalid([&] { layer.forward_cached(bad_time, cache); }, "x_t");
    nn::Tensor grad(ctx, {1, 1, 8});
    grad.fill(1.f);
    expect_invalid([&] { layer.backward(grad); }, "forward_cached");
    const nn::Tensor& again = layer.forward(sequence);
    nn::Tensor grad_full(ctx, again.shape());
    grad_full.fill(1.f);
    CHECK(layer.backward(grad_full).shape() == sequence.shape());
  }

  nn::TransformerConfig config;
  config.vocab_size = 10;
  config.d_model = 8;
  config.num_heads = 2;
  config.num_layers = 2;
  config.d_ff = 16;
  config.max_seq_len = 8;
  config.dropout = 0.f;
  config.bias = false;
  config.activation = nn::TransformerActivation::SwiGLU;
  config.positional = nn::PositionalEncoding::LearnedAbsolute;
  nn::Transformer model(ctx, config, 7);
  model.eval();

  std::mt19937 prompt_rng(11);
  std::uniform_int_distribution<int64_t> pick(0, config.vocab_size - 1);
  std::vector<int64_t> prompt(3);
  for (int64_t& id : prompt) id = pick(prompt_rng);

  for (int length = 1; length <= 3; ++length) {
    nn::IndexTensor tokens(ctx, {1, length});
    tokens.copy_from_host(prompt.data(), length);
    const std::vector<float> full = model.forward(tokens).to_host();
    const std::vector<float> want = position_row(full, 0, length, length - 1, config.vocab_size);

    nn::Transformer::GenerateState prefilled;
    const nn::Tensor& prefill_logits = model.prefill(tokens, prefilled, config.max_seq_len);
    CHECK(prefill_logits.shape() == std::vector<int>({1, 1, config.vocab_size}));
    expect_close(prefill_logits.to_host(), want, "prefill-last");
    CHECK(prefilled.t == length);
    CHECK(prefilled.layer_caches.size() == static_cast<size_t>(config.num_layers));
    for (const nn::MultiHeadAttention::Cache& cache : prefilled.layer_caches) CHECK(cache.t == length);

    nn::Transformer::GenerateState stepped;
    model.reset_generate_state(stepped, 1, config.max_seq_len);
    CHECK(stepped.layer_caches[0].k.shape() == std::vector<int>({1, 2, config.max_seq_len, 4}));
    const nn::Tensor* logits = nullptr;
    for (int t = 0; t < length; ++t) {
      nn::IndexTensor step(ctx, {1, 1});
      step.copy_from_host(&prompt[static_cast<size_t>(t)], 1);
      logits = &model.decode_step(step, stepped);
    }
    CHECK(logits->shape() == std::vector<int>({1, 1, config.vocab_size}));
    expect_close(logits->to_host(), want, "decode-last");
    CHECK(stepped.t == length);
    for (const nn::MultiHeadAttention::Cache& cache : stepped.layer_caches) CHECK(cache.t == length);

    const std::vector<float> again = model.forward(tokens).to_host();
    expect_close(position_row(again, 0, length, length - 1, config.vocab_size), want, "forward-stable");
  }

  {
    nn::IndexTensor batch_tokens(ctx, {2, 2});
    const std::vector<int64_t> ids = {1, 4, 2, 0};
    batch_tokens.copy_from_host(ids.data(), static_cast<int64_t>(ids.size()));
    const std::vector<float> full = model.forward(batch_tokens).to_host();
    nn::Transformer::GenerateState st;
    const std::vector<float> cached = model.prefill(batch_tokens, st, config.max_seq_len).to_host();
    CHECK(st.t == 2);
    for (int batch = 0; batch < 2; ++batch) {
      expect_close(position_row(cached, batch, 1, 0, config.vocab_size),
                   position_row(full, batch, 2, 1, config.vocab_size), "prefill-batch");
    }
  }

  nn::IndexTensor one(ctx, {1, 1});
  const int64_t id = prompt[0];
  one.copy_from_host(&id, 1);
  model.train();
  nn::Transformer::GenerateState st;
  model.reset_generate_state(st, 1, 2);
  expect_invalid([&] { model.decode_step(one, st); }, "decode_step");

  model.eval();
  model.reset_generate_state(st, 1, 2);
  model.decode_step(one, st);
  model.decode_step(one, st);
  CHECK(st.t == 2);
  expect_invalid([&] { model.decode_step(one, st); }, "cache");

  model.reset_generate_state(st, 1, config.max_seq_len);
  st.t = config.max_seq_len;
  expect_invalid([&] { model.decode_step(one, st); }, "max_seq_len");

  nn::IndexTensor wide(ctx, {1, 2});
  expect_invalid([&] { model.decode_step(wide, st); }, "next_ids");
  nn::IndexTensor too_long(ctx, {1, 3});
  const std::vector<int64_t> ids = {0, 1, 2};
  too_long.copy_from_host(ids.data(), static_cast<int64_t>(ids.size()));
  expect_invalid([&] { model.prefill(too_long, st, 2); }, "t_max");

  config.positional = nn::PositionalEncoding::Rope;
  nn::Transformer rope_model(ctx, config, 9);
  rope_model.eval();
  const std::vector<nn::NamedParam> rope_params = rope_model.parameters();
  CHECK(rope_params[0].name == "tok_emb.weight");
  CHECK(rope_params[1].name == "blocks.0.ln1.weight");
  for (const nn::NamedParam& param : rope_params) {
    CHECK(param.name.find("pos_emb") == std::string::npos);
    CHECK(param.name.find(".bias") == std::string::npos);
  }
  for (int length = 1; length <= 3; ++length) {
    nn::IndexTensor tokens(ctx, {1, length});
    tokens.copy_from_host(prompt.data(), length);
    const std::vector<float> full = rope_model.forward(tokens).to_host();
    const std::vector<float> want = position_row(full, 0, length, length - 1, config.vocab_size);
    nn::Transformer::GenerateState stepped;
    rope_model.reset_generate_state(stepped, 1, config.max_seq_len);
    const nn::Tensor* logits = nullptr;
    for (int t = 0; t < length; ++t) {
      nn::IndexTensor step(ctx, {1, 1});
      step.copy_from_host(&prompt[static_cast<size_t>(t)], 1);
      logits = &rope_model.decode_step(step, stepped);
    }
    expect_close(logits->to_host(), want, "rope-decode-last");
    nn::Transformer::GenerateState prefilled;
    expect_close(rope_model.prefill(tokens, prefilled, config.max_seq_len).to_host(), want,
                 "rope-prefill-last");
  }
  nn::Transformer::GenerateState rope_state;
  rope_model.reset_generate_state(rope_state, 1, config.max_seq_len);
  rope_state.t = config.max_seq_len;
  expect_invalid([&] { rope_model.decode_step(one, rope_state); }, "max_seq_len");
}

}  // namespace

int test_shapes() {
  nn::Context ctx;
  std::mt19937 rng(1);

  {
    nn::Linear linear(ctx, 2, 2, rng);
    std::vector<nn::NamedParam> params = linear.parameters();
    CHECK(params.size() == 2);
    params[0].value->copy_from_host(std::vector<float>{3.f, 4.f, 5.f, 6.f});
    params[1].value->copy_from_host(std::vector<float>{0.f, 0.f});
    nn::Tensor input(ctx, {1, 2});
    input.copy_from_host(std::vector<float>{1.f, 2.f});
    const nn::Tensor& output = linear.forward(input);
    const std::vector<float> got = output.to_host();
    CHECK(got.size() == 2);
    CHECK(got[0] == 11.f);
    CHECK(got[1] == 17.f);

    nn::Tensor grad_output(ctx, {1, 2});
    grad_output.copy_from_host(std::vector<float>{1.5f, -0.25f});
    const std::vector<float> before = grad_output.to_host();
    linear.backward(grad_output);
    const std::vector<float> after = grad_output.to_host();
    CHECK(before.size() == after.size());
    CHECK(std::memcmp(before.data(), after.data(), before.size() * sizeof(float)) == 0);
  }

  {
    nn::Linear linear(ctx, 2, 2, false, rng);
    std::vector<nn::NamedParam> params = linear.parameters();
    CHECK(params.size() == 1);
    CHECK(params[0].name == "weight");
    params[0].value->copy_from_host(std::vector<float>{3.f, 4.f, 5.f, 6.f});
    linear.zero_grad();
    nn::Tensor input(ctx, {1, 2});
    input.copy_from_host(std::vector<float>{1.f, 2.f});
    expect_close(linear.forward(input).to_host(), {11.f, 17.f}, "linear-nobias");
    nn::Tensor grad_output(ctx, {1, 2});
    grad_output.fill(1.f);
    const nn::Tensor& grad_input = linear.backward(grad_output);
    CHECK(grad_input.shape() == input.shape());
    expect_close(grad_input.to_host(), {8.f, 10.f}, "linear-nobias-dx");
    expect_close(params[0].grad->to_host(), {1.f, 2.f, 1.f, 2.f}, "linear-nobias-dw");
  }

  {
    nn::Linear linear(ctx, 2, 2, true, rng);
    std::vector<nn::NamedParam> params = linear.parameters();
    CHECK(params.size() == 2);
    CHECK(params[0].name == "weight");
    CHECK(params[1].name == "bias");
    params[0].value->copy_from_host(std::vector<float>{3.f, 4.f, 5.f, 6.f});
    params[1].value->copy_from_host(std::vector<float>{0.5f, -1.f});
    linear.zero_grad();
    nn::Tensor input(ctx, {1, 2, 2});
    input.copy_from_host(std::vector<float>{1.f, 2.f, 0.f, 1.f});
    const nn::Tensor& output = linear.forward(input);
    CHECK(output.shape() == std::vector<int>({1, 2, 2}));
    expect_close(output.to_host(), {11.5f, 16.f, 4.5f, 5.f}, "linear-rank3");
    nn::Tensor grad_output(ctx, {1, 2, 2});
    grad_output.fill(1.f);
    const nn::Tensor& grad_input = linear.backward(grad_output);
    CHECK(grad_input.shape() == input.shape());
    expect_close(grad_input.to_host(), {8.f, 10.f, 8.f, 10.f}, "linear-rank3-dx");
    expect_close(params[0].grad->to_host(), {1.f, 3.f, 1.f, 3.f}, "linear-rank3-dw");
    expect_close(params[1].grad->to_host(), {2.f, 2.f}, "linear-rank3-db");
    nn::Tensor bad_rank(ctx, {1, 2, 2, 2});
    expect_error([&] { linear.forward(bad_rank); }, "Linear input shape");
    nn::Tensor bad_grad(ctx, {1, 3, 2});
    expect_error([&] { linear.backward(bad_grad); }, "Linear backward shape");
  }

  {
    nn::Linear linear(ctx, 2, 2, false, rng);
    std::vector<nn::NamedParam> params = linear.parameters();
    CHECK(params.size() == 1);
    CHECK(params[0].name == "weight");
    params[0].value->copy_from_host(std::vector<float>{3.f, 4.f, 5.f, 6.f});
    linear.zero_grad();
    nn::Tensor input(ctx, {2, 2, 2});
    input.copy_from_host(std::vector<float>{1.f, 2.f, 0.f, 1.f, 1.f, 0.f, -1.f, 1.f});
    const nn::Tensor& output = linear.forward(input);
    CHECK(output.shape() == std::vector<int>({2, 2, 2}));
    expect_close(output.to_host(), {11.f, 17.f, 4.f, 6.f, 3.f, 5.f, 1.f, 1.f}, "linear-rank3-nobias");
    nn::Tensor grad_output(ctx, output.shape());
    grad_output.fill(1.f);
    CHECK(linear.backward(grad_output).shape() == input.shape());
    expect_close(params[0].grad->to_host(), {1.f, 4.f, 1.f, 4.f}, "linear-rank3-nobias-dw");
    nn::Tensor bad_features(ctx, {2, 2, 3});
    expect_error([&] { linear.forward(bad_features); }, "Linear input shape");
  }

  {
    nn::GELU layer(ctx);
    const std::vector<float> host = {0.f, 1.f, -1.f, 0.5f, -0.25f, 2.f};
    nn::Tensor input(ctx, {2, 3});
    input.copy_from_host(host);
    std::vector<float> want(host.size());
    for (size_t i = 0; i < host.size(); ++i) want[i] = gelu_ref(host[i]);
    expect_close(layer.forward(input).to_host(), want, "gelu");
    nn::Tensor grad(ctx, {2, 3});
    grad.fill(1.f);
    CHECK(layer.backward(grad).shape() == input.shape());
    nn::Tensor rank4(ctx, {1, 2, 2, 2});
    rank4.fill(0.25f);
    CHECK(layer.forward(rank4).shape() == rank4.shape());
    nn::Tensor rank3(ctx, {2, 1, 3});
    rank3.copy_from_host(host);
    const nn::Tensor& gelu3 = layer.forward(rank3);
    CHECK(gelu3.shape() == rank3.shape());
    expect_close(gelu3.to_host(), want, "gelu-rank3");
    nn::Tensor grad3(ctx, rank3.shape());
    grad3.fill(1.f);
    CHECK(layer.backward(grad3).shape() == rank3.shape());
    nn::Tensor bad_rank(ctx, {6});
    expect_error([&] { layer.forward(bad_rank); }, "GELU rank");
    nn::GELU fresh(ctx);
    nn::Tensor grad_rank(ctx, {2, 2});
    expect_error([&] { fresh.backward(grad_rank); }, "GELU backward shape");
  }

  {
    nn::SwiGLU layer(ctx);
    const std::vector<float> host = {1.f, -0.5f, 0.25f, -1.f, 0.5f, 2.f};
    nn::Tensor input(ctx, {1, 6});
    input.copy_from_host(host);
    const nn::Tensor& output = layer.forward(input);
    CHECK(output.shape() == std::vector<int>({1, 3}));
    expect_close(output.to_host(), swiglu_ref(host, 1, 3), "swiglu");
    nn::Tensor grad(ctx, {1, 3});
    grad.fill(1.f);
    CHECK(layer.backward(grad).shape() == input.shape());

    const std::vector<float> nchw = {0.5f, -0.25f, 1.5f, -1.f, 0.75f, 0.125f, -0.5f, 2.f};
    nn::Tensor image(ctx, {1, 4, 1, 2});
    image.copy_from_host(nchw);
    const nn::Tensor& pooled = layer.forward(image);
    CHECK(pooled.shape() == std::vector<int>({1, 2, 1, 2}));
    expect_close(pooled.to_host(), swiglu_ref(nchw, 1, 4), "swiglu-nchw");

    nn::Tensor odd(ctx, {2, 3});
    expect_error([&] { layer.forward(odd); }, "SwiGLU size");
    nn::Tensor bad_rank(ctx, {4});
    expect_error([&] { layer.forward(bad_rank); }, "SwiGLU rank");
    nn::Tensor bad_grad(ctx, {1, 6});
    expect_error([&] { layer.backward(bad_grad); }, "SwiGLU backward shape");

    // Dim 1 is odd, so a split on axis 1 would throw. Last dim 4 splits to 2.
    const std::vector<float> seq = {0.5f,  -0.25f, 1.f,  -1.f,  0.25f, 0.75f, -0.5f, 2.f,
                                    -1.f,  0.5f,   0.f,  1.5f,  1.25f, -0.75f, 0.5f,  -0.25f,
                                    0.125f, 1.f,   -2.f, 0.25f, -0.5f, 0.5f,  1.5f,  -1.f};
    nn::Tensor rank3(ctx, {2, 3, 4});
    rank3.copy_from_host(seq);
    const nn::Tensor& gated = layer.forward(rank3);
    CHECK(gated.shape() == std::vector<int>({2, 3, 2}));
    expect_close(gated.to_host(), swiglu_ref(seq, 6, 2), "swiglu-rank3");
    nn::Tensor grad3(ctx, {2, 3, 2});
    grad3.fill(1.f);
    CHECK(layer.backward(grad3).shape() == rank3.shape());
    nn::Tensor odd_last(ctx, {2, 2, 3});
    expect_error([&] { layer.forward(odd_last); }, "SwiGLU size");
  }

  {
    nn::ReLU layer(ctx);
    const std::vector<float> host = {-1.5f, 0.5f, 2.f, -0.25f, 4.f, -3.f};
    std::vector<float> want(host.size());
    for (size_t i = 0; i < host.size(); ++i) want[i] = host[i] > 0.f ? host[i] : 0.f;
    std::vector<float> want_grad(host.size());
    for (size_t i = 0; i < host.size(); ++i) want_grad[i] = host[i] > 0.f ? 1.f : 0.f;

    nn::Tensor rank2(ctx, {2, 3});
    rank2.copy_from_host(host);
    const nn::Tensor& y2 = layer.forward(rank2);
    CHECK(y2.shape() == rank2.shape());
    expect_close(y2.to_host(), want, "relu");
    nn::Tensor grad2(ctx, rank2.shape());
    grad2.fill(1.f);
    expect_close(layer.backward(grad2).to_host(), want_grad, "relu-grad");

    nn::Tensor rank3(ctx, {1, 2, 3});
    rank3.copy_from_host(host);
    const nn::Tensor& y3 = layer.forward(rank3);
    CHECK(y3.shape() == rank3.shape());
    expect_close(y3.to_host(), want, "relu-rank3");
    nn::Tensor grad3(ctx, rank3.shape());
    grad3.fill(1.f);
    CHECK(layer.backward(grad3).shape() == rank3.shape());
    expect_close(layer.backward(grad3).to_host(), want_grad, "relu-rank3-grad");

    nn::Tensor rank4(ctx, {1, 1, 2, 3});
    rank4.copy_from_host(host);
    CHECK(layer.forward(rank4).shape() == rank4.shape());
    nn::Tensor bad_rank(ctx, {6});
    expect_error([&] { layer.forward(bad_rank); }, "ReLU rank");
  }

  {
    expect_error([&] { nn::LayerNorm bad(ctx, 0); }, "LayerNorm size");
    expect_error([&] { nn::LayerNorm bad(ctx, -1); }, "LayerNorm size");
    expect_error([&] { nn::LayerNorm bad(ctx, 2, 0.f); }, "LayerNorm eps");
    expect_error([&] { nn::LayerNorm bad(ctx, 2, -1.f); }, "LayerNorm eps");
    expect_error([&] { nn::LayerNorm bad(ctx, 2, std::nanf("")); }, "LayerNorm eps");
    expect_error(
        [&] { nn::LayerNorm bad(ctx, 2, std::numeric_limits<float>::infinity()); }, "LayerNorm eps");

    constexpr float eps = 1e-3f;
    nn::LayerNorm layer(ctx, 3, eps);
    std::vector<nn::NamedParam> params = layer.parameters();
    CHECK(params.size() == 2);
    CHECK(params[0].name == "weight");
    CHECK(params[1].name == "bias");
    CHECK(params[0].value->shape() == std::vector<int>({3}));
    CHECK(params[1].value->shape() == std::vector<int>({3}));
    CHECK(layer.buffers().empty());
    expect_close(params[0].value->to_host(), {1.f, 1.f, 1.f}, "ln-gamma-init");
    expect_close(params[1].value->to_host(), {0.f, 0.f, 0.f}, "ln-beta-init");
    const std::vector<float> gamma = {1.f, 0.5f, -2.f};
    const std::vector<float> beta = {0.25f, -0.5f, 1.f};
    params[0].value->copy_from_host(gamma);
    params[1].value->copy_from_host(beta);

    const std::vector<float> host = {1.f, 2.f, 3.f, 4.f, 4.f, 4.f, -1.f, 0.f, 2.f};
    const std::vector<float> dy = {1.f, -1.f, 0.5f, 1.f, 1.f, 1.f, 0.f, 2.f, -1.f};
    nn::Tensor input(ctx, {3, 3});
    input.copy_from_host(host);
    const LayerNormRef ref = layer_norm_ref(host, gamma, beta, dy, 3, 3, eps);
    expect_close(layer.forward(input).to_host(), ref.y, "layernorm");
    layer.set_training(false);
    expect_close(layer.forward(input).to_host(), ref.y, "layernorm-eval");
    layer.set_training(true);
    expect_close(layer.forward(input).to_host(), ref.y, "layernorm-train");

    layer.zero_grad();
    nn::Tensor grad(ctx, input.shape());
    grad.copy_from_host(dy);
    expect_close(layer.backward(grad).to_host(), ref.dx, "layernorm-dx");
    expect_close(params[0].grad->to_host(), ref.dgamma, "layernorm-dw");
    expect_close(params[1].grad->to_host(), ref.dbeta, "layernorm-db");
    const std::vector<float> once_w = params[0].grad->to_host();
    const std::vector<float> once_b = params[1].grad->to_host();
    layer.backward(grad);
    std::vector<float> twice_w(once_w.size());
    std::vector<float> twice_b(once_b.size());
    for (size_t i = 0; i < once_w.size(); ++i) {
      twice_w[i] = once_w[i] + once_w[i];
      twice_b[i] = once_b[i] + once_b[i];
    }
    expect_close(params[0].grad->to_host(), twice_w, "layernorm-dw-accum");
    expect_close(params[1].grad->to_host(), twice_b, "layernorm-db-accum");

    const std::vector<float> seq = {1.f, 2.f, 3.f, 0.f, 1.f, -1.f, 4.f, 4.f, 4.f, -2.f, 0.5f, 1.5f};
    const std::vector<float> dy3 = {0.5f, -0.5f, 1.f, 1.f, 0.f, -1.f, 0.25f, 0.25f, 0.25f, 1.f, -2.f, 0.5f};
    nn::Tensor rank3(ctx, {2, 2, 3});
    rank3.copy_from_host(seq);
    const LayerNormRef ref3 = layer_norm_ref(seq, gamma, beta, dy3, 4, 3, eps);
    CHECK(layer.forward(rank3).shape() == rank3.shape());
    expect_close(layer.forward(rank3).to_host(), ref3.y, "layernorm-rank3");
    layer.zero_grad();
    nn::Tensor grad3(ctx, rank3.shape());
    grad3.copy_from_host(dy3);
    const nn::Tensor& dx3 = layer.backward(grad3);
    CHECK(dx3.shape() == rank3.shape());
    expect_close(dx3.to_host(), ref3.dx, "layernorm-rank3-dx");
    expect_close(params[0].grad->to_host(), ref3.dgamma, "layernorm-rank3-dw");
    expect_close(params[1].grad->to_host(), ref3.dbeta, "layernorm-rank3-db");

    nn::LayerNorm defaults(ctx, 3);
    nn::LayerNorm explicit_eps(ctx, 3, 1e-5f);
    nn::Tensor probe(ctx, {2, 3});
    probe.copy_from_host(std::vector<float>{1.f, 2.f, 4.f, -1.f, 0.5f, 0.25f});
    const std::vector<float> default_y = defaults.forward(probe).to_host();
    const std::vector<float> explicit_y = explicit_eps.forward(probe).to_host();
    expect_close(default_y, explicit_y, "layernorm-default-eps");

    nn::Tensor bad_rank(ctx, {3});
    expect_error([&] { layer.forward(bad_rank); }, "LayerNorm input shape");
    nn::Tensor bad_rank4(ctx, {1, 1, 1, 3});
    expect_error([&] { layer.forward(bad_rank4); }, "LayerNorm input shape");
    nn::Tensor bad_last(ctx, {2, 2, 4});
    expect_error([&] { layer.forward(bad_last); }, "LayerNorm input shape");
    nn::Tensor bad_grad(ctx, {2, 3});
    expect_error([&] { layer.backward(bad_grad); }, "LayerNorm backward shape");
    nn::LayerNorm fresh(ctx, 3);
    expect_error([&] { fresh.backward(grad); }, "LayerNorm backward shape");
  }

  {
    expect_error([&] { nn::RMSNorm bad(ctx, 0); }, "RMSNorm size");
    expect_error([&] { nn::RMSNorm bad(ctx, -1); }, "RMSNorm size");
    expect_error([&] { nn::RMSNorm bad(ctx, 2, 0.f); }, "RMSNorm eps");
    expect_error([&] { nn::RMSNorm bad(ctx, 2, -1.f); }, "RMSNorm eps");
    expect_error([&] { nn::RMSNorm bad(ctx, 2, std::nanf("")); }, "RMSNorm eps");

    constexpr float eps = 1e-5f;
    nn::RMSNorm layer(ctx, 4, eps);
    std::vector<nn::NamedParam> params = layer.parameters();
    CHECK(params.size() == 1);
    CHECK(params[0].name == "weight");
    CHECK(params[0].value->shape() == std::vector<int>({4}));
    const std::vector<float> weight = {0.5f, -1.f, 1.5f, 0.25f};
    params[0].value->copy_from_host(weight);
    const std::vector<float> host = {1.f, -2.f, 0.5f, 0.25f, 0.2f, 0.4f, -0.1f, 0.3f};
    auto rms_ref = [&](const std::vector<float>& x, int rows, int cols) {
      std::vector<float> y(x.size());
      for (int row = 0; row < rows; ++row) {
        double sq = 0.0;
        for (int col = 0; col < cols; ++col) {
          const double value = x[static_cast<size_t>(row * cols + col)];
          sq += value * value;
        }
        const double inv = 1.0 / std::sqrt(sq / static_cast<double>(cols) + static_cast<double>(eps));
        for (int col = 0; col < cols; ++col) {
          y[static_cast<size_t>(row * cols + col)] = static_cast<float>(
              x[static_cast<size_t>(row * cols + col)] * inv * static_cast<double>(weight[static_cast<size_t>(col)]));
        }
      }
      return y;
    };
    nn::Tensor rank2(ctx, {2, 4});
    rank2.copy_from_host(host);
    expect_close(layer.forward(rank2).to_host(), rms_ref(host, 2, 4), "rmsnorm");
    nn::Tensor rank3(ctx, {2, 1, 4});
    rank3.copy_from_host(host);
    expect_close(layer.forward(rank3).to_host(), rms_ref(host, 2, 4), "rmsnorm-rank3");
    nn::Tensor bad_rank(ctx, {4});
    expect_error([&] { layer.forward(bad_rank); }, "RMSNorm input shape");
    nn::Tensor bad_last(ctx, {2, 3});
    expect_error([&] { layer.forward(bad_last); }, "RMSNorm input shape");
    nn::Tensor bad_grad(ctx, {2, 3});
    expect_error([&] { layer.backward(bad_grad); }, "RMSNorm backward shape");
    nn::RMSNorm fresh(ctx, 4);
    nn::Tensor grad(ctx, {2, 4});
    grad.fill(1.f);
    expect_error([&] { fresh.backward(grad); }, "RMSNorm backward shape");
  }

  {
    expect_error([&] { nn::Dropout bad(ctx, 1.f); }, "Dropout p");
    expect_error([&] { nn::Dropout bad(ctx, -0.1f); }, "Dropout p");
    expect_error([&] { nn::Dropout bad(ctx, std::nanf("")); }, "Dropout p");

    nn::Dropout identity(ctx, 0.f);
    CHECK(identity.parameters().empty());
    CHECK(identity.buffers().empty());
    const std::vector<float> host = {1.f, -1.f, 0.5f, 2.f, -3.f, 4.f};
    nn::Tensor input(ctx, {2, 3});
    input.copy_from_host(host);
    expect_close(identity.forward(input).to_host(), host, "dropout-p0");
    nn::Tensor grad(ctx, input.shape());
    const std::vector<float> dy = {0.25f, 0.5f, -1.f, 1.5f, 2.f, -0.5f};
    grad.copy_from_host(dy);
    expect_close(identity.backward(grad).to_host(), dy, "dropout-p0-grad");

    nn::Tensor rank3(ctx, {2, 2, 3});
    rank3.copy_from_host(std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, -1.f, -2.f, 0.5f, 0.25f, 8.f, -4.f});
    const std::vector<float> rank3_host = rank3.to_host();
    expect_close(identity.forward(rank3).to_host(), rank3_host, "dropout-p0-rank3");
    nn::Tensor grad3(ctx, rank3.shape());
    grad3.fill(1.25f);
    expect_close(identity.backward(grad3).to_host(),
                 std::vector<float>(static_cast<size_t>(rank3.numel()), 1.25f), "dropout-p0-rank3-grad");

    nn::Dropout layer(ctx, 0.5f);
    layer.set_training(false);
    expect_close(layer.forward(input).to_host(), host, "dropout-eval");
    expect_close(layer.backward(grad).to_host(), dy, "dropout-eval-grad");
    nn::Tensor eval_rank3(ctx, rank3.shape());
    eval_rank3.copy_from_host(rank3_host);
    CHECK(layer.forward(eval_rank3).shape() == eval_rank3.shape());
    expect_close(layer.forward(eval_rank3).to_host(), rank3_host, "dropout-eval-rank3");

    layer.set_training(true);
    std::vector<float> varied(128);
    for (size_t i = 0; i < varied.size(); ++i) {
      varied[i] = 0.25f * static_cast<float>(static_cast<int>(i) - 64);
    }
    nn::Tensor train_input(ctx, {4, 32});
    train_input.copy_from_host(varied);
    const std::vector<float> dropped = layer.forward(train_input).to_host();
    nn::Tensor train_grad(ctx, train_input.shape());
    train_grad.fill(1.f);
    const std::vector<float> dx = layer.backward(train_grad).to_host();
    const std::vector<float> dx_again = layer.backward(train_grad).to_host();
    CHECK(dx == dx_again);
    constexpr float scale = 2.f;
    int kept = 0;
    int dropped_n = 0;
    CHECK(dropped.size() == varied.size());
    CHECK(dx.size() == varied.size());
    for (size_t i = 0; i < varied.size(); ++i) {
      const bool is_dropped = dx[i] == 0.f;
      const bool is_kept = dx[i] == scale;
      CHECK(is_dropped || is_kept);
      if (is_dropped) {
        CHECK(dropped[i] == 0.f);
        ++dropped_n;
      }
      if (is_kept) {
        CHECK(dropped[i] == varied[i] * scale);
        ++kept;
      }
    }
    CHECK(kept > 0);
    CHECK(dropped_n > 0);

    layer.set_training(false);
    expect_close(layer.forward(train_input).to_host(), varied, "dropout-eval-after-train");
    train_grad.fill(1.5f);
    expect_close(layer.backward(train_grad).to_host(),
                 std::vector<float>(varied.size(), 1.5f), "dropout-eval-after-train-grad");

    nn::Dropout fresh(ctx, 0.f);
    nn::Tensor missing(ctx, {1, 2});
    expect_error([&] { fresh.backward(missing); }, "Dropout backward shape");
    nn::Tensor tiny(ctx, {1, 2});
    tiny.fill(1.f);
    fresh.forward(tiny);
    nn::Tensor bad_grad(ctx, {2, 2});
    expect_error([&] { fresh.backward(bad_grad); }, "Dropout backward shape");
  }

  {
    nn::MlpConfig config;
    config.input_size = 784;
    config.hidden = {256, 128};
    config.output_size = 10;
    nn::Network net = nn::build_mlp(ctx, config, 1);
    expect_params(net, {{"fc.0.weight", {256, 784}},
                        {"fc.0.bias", {256}},
                        {"fc.1.weight", {128, 256}},
                        {"fc.1.bias", {128}},
                        {"fc.2.weight", {10, 128}},
                        {"fc.2.bias", {10}}});
    check_io(ctx, net, {3, 784}, {3, 10});
  }

  {
    nn::MlpConfig config;
    config.input_size = 16;
    config.hidden = {};
    config.output_size = 4;
    nn::Network net = nn::build_mlp(ctx, config, 1);
    expect_params(net, {{"fc.0.weight", {4, 16}}, {"fc.0.bias", {4}}});
    check_io(ctx, net, {2, 16}, {2, 4});
  }

  {
    nn::MlpConfig config;
    config.input_size = 8;
    config.hidden = {7, 6, 5};
    config.output_size = 4;
    nn::Network net = nn::build_mlp(ctx, config, 1);
    expect_params(net, {{"fc.0.weight", {7, 8}},
                        {"fc.0.bias", {7}},
                        {"fc.1.weight", {6, 7}},
                        {"fc.1.bias", {6}},
                        {"fc.2.weight", {5, 6}},
                        {"fc.2.bias", {5}},
                        {"fc.3.weight", {4, 5}},
                        {"fc.3.bias", {4}}});
    check_io(ctx, net, {1, 8}, {1, 4});
  }

  {
    nn::MlpConfig config;
    config.input_size = 8;
    config.hidden = {7, 6};
    config.output_size = 4;
    config.activation = nn::Activation::Gelu;
    nn::Network net = nn::build_mlp(ctx, config, 1);
    expect_params(net, {{"fc.0.weight", {7, 8}},
                        {"fc.0.bias", {7}},
                        {"fc.1.weight", {6, 7}},
                        {"fc.1.bias", {6}},
                        {"fc.2.weight", {4, 6}},
                        {"fc.2.bias", {4}}});
    check_io(ctx, net, {2, 8}, {2, 4});
  }

  {
    nn::MlpConfig config;
    config.input_size = 8;
    config.hidden = {6, 5};
    config.output_size = 4;
    config.activation = nn::Activation::SwiGLU;
    nn::Network net = nn::build_mlp(ctx, config, 1);
    expect_params(net, {{"fc.0.weight", {12, 8}},
                        {"fc.0.bias", {12}},
                        {"fc.1.weight", {10, 6}},
                        {"fc.1.bias", {10}},
                        {"fc.2.weight", {4, 5}},
                        {"fc.2.bias", {4}}});
    check_io(ctx, net, {2, 8}, {2, 4});
  }

  {
    nn::MlpConfig config;
    config.input_size = 16;
    config.hidden = {};
    config.output_size = 4;
    config.activation = nn::Activation::SwiGLU;
    nn::Network net = nn::build_mlp(ctx, config, 1);
    expect_params(net, {{"fc.0.weight", {4, 16}}, {"fc.0.bias", {4}}});
    check_io(ctx, net, {2, 16}, {2, 4});
  }

  {
    nn::MlpConfig config;
    config.input_size = 4;
    config.hidden = {4};
    config.output_size = 2;
    config.activation = static_cast<nn::Activation>(0);
    expect_invalid([&] { nn::validate(config); }, "activation");
    config.activation = nn::Activation::SwiGLU;
    config.hidden = {1 << 30};
    expect_invalid([&] { nn::validate(config); }, "hidden");
  }

  {
    nn::LastStep layer;
    const std::vector<float> host = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f};
    nn::Tensor input(ctx, {2, 3, 2});
    input.copy_from_host(host);
    const nn::Tensor& output = layer.forward(input);
    CHECK(output.shape() == std::vector<int>({2, 2}));
    expect_close(output.to_host(), {4.f, 5.f, 10.f, 11.f}, "last-step");
    nn::Tensor grad(ctx, {2, 2});
    grad.copy_from_host(std::vector<float>{1.f, 2.f, 3.f, 4.f});
    const std::vector<float> got = layer.backward(grad).to_host();
    expect_close(got, {0.f, 0.f, 0.f, 0.f, 1.f, 2.f, 0.f, 0.f, 0.f, 0.f, 3.f, 4.f}, "last-step-grad");
    nn::Tensor bad_rank(ctx, {2, 3});
    expect_error([&] { layer.forward(bad_rank); }, "LastStep input rank");
    nn::LastStep fresh;
    expect_error([&] { fresh.backward(grad); }, "LastStep backward without forward");
  }

  {
    nn::RNN layer(ctx, 1, 1, rng);
    std::vector<nn::NamedParam> params = layer.parameters();
    CHECK(params.size() == 4);
    params[0].value->copy_from_host(std::vector<float>{1.f});
    params[1].value->copy_from_host(std::vector<float>{1.f});
    params[2].value->fill(0.f);
    params[3].value->fill(0.f);
    nn::Tensor input(ctx, {1, 2, 1});
    input.copy_from_host(std::vector<float>{0.5f, 0.25f});
    const float h0 = std::tanh(0.5f);
    const float h1 = std::tanh(0.25f + h0);
    const nn::Tensor& output = layer.forward(input);
    CHECK(output.shape() == std::vector<int>({1, 2, 1}));
    expect_close(output.to_host(), {h0, h1}, "rnn");
    nn::Tensor grad(ctx, {1, 2, 1});
    grad.fill(1.f);
    CHECK(layer.backward(grad).shape() == input.shape());
    nn::Tensor bad_rank(ctx, {1, 1});
    expect_error([&] { layer.forward(bad_rank); }, "RNN input shape");
    expect_error([&] { nn::RNN bad(ctx, 0, 1, rng); }, "RNN size");
  }

  {
    nn::RnnConfig config;
    config.input_size = 3;
    config.hidden = {4, 6};
    config.output_size = 2;
    nn::Network net = nn::build_rnn(ctx, config, 1);
    expect_params(net, {{"rnn.0.weight_ih", {4, 3}},
                        {"rnn.0.weight_hh", {4, 4}},
                        {"rnn.0.bias_ih", {4}},
                        {"rnn.0.bias_hh", {4}},
                        {"rnn.1.weight_ih", {6, 4}},
                        {"rnn.1.weight_hh", {6, 6}},
                        {"rnn.1.bias_ih", {6}},
                        {"rnn.1.bias_hh", {6}},
                        {"head.weight", {2, 6}},
                        {"head.bias", {2}}});
    check_io(ctx, net, {2, 5, 3}, {2, 2});
    CHECK(net.family() == nn::Family::Rnn);
  }

  {
    nn::RnnConfig config;
    config.input_size = 4;
    config.output_size = 2;
    expect_invalid([&] { nn::validate(config); }, "hidden");
  }

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
    config.tie_embeddings = true;
    expect_invalid([&] { nn::validate(config); }, "tie_embeddings");
    expect_invalid([&] { nn::build_transformer_network(ctx, config, 1); }, "tie_embeddings");

    config.tie_embeddings = false;
    config.positional = nn::PositionalEncoding::Rope;
    nn::validate(config);
    config.d_model = 6;
    expect_invalid([&] { nn::validate(config); }, "use_rope");
    config.d_model = 8;
    config.positional = static_cast<nn::PositionalEncoding>(0);
    expect_invalid([&] { nn::validate(config); }, "positional");
    config.positional = nn::PositionalEncoding::LearnedAbsolute;
    config.dropout = 1.f;
    expect_invalid([&] { nn::validate(config); }, "dropout");
    config.dropout = std::nanf("");
    expect_invalid([&] { nn::validate(config); }, "dropout");
    config.dropout = 0.f;
    config.num_heads = 3;
    expect_invalid([&] { nn::validate(config); }, "num_heads");
    config.num_heads = 2;
    config.activation = static_cast<nn::TransformerActivation>(0);
    expect_invalid([&] { nn::validate(config); }, "activation");
    config.activation = nn::TransformerActivation::SwiGLU;
    config.d_ff = std::numeric_limits<int>::max();
    expect_invalid([&] { nn::validate(config); }, "d_ff");
    config.activation = nn::TransformerActivation::Gelu;
    nn::validate(config);
    config.activation = nn::TransformerActivation::SwiGLU;
    config.d_ff = 16;

    nn::Network net = nn::build_transformer_network(ctx, config, 1);
    CHECK(net.family() == nn::Family::Transformer);
    CHECK(has_param(net, "blocks.0.ln1.weight"));
    CHECK(has_param(net, "final_ln.weight"));
    CHECK(has_param(net, "final_ln.bias"));
    CHECK(has_param(net, "lm_head.weight"));
    CHECK(!has_param(net, "lm_head.bias"));
    CHECK(find_param(net, "blocks.0.fc1.weight")->shape() == std::vector<int>({32, 8}));
    CHECK(find_param(net, "lm_head.weight")->shape() == std::vector<int>({10, 8}));
    for (const nn::NamedParam& param : net.parameters()) {
      CHECK(param.name.find("tok_emb") == std::string::npos);
    }
    check_io(ctx, net, {1, 3, 8}, {1, 3, 10});
    check_io(ctx, net, {1, 5, 8}, {1, 5, 10});

    config.positional = nn::PositionalEncoding::Rope;
    nn::Network rope_net = nn::build_transformer_network(ctx, config, 1);
    CHECK(has_param(rope_net, "blocks.0.ln1.weight"));
    CHECK(has_param(rope_net, "final_ln.weight"));
    CHECK(!has_param(rope_net, "final_ln.bias"));
    CHECK(!has_param(rope_net, "blocks.0.ln1.bias"));
    CHECK(!has_param(rope_net, "blocks.0.ln2.bias"));
    check_io(ctx, rope_net, {1, 3, 8}, {1, 3, 10});
    config.positional = nn::PositionalEncoding::LearnedAbsolute;
  }

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
    config.tie_embeddings = true;
    expect_invalid([&] { nn::Transformer(ctx, config, 1); }, "tie_embeddings");
    config.tie_embeddings = false;

    nn::Transformer model(ctx, config, 1);
    const std::vector<nn::NamedParam> params = model.parameters();
    CHECK(params.size() >= 3);
    CHECK(params[0].name == "tok_emb.weight");
    CHECK(params[0].value->shape() == std::vector<int>({10, 8}));
    CHECK(params[1].name == "pos_emb.weight");
    CHECK(params[1].value->shape() == std::vector<int>({4, 8}));
    CHECK(params[2].name == "blocks.0.ln1.weight");
    bool saw_final = false;
    bool saw_head = false;
    for (const nn::NamedParam& param : params) {
      if (param.name == "final_ln.weight") saw_final = true;
      if (param.name == "lm_head.weight") saw_head = true;
    }
    CHECK(saw_final);
    CHECK(saw_head);
    for (const nn::NamedParam& param : model.network().parameters()) {
      CHECK(param.name.find("tok_emb") == std::string::npos);
    }

    nn::IndexTensor tokens(ctx, {1, 3});
    const std::vector<int64_t> ids = {1, 0, 2};
    tokens.copy_from_host(ids.data(), static_cast<int64_t>(ids.size()));
    CHECK(model.forward(tokens).shape() == std::vector<int>({1, 3, 10}));

    nn::IndexTensor long_tokens(ctx, {1, 5});
    const std::vector<int64_t> long_ids(5, 0);
    long_tokens.copy_from_host(long_ids.data(), static_cast<int64_t>(long_ids.size()));
    expect_invalid([&] { model.forward(long_tokens); }, "max_seq_len");

    model.zero_grad();
    nn::Tensor hidden(ctx, {1, 3, 8});
    hidden.fill(0.1f);
    const nn::Tensor& logits = model.forward(hidden);
    CHECK(logits.shape() == std::vector<int>({1, 3, 10}));
    nn::Tensor grad(ctx, logits.shape());
    grad.fill(1.f);
    model.backward(grad);
    const std::vector<float> emb_grad = model.parameters()[0].grad->to_host();
    for (float value : emb_grad) CHECK(value == 0.f);
  }

  CHECK(nn::conv_out_size(28, 5, 1, 2) == 28);
  CHECK(nn::conv_out_size(28, 2, 2, 0) == 14);
  CHECK(nn::conv_out_size(14, 5, 1, 2) == 14);
  CHECK(nn::conv_out_size(14, 2, 2, 0) == 7);
  CHECK(nn::conv_out_size(7, 2, 2, 0) == 3);
  CHECK(nn::conv_out_size(28, 3, 1, 1) == 28);
  CHECK(nn::conv_out_size(28, 3, 2, 1) == 14);
  CHECK(nn::conv_out_size(14, 3, 2, 1) == 7);

  {
    nn::ConvConfig config;
    config.in_channels = 1;
    config.in_height = 28;
    config.in_width = 28;
    config.convs = {nn::ConvSpec{32, 5, 1, 2}, nn::ConvSpec{64, 5, 1, 2}};
    config.pool = nn::PoolSpec{2, 2};
    config.fc = {128, 10};
    nn::Network net = nn::build_conv(ctx, config, 1);
    expect_params(net, {{"conv.0.weight", {32, 1, 5, 5}},
                        {"conv.0.bias", {32}},
                        {"conv.1.weight", {64, 32, 5, 5}},
                        {"conv.1.bias", {64}},
                        {"fc.0.weight", {128, 3136}},
                        {"fc.0.bias", {128}},
                        {"fc.1.weight", {10, 128}},
                        {"fc.1.bias", {10}}});
    check_io(ctx, net, {2, 1, 28, 28}, {2, 10});
  }

  {
    nn::ConvConfig config;
    config.in_channels = 1;
    config.in_height = 28;
    config.in_width = 28;
    config.convs = {nn::ConvSpec{8, 3, 1, 1}, nn::ConvSpec{8, 3, 1, 1}, nn::ConvSpec{16, 3, 1, 1}};
    config.pool = nn::PoolSpec{2, 2};
    config.fc = {10};
    nn::Network net = nn::build_conv(ctx, config, 1);
    nn::Tensor* weight = find_param(net, "fc.0.weight");
    CHECK(weight != nullptr);
    CHECK(weight->shape() == std::vector<int>({10, 144}));
    nn::Tensor* conv2 = find_param(net, "conv.2.weight");
    CHECK(conv2 != nullptr);
    CHECK(conv2->shape() == std::vector<int>({16, 8, 3, 3}));
    check_io(ctx, net, {2, 1, 28, 28}, {2, 10});
  }

  {
    nn::ResNetConfig config;
    config.in_channels = 1;
    config.in_height = 28;
    config.in_width = 28;
    config.stem_channels = 16;
    config.blocks_per_stage = {2, 2, 2};
    config.stage_widths = {16, 32, 64};
    config.output_size = 10;
    nn::Network net = nn::build_resnet(ctx, config, 1);
    CHECK(find_param(net, "stem.weight")->shape() == std::vector<int>({16, 1, 3, 3}));
    CHECK(find_param(net, "head.weight")->shape() == std::vector<int>({10, 64}));
    CHECK(find_param(net, "head.bias")->shape() == std::vector<int>({10}));
    CHECK(!has_param(net, "stages.0.blocks.0.proj.weight"));
    CHECK(!has_param(net, "stages.0.blocks.1.proj.weight"));
    CHECK(find_param(net, "stages.1.blocks.0.proj.weight")->shape() == std::vector<int>({32, 16, 1, 1}));
    CHECK(!has_param(net, "stages.1.blocks.1.proj.weight"));
    CHECK(find_param(net, "stages.2.blocks.0.proj.weight")->shape() == std::vector<int>({64, 32, 1, 1}));
    CHECK(!has_param(net, "stages.2.blocks.1.proj.weight"));
    CHECK(find_param(net, "stages.0.blocks.0.conv1.weight")->shape() == std::vector<int>({16, 16, 3, 3}));
    CHECK(find_param(net, "stages.1.blocks.0.conv1.weight")->shape() == std::vector<int>({32, 16, 3, 3}));
    check_io(ctx, net, {2, 1, 28, 28}, {2, 10});
  }

  {
    nn::ResNetConfig config;
    config.in_channels = 1;
    config.in_height = 28;
    config.in_width = 28;
    config.stem_channels = 8;
    config.blocks_per_stage = {1, 1};
    config.stage_widths = {8, 16};
    config.output_size = 5;
    nn::Network net = nn::build_resnet(ctx, config, 1);
    CHECK(find_param(net, "stem.weight")->shape() == std::vector<int>({8, 1, 3, 3}));
    CHECK(find_param(net, "head.weight")->shape() == std::vector<int>({5, 16}));
    CHECK(!has_param(net, "stages.0.blocks.0.proj.weight"));
    CHECK(find_param(net, "stages.1.blocks.0.proj.weight")->shape() == std::vector<int>({16, 8, 1, 1}));
    check_io(ctx, net, {2, 1, 28, 28}, {2, 5});

    std::mt19937 stem_rng(1);
    nn::Conv2d stem(ctx, 1, 8, 3, 1, 1, false, stem_rng);
    nn::Tensor image(ctx, {2, 1, 28, 28});
    image.fill(0.1f);
    CHECK(stem.forward(image).shape() == std::vector<int>({2, 8, 28, 28}));
    nn::ResidualBlock stage0(ctx, 8, 8, 1, stem_rng);
    nn::Tensor map(ctx, {2, 8, 28, 28});
    map.fill(0.1f);
    CHECK(stage0.forward(map).shape() == std::vector<int>({2, 8, 28, 28}));
    nn::ResidualBlock stage1(ctx, 8, 16, 2, stem_rng);
    CHECK(stage1.forward(map).shape() == std::vector<int>({2, 16, 14, 14}));
  }

  check_block(ctx, 4, 4, 1, {2, 4, 8, 8}, {2, 4, 8, 8}, false);
  check_block(ctx, 4, 8, 1, {2, 4, 8, 8}, {2, 8, 8, 8}, true);
  check_block(ctx, 4, 8, 2, {2, 4, 8, 8}, {2, 8, 4, 4}, true);

  {
    nn::ConvResNetConfig config;
    config.in_channels = 1;
    config.in_height = 8;
    config.in_width = 8;
    config.convs = {nn::ConvSpec{4, 3, 1, 1}};
    config.pool = nn::PoolSpec{2, 2};
    config.stem_channels = 4;
    config.blocks_per_stage = {1, 1};
    config.stage_widths = {4, 8};
    config.fc = {4};
    nn::ConvResNet model(ctx, config, 1);
    CHECK(model.family() == nn::Family::ConvResNet);
    CHECK(model.training());
    CHECK(model.config().stem_channels == 4);
    CHECK(model.parameters().size() == model.network().parameters().size());
    expect_params(model.network(), {{"conv.0.weight", {4, 1, 3, 3}},
                                    {"conv.0.bias", {4}},
                                    {"stem.weight", {4, 4, 3, 3}},
                                    {"stem.bn.weight", {4}},
                                    {"stem.bn.bias", {4}},
                                    {"stages.0.blocks.0.conv1.weight", {4, 4, 3, 3}},
                                    {"stages.0.blocks.0.bn1.weight", {4}},
                                    {"stages.0.blocks.0.bn1.bias", {4}},
                                    {"stages.0.blocks.0.conv2.weight", {4, 4, 3, 3}},
                                    {"stages.0.blocks.0.bn2.weight", {4}},
                                    {"stages.0.blocks.0.bn2.bias", {4}},
                                    {"stages.1.blocks.0.conv1.weight", {8, 4, 3, 3}},
                                    {"stages.1.blocks.0.bn1.weight", {8}},
                                    {"stages.1.blocks.0.bn1.bias", {8}},
                                    {"stages.1.blocks.0.conv2.weight", {8, 8, 3, 3}},
                                    {"stages.1.blocks.0.bn2.weight", {8}},
                                    {"stages.1.blocks.0.bn2.bias", {8}},
                                    {"stages.1.blocks.0.proj.weight", {8, 4, 1, 1}},
                                    {"stages.1.blocks.0.proj_bn.weight", {8}},
                                    {"stages.1.blocks.0.proj_bn.bias", {8}},
                                    {"fc.0.weight", {4, 8}},
                                    {"fc.0.bias", {4}}});
    CHECK(!has_param(model.network(), "stages.0.blocks.0.proj.weight"));
    check_io(ctx, model.network(), {2, 1, 8, 8}, {2, 4});
  }

  {
    nn::ConvResNetConfig config;
    config.in_channels = 1;
    config.in_height = 8;
    config.in_width = 8;
    config.convs = {nn::ConvSpec{4, 3, 1, 1}};
    config.pool = nn::PoolSpec{2, 2};
    config.fc = {4};
    nn::ConvResNet model(ctx, config, 1);
    expect_params(model.network(), {{"conv.0.weight", {4, 1, 3, 3}},
                                    {"conv.0.bias", {4}},
                                    {"fc.0.weight", {4, 64}},
                                    {"fc.0.bias", {4}}});
    CHECK(!has_param(model.network(), "stem.weight"));
    check_io(ctx, model.network(), {2, 1, 8, 8}, {2, 4});
  }

  {
    nn::ConvResNetConfig config;
    config.in_channels = 1;
    config.in_height = 8;
    config.in_width = 8;
    config.stem_channels = 4;
    config.blocks_per_stage = {1, 1};
    config.stage_widths = {4, 8};
    config.fc = {4};
    nn::ConvResNet model(ctx, config, 1);
    CHECK(find_param(model.network(), "stem.weight")->shape() == std::vector<int>({4, 1, 3, 3}));
    CHECK(!has_param(model.network(), "conv.0.weight"));
    CHECK(!has_param(model.network(), "stages.0.blocks.0.proj.weight"));
    CHECK(find_param(model.network(), "stages.1.blocks.0.proj.weight")->shape() ==
          std::vector<int>({8, 4, 1, 1}));
    CHECK(find_param(model.network(), "fc.0.weight")->shape() == std::vector<int>({4, 8}));
    check_io(ctx, model.network(), {2, 1, 8, 8}, {2, 4});
  }

  {
    nn::MlpConfig mlp;
    mlp.input_size = 4;
    mlp.output_size = 2;
    bool threw = false;
    try {
      nn::ConvResNet model(nn::build_mlp(ctx, mlp, 1));
      (void)model;
    } catch (const nn::Error& error) {
      threw = true;
      CHECK(std::string(error.what()) == "ConvResNet family");
    }
    CHECK(threw);
  }

  check_multi_head_attention(ctx, rng);
  check_transformer_block(ctx, rng);
  check_cached_decode(ctx, rng);

  {
    expect_error([&] { nn::IndexTensor(ctx, {}); }, "index tensor rank out of range");
    expect_error([&] { nn::IndexTensor(ctx, {1, 1, 1}); }, "index tensor rank out of range");
    expect_error([&] { nn::IndexTensor(ctx, {0}); }, "index tensor dimension out of range");
    expect_error([&] { nn::IndexTensor(ctx, {2, 0}); }, "index tensor dimension out of range");
    expect_error([&] { nn::IndexTensor(ctx, {std::numeric_limits<int>::max(), 2}); },
                 "index tensor numel out of range");

    nn::IndexTensor ids(ctx, {2, 2});
    CHECK(ids.data() != nullptr);
    CHECK(ids.numel() == 4);
    CHECK(ids.shape() == std::vector<int>({2, 2}));
    CHECK(&ids.context() == &ctx);
    const int64_t one = 0;
    expect_error([&] { ids.copy_from_host(&one, 1); }, "index tensor host copy numel mismatch");

    nn::Tensor sample(ctx, {4});
    expect_error([&] { nn::normal_(sample, 0.f, 0.f, rng); }, "normal_ std");
    expect_error([&] { nn::normal_(sample, 0.f, -0.1f, rng); }, "normal_ std");
    expect_error([&] { nn::normal_(sample, std::numeric_limits<float>::infinity(), 0.02f, rng); },
                 "normal_ mean");
    std::mt19937 draw_rng(5);
    std::mt19937 ref_rng(5);
    nn::normal_(sample, 0.25f, 0.02f, draw_rng);
    std::normal_distribution<float> dist(0.25f, 0.02f);
    std::vector<float> drawn(4);
    for (float& value : drawn) value = dist(ref_rng);
    expect_close(sample.to_host(), drawn, "normal_");

    expect_error([&] { nn::Embedding(ctx, 0, 2, rng); }, "Embedding size");
    expect_error([&] { nn::Embedding(ctx, 2, -1, rng); }, "Embedding size");

    nn::Embedding layer(ctx, 3, 2, rng);
    nn::Tensor float_input(ctx, {2, 2});
    expect_error([&] { layer.forward(float_input); }, "Embedding requires indices");

    std::vector<nn::NamedParam> params = layer.parameters();
    CHECK(params.size() == 1);
    CHECK(params[0].name == "weight");
    CHECK(params[0].value->shape() == std::vector<int>({3, 2}));
    CHECK(params[0].grad->shape() == params[0].value->shape());
    params[0].value->copy_from_host(std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});

    const std::vector<int64_t> host_ids = {2, 0, 1, 0};
    ids.copy_from_host(host_ids.data(), static_cast<int64_t>(host_ids.size()));
    const nn::Tensor& output = layer.forward_indices(ids);
    CHECK(output.shape() == std::vector<int>({2, 2, 2}));
    expect_close(output.to_host(), {5.f, 6.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f}, "embedding");

    nn::IndexTensor bad(ctx, {2});
    const std::vector<int64_t> oob = {0, 3};
    bad.copy_from_host(oob.data(), static_cast<int64_t>(oob.size()));
    expect_error([&] { layer.forward_indices(bad); }, "index out of range");
    const std::vector<int64_t> neg = {-1, 1};
    bad.copy_from_host(neg.data(), static_cast<int64_t>(neg.size()));
    expect_error([&] { layer.forward_indices(bad); }, "index out of range");

    // A rejected index must leave the ids captured by the last successful forward.
    layer.zero_grad();
    expect_close(params[0].grad->to_host(), std::vector<float>(6, 0.f), "embedding zero");
    nn::Tensor grad(ctx, {2, 2, 2});
    grad.copy_from_host(std::vector<float>{1.f, 10.f, 2.f, 20.f, 3.f, 30.f, 4.f, 40.f});
    const std::vector<float> before = grad.to_host();
    const nn::Tensor& back = layer.backward(grad);
    CHECK(back.shape() == grad.shape());
    expect_close(back.to_host(), std::vector<float>(static_cast<size_t>(grad.numel()), 0.f),
                 "embedding backward return");
    CHECK(std::memcmp(before.data(), grad.to_host().data(), before.size() * sizeof(float)) == 0);
    expect_close(params[0].grad->to_host(), {6.f, 60.f, 3.f, 30.f, 1.f, 10.f}, "embedding grad");
    layer.backward(grad);
    expect_close(params[0].grad->to_host(), {12.f, 120.f, 6.f, 60.f, 2.f, 20.f},
                 "embedding grad accum");

    nn::Tensor bad_grad(ctx, {2, 2});
    expect_error([&] { layer.backward(bad_grad); }, "Embedding backward shape");
    nn::Embedding fresh(ctx, 2, 2, rng);
    expect_error([&] { fresh.backward(bad_grad); }, "Embedding backward without forward");

    nn::IndexTensor rows(ctx, {3});
    const std::vector<int64_t> row_ids = {1, 2, 0};
    rows.copy_from_host(row_ids.data(), static_cast<int64_t>(row_ids.size()));
    const nn::Tensor& row_out = layer.forward_indices(rows);
    CHECK(row_out.shape() == std::vector<int>({3, 2}));
    expect_close(row_out.to_host(), {3.f, 4.f, 5.f, 6.f, 1.f, 2.f}, "embedding rows");

    nn::Context other;
    nn::IndexTensor foreign(other, {1});
    const int64_t zero = 0;
    foreign.copy_from_host(&zero, 1);
    expect_error([&] { layer.forward_indices(foreign); }, "Embedding context");
  }

  {
    const std::vector<float> rank3_logits = {0.2f, -0.4f, 0.7f, 1.1f, 0.3f, -0.2f,
                                             -0.5f, 0.4f, 0.1f, 0.8f, -0.6f, 0.2f};
    // Default ignore_index is -100. Every position dropped: loss 0, grad 0.
    check_sequence_case(ctx, {2, 2, 3}, rank3_logits, {-100, -100, -100, -100}, -100);
    nn::Tensor default_logits(ctx, {2, 2, 3});
    default_logits.copy_from_host(rank3_logits);
    nn::IndexTensor default_labels(ctx, {2, 2});
    const std::vector<int64_t> all_ignored = {-100, -100, -100, -100};
    default_labels.copy_from_host(all_ignored.data(), static_cast<int64_t>(all_ignored.size()));
    nn::LossResult defaults = nn::sequence_cross_entropy(ctx, default_logits, default_labels);
    CHECK(defaults.loss == 0.f);
    for (float value : defaults.grad_logits.to_host()) CHECK(value == 0.f);

    // ignore_index outside [0, V) is still an ignore, not an out-of-range error.
    check_sequence_case(ctx, {2, 4}, {0.1f, 0.2f, 0.3f, 0.4f, -0.2f, 0.5f, 0.0f, 0.7f}, {7, 7}, 7);

    // Mixed ignore. Kept rows match softmax_cross_entropy over only those rows.
    check_sequence_case(ctx, {2, 2, 3}, rank3_logits, {-100, 1, 0, -100}, -100);
    check_sequence_case(ctx, {3, 4},
                        {0.2f, -1.f, 0.5f, 0.1f, 0.3f, 0.4f, -0.2f, 0.6f, -0.4f, 0.7f, 0.2f, 0.0f},
                        {1, 0, 1}, 1);

    check_sequence_case(ctx, {2, 3, 5},
                        {0.1f, 0.2f, -0.3f, 0.4f, 0.0f, -0.2f, 0.5f, 0.1f, -0.4f, 0.3f,
                         0.6f, -0.1f, 0.2f, 0.0f, -0.5f, 0.4f, 0.3f, -0.2f, 0.1f, 0.5f,
                         -0.6f, 0.2f, 0.3f, -0.1f, 0.4f, 0.1f, -0.3f, 0.2f, 0.6f, -0.4f},
                        {0, 2, 4, 1, 3, 0}, -100);
    check_sequence_case(ctx, {6, 5},
                        {0.1f, 0.2f, -0.3f, 0.4f, 0.0f, -0.2f, 0.5f, 0.1f, -0.4f, 0.3f,
                         0.6f, -0.1f, 0.2f, 0.0f, -0.5f, 0.4f, 0.3f, -0.2f, 0.1f, 0.5f,
                         -0.6f, 0.2f, 0.3f, -0.1f, 0.4f, 0.1f, -0.3f, 0.2f, 0.6f, -0.4f},
                        {0, 2, 4, 1, 3, 0}, -100);
    check_sequence_case(ctx, {2, 1, 4}, {0.2f, -0.1f, 0.4f, 0.0f, -0.3f, 0.5f, 0.1f, -0.2f}, {1, 3},
                        -100);

    // Wider than the 128-thread reduction, with ignored rows that must be entirely zero.
    constexpr int wide_classes = 160;
    constexpr int wide_rows = 4;
    std::vector<float> wide_logits(static_cast<size_t>(wide_rows * wide_classes));
    for (int i = 0; i < wide_rows * wide_classes; ++i) {
      wide_logits[static_cast<size_t>(i)] = 0.015f * static_cast<float>((i % 23) - 11);
    }
    check_sequence_case(ctx, {2, 2, wide_classes}, wide_logits, {-100, 7, 159, -100}, -100);
    check_sequence_case(ctx, {2, 2, wide_classes}, wide_logits, {3, 8, 1, 90}, -100);

    nn::Tensor bad_logits(ctx, {1, 3, 4});
    bad_logits.fill(0.1f);
    nn::IndexTensor bad_labels(ctx, {1, 3});
    const std::vector<int64_t> too_high = {0, 1, 4};
    bad_labels.copy_from_host(too_high.data(), static_cast<int64_t>(too_high.size()));
    expect_error([&] { nn::sequence_cross_entropy(ctx, bad_logits, bad_labels); }, "label out of range");
    const std::vector<int64_t> negative = {0, -1, 1};
    bad_labels.copy_from_host(negative.data(), static_cast<int64_t>(negative.size()));
    expect_error([&] { nn::sequence_cross_entropy(ctx, bad_logits, bad_labels); }, "label out of range");
    const std::vector<int64_t> bad_among_ignored = {1, -100, 4};
    bad_labels.copy_from_host(bad_among_ignored.data(), static_cast<int64_t>(bad_among_ignored.size()));
    expect_error([&] { nn::sequence_cross_entropy(ctx, bad_logits, bad_labels); }, "label out of range");
    const std::vector<int64_t> not_this_ignore = {-100, 1, 0};
    bad_labels.copy_from_host(not_this_ignore.data(), static_cast<int64_t>(not_this_ignore.size()));
    expect_error([&] { nn::sequence_cross_entropy(ctx, bad_logits, bad_labels, 0); },
                 "label out of range");

    nn::Tensor cls_logits(ctx, {2, 3});
    cls_logits.fill(0.2f);
    nn::DeviceIndices cls_labels(ctx, 2);
    const std::vector<int64_t> cls_negative = {0, -1};
    cls_labels.copy_from_host(cls_negative.data(), static_cast<int64_t>(cls_negative.size()));
    expect_error([&] { nn::softmax_cross_entropy(ctx, cls_logits, cls_labels); }, "label out of range");

    nn::IndexTensor flat_labels(ctx, {3});
    const std::vector<int64_t> flat_values = {0, 1, 2};
    flat_labels.copy_from_host(flat_values.data(), static_cast<int64_t>(flat_values.size()));
    expect_error([&] { nn::sequence_cross_entropy(ctx, bad_logits, flat_labels); }, "label shape");
    nn::IndexTensor wrong_t(ctx, {1, 2});
    const std::vector<int64_t> wrong_t_values = {0, 1};
    wrong_t.copy_from_host(wrong_t_values.data(), static_cast<int64_t>(wrong_t_values.size()));
    expect_error([&] { nn::sequence_cross_entropy(ctx, bad_logits, wrong_t); }, "label shape");

    nn::Tensor row_logits(ctx, {4, 3});
    row_logits.fill(0.f);
    nn::IndexTensor short_labels(ctx, {3});
    short_labels.copy_from_host(flat_values.data(), static_cast<int64_t>(flat_values.size()));
    expect_error([&] { nn::sequence_cross_entropy(ctx, row_logits, short_labels); }, "label shape");
    nn::IndexTensor square_labels(ctx, {2, 2});
    const std::vector<int64_t> square_values = {0, 1, 0, 1};
    square_labels.copy_from_host(square_values.data(), static_cast<int64_t>(square_values.size()));
    expect_error([&] { nn::sequence_cross_entropy(ctx, row_logits, square_labels); }, "label shape");

    nn::Tensor rank4(ctx, {1, 1, 2, 3});
    expect_error([&] { nn::sequence_cross_entropy(ctx, rank4, square_labels); }, "logits rank");
    nn::Tensor rank1(ctx, {4});
    expect_error([&] { nn::sequence_cross_entropy(ctx, rank1, short_labels); }, "logits rank");
    nn::Tensor one_class(ctx, {2, 2, 1});
    nn::IndexTensor one_labels(ctx, {2, 2});
    const std::vector<int64_t> zeros4 = {0, 0, 0, 0};
    one_labels.copy_from_host(zeros4.data(), static_cast<int64_t>(zeros4.size()));
    expect_error([&] { nn::sequence_cross_entropy(ctx, one_class, one_labels); }, "logits shape");
  }

  std::printf("shapes ok\n");
  return 0;
}
