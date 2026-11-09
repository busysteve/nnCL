#include "check.hpp"

#include "nn/builders.hpp"
#include "nn/conv_resnet.hpp"
#include "nn/error.hpp"
#include "nn/index_tensor.hpp"
#include "nn/loss.hpp"
#include "nn/transformer.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <random>
#include <string>
#include <variant>
#include <vector>

namespace {

std::vector<float> uniform(int64_t n, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> values(static_cast<size_t>(n));
  for (float& value : values) value = dist(rng);
  return values;
}

bool same_bits(const std::vector<float>& left, const std::vector<float>& right) {
  if (left.size() != right.size()) return false;
  return std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0;
}

void expect_bits(const std::vector<float>& left, const std::vector<float>& right, const char* name) {
  if (!same_bits(left, right)) {
    std::fprintf(stderr, "bitwise mismatch: %s (%zu vs %zu)\n", name, left.size(), right.size());
    CHECK(false);
  }
}

void expect_throw(const std::function<void()>& call, const std::string& message) {
  bool threw = false;
  try {
    call();
  } catch (const nn::Error& error) {
    threw = true;
    if (error.what() != message) {
      std::fprintf(stderr, "expected \"%s\", got \"%s\"\n", message.c_str(), error.what());
      CHECK(false);
    }
  }
  CHECK(threw);
}

struct Batch {
  std::vector<int> shape;
  std::vector<float> input;
  std::vector<int64_t> labels;
};

void train_step(nn::Network& net, nn::Sgd& optimizer, nn::Context& ctx, const Batch& batch) {
  net.train();
  net.zero_grad();
  nn::Tensor input(ctx, batch.shape);
  input.copy_from_host(batch.input);
  nn::DeviceIndices labels(ctx, static_cast<int64_t>(batch.labels.size()));
  labels.copy_from_host(batch.labels.data(), labels.size());
  const nn::Tensor& logits = net.forward(input);
  nn::LossResult loss = nn::softmax_cross_entropy(ctx, logits, labels);
  net.backward(loss.grad_logits);
  optimizer.step();
}

std::vector<float> eval_logits(nn::Network& net, nn::Context& ctx, const Batch& batch) {
  net.eval();
  nn::Tensor input(ctx, batch.shape);
  input.copy_from_host(batch.input);
  return net.forward(input).to_host();
}

void compare_state(const nn::Network& left, const nn::Network& right, const nn::Sgd& left_opt,
                   const nn::Sgd& right_opt, bool running_stats) {
  const std::vector<nn::NamedParam> a = left.parameters();
  const std::vector<nn::NamedParam> b = right.parameters();
  CHECK(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    CHECK(a[i].name == b[i].name);
    expect_bits(a[i].value->to_host(), b[i].value->to_host(), a[i].name.c_str());
  }
  if (running_stats) {
    const std::vector<nn::NamedBuffer> ab = left.buffers();
    const std::vector<nn::NamedBuffer> bb = right.buffers();
    CHECK(ab.size() == bb.size());
    for (size_t i = 0; i < ab.size(); ++i) {
      CHECK(ab[i].name == bb[i].name);
      CHECK(*ab[i].batches == *bb[i].batches);
      expect_bits(ab[i].value->to_host(), bb[i].value->to_host(), ab[i].name.c_str());
    }
  }
  CHECK(left_opt.step_count() == right_opt.step_count());
  CHECK(left_opt.learning_rate() == right_opt.learning_rate());
  CHECK(left_opt.momentum() == right_opt.momentum());
  CHECK(left_opt.velocity().size() == right_opt.velocity().size());
  for (size_t i = 0; i < left_opt.velocity().size(); ++i) {
    expect_bits(left_opt.velocity()[i].to_host(), right_opt.velocity()[i].to_host(), "velocity");
  }
}

void train_step_adamw(nn::Network& net, nn::AdamW& optimizer, nn::Context& ctx, const Batch& batch) {
  net.train();
  net.zero_grad();
  nn::Tensor input(ctx, batch.shape);
  input.copy_from_host(batch.input);
  nn::DeviceIndices labels(ctx, static_cast<int64_t>(batch.labels.size()));
  labels.copy_from_host(batch.labels.data(), labels.size());
  const nn::Tensor& logits = net.forward(input);
  nn::LossResult loss = nn::softmax_cross_entropy(ctx, logits, labels);
  net.backward(loss.grad_logits);
  optimizer.step();
}

nn::AdamW make_adamw(nn::Network& net) {
  nn::AdamWGroup decay;
  nn::AdamWGroup plain;
  decay.weight_decay = 0.05f;
  plain.weight_decay = 0.f;
  for (const nn::NamedParam& param : net.parameters()) {
    if (param.name.compare(0, 5, "fc.0.") == 0) decay.params.push_back(param);
    else plain.params.push_back(param);
  }
  return nn::AdamW(std::vector<nn::AdamWGroup>{std::move(decay), std::move(plain)}, 1e-2f, 0.9f, 0.999f,
                   1e-8f);
}

void step_facade_adamw(nn::Transformer& model, nn::AdamW& opt, nn::Context& ctx) {
  const std::vector<int64_t> ids = {1, 0, 1};
  const std::vector<int64_t> labels = {0, 1, 2};
  model.train();
  model.zero_grad();
  nn::IndexTensor tokens(ctx, {1, 3});
  nn::IndexTensor label_ids(ctx, {1, 3});
  tokens.copy_from_host(ids.data(), static_cast<int64_t>(ids.size()));
  label_ids.copy_from_host(labels.data(), static_cast<int64_t>(labels.size()));
  ctx.synchronize();
  const nn::Tensor& logits = model.forward(tokens);
  nn::LossResult loss = nn::sequence_cross_entropy(ctx, logits, label_ids, -100);
  model.backward(loss.grad_logits);
  opt.step();
}

nn::AdamW make_facade_adamw(nn::Transformer& model) {
  std::vector<nn::AdamWGroup> groups;
  for (const nn::NamedParam& param : model.parameters()) {
    const bool decay = param.value->rank() == 2 && param.name.find("ln") == std::string::npos;
    const float weight_decay = decay ? 0.05f : 0.f;
    if (groups.empty() || groups.back().weight_decay != weight_decay) {
      nn::AdamWGroup group;
      group.weight_decay = weight_decay;
      groups.push_back(std::move(group));
    }
    groups.back().params.push_back(param);
  }
  return nn::AdamW(std::move(groups), 1e-2f, 0.9f, 0.95f, 1e-8f);
}

void compare_facade_adamw(const nn::Transformer& left, const nn::Transformer& right,
                          const nn::AdamW& left_opt, const nn::AdamW& right_opt) {
  const std::vector<nn::NamedParam> a = left.parameters();
  const std::vector<nn::NamedParam> b = right.parameters();
  CHECK(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    CHECK(a[i].name == b[i].name);
    expect_bits(a[i].value->to_host(), b[i].value->to_host(), a[i].name.c_str());
  }
  CHECK(left_opt.step_count() == right_opt.step_count());
  CHECK(left_opt.learning_rate() == right_opt.learning_rate());
  CHECK(left_opt.beta1() == right_opt.beta1());
  CHECK(left_opt.beta2() == right_opt.beta2());
  CHECK(left_opt.eps() == right_opt.eps());
  CHECK(left_opt.groups().size() == right_opt.groups().size());
  CHECK(left_opt.groups().size() >= 2);
  for (size_t g = 0; g < left_opt.groups().size(); ++g) {
    CHECK(left_opt.groups()[g].weight_decay == right_opt.groups()[g].weight_decay);
    CHECK(left_opt.groups()[g].params.size() == right_opt.groups()[g].params.size());
  }
  CHECK(left_opt.m().size() == right_opt.m().size());
  CHECK(left_opt.v().size() == right_opt.v().size());
  for (size_t i = 0; i < left_opt.m().size(); ++i) {
    expect_bits(left_opt.m()[i].to_host(), right_opt.m()[i].to_host(), "m");
    expect_bits(left_opt.v()[i].to_host(), right_opt.v()[i].to_host(), "v");
  }
}

void compare_adamw(const nn::Network& left, const nn::Network& right, const nn::AdamW& left_opt,
                   const nn::AdamW& right_opt) {
  const std::vector<nn::NamedParam> a = left.parameters();
  const std::vector<nn::NamedParam> b = right.parameters();
  CHECK(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    CHECK(a[i].name == b[i].name);
    expect_bits(a[i].value->to_host(), b[i].value->to_host(), a[i].name.c_str());
  }
  CHECK(left_opt.step_count() == right_opt.step_count());
  CHECK(left_opt.learning_rate() == right_opt.learning_rate());
  CHECK(left_opt.beta1() == right_opt.beta1());
  CHECK(left_opt.beta2() == right_opt.beta2());
  CHECK(left_opt.eps() == right_opt.eps());
  CHECK(left_opt.groups().size() == right_opt.groups().size());
  for (size_t g = 0; g < left_opt.groups().size(); ++g) {
    CHECK(left_opt.groups()[g].weight_decay == right_opt.groups()[g].weight_decay);
    CHECK(left_opt.groups()[g].params.size() == right_opt.groups()[g].params.size());
  }
  CHECK(left_opt.m().size() == right_opt.m().size());
  CHECK(left_opt.v().size() == right_opt.v().size());
  for (size_t i = 0; i < left_opt.m().size(); ++i) {
    expect_bits(left_opt.m()[i].to_host(), right_opt.m()[i].to_host(), "m");
    expect_bits(left_opt.v()[i].to_host(), right_opt.v()[i].to_host(), "v");
  }
}

void round_trip(nn::Network net, nn::Context& ctx_a, const std::filesystem::path& path, const Batch& train,
                const Batch& probe, bool running_stats) {
  net.train();
  nn::Sgd optimizer(net.parameters(), 0.1f, 0.9f);
  train_step(net, optimizer, ctx_a, train);
  CHECK(optimizer.step_count() == 1);
  const std::vector<float> logits = eval_logits(net, ctx_a, probe);
  net.save(path, &optimizer);

  nn::Context ctx_b;
  nn::Network::Loaded loaded = nn::Network::load(ctx_b, path, nn::LoadMode::Resume);
  CHECK(loaded.optimizer.has_value());
  CHECK(!loaded.adamw.has_value());
  CHECK(loaded.step == 1);
  CHECK(!loaded.network.training());
  compare_state(net, loaded.network, optimizer, *loaded.optimizer, running_stats);
  expect_bits(logits, eval_logits(loaded.network, ctx_b, probe), "logits");

  train_step(net, optimizer, ctx_a, train);
  train_step(loaded.network, *loaded.optimizer, ctx_b, train);
  compare_state(net, loaded.network, optimizer, *loaded.optimizer, running_stats);
}

uint32_t read_u32(const std::vector<uint8_t>& bytes, size_t offset) {
  return static_cast<uint32_t>(bytes[offset]) | (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
         (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
         (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

void write_u32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
  bytes[offset] = static_cast<uint8_t>(value);
  bytes[offset + 1] = static_cast<uint8_t>(value >> 8);
  bytes[offset + 2] = static_cast<uint8_t>(value >> 16);
  bytes[offset + 3] = static_cast<uint8_t>(value >> 24);
}

void write_u64(std::vector<uint8_t>& bytes, size_t offset, uint64_t value) {
  write_u32(bytes, offset, static_cast<uint32_t>(value));
  write_u32(bytes, offset + 4, static_cast<uint32_t>(value >> 32));
}

float read_f32(const std::vector<uint8_t>& bytes, size_t offset) {
  const uint32_t bits = read_u32(bytes, offset);
  float value = 0.f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
  bytes.push_back(static_cast<uint8_t>(value));
  bytes.push_back(static_cast<uint8_t>(value >> 8));
  bytes.push_back(static_cast<uint8_t>(value >> 16));
  bytes.push_back(static_cast<uint8_t>(value >> 24));
}

void append_f32(std::vector<uint8_t>& bytes, float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  append_u32(bytes, bits);
}

size_t align8(size_t offset) { return (offset + 7u) & ~size_t{7}; }

size_t skip_string(const std::vector<uint8_t>& bytes, size_t offset) {
  const uint32_t length = read_u32(bytes, offset);
  return align8(offset + 4 + length);
}

size_t skip_shape(const std::vector<uint8_t>& bytes, size_t offset) {
  const uint32_t rank = read_u32(bytes, offset);
  return align8(offset + 4 + 4 * rank);
}

std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  CHECK(static_cast<bool>(in));
  in.seekg(0, std::ios::end);
  const auto size = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  in.read(reinterpret_cast<char*>(bytes.data()), size);
  CHECK(static_cast<bool>(in));
  return bytes;
}

void write_bytes(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
  std::ofstream out(path, std::ios::binary);
  CHECK(static_cast<bool>(out));
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  CHECK(static_cast<bool>(out));
}

size_t optimizer_offset(const std::vector<uint8_t>& bytes) {
  size_t pos = 32;
  const uint32_t hidden = read_u32(bytes, pos + 12);
  pos = align8(pos + 16 + 4 * hidden);
  const uint32_t param_count = read_u32(bytes, pos);
  pos += 8;
  for (uint32_t i = 0; i < param_count; ++i) {
    pos = skip_string(bytes, pos);
    pos = skip_shape(bytes, pos);
    const uint32_t numel = read_u32(bytes, pos + 4);
    pos = align8(pos + 8 + 4 * static_cast<size_t>(numel));
  }
  const uint32_t bn_count = read_u32(bytes, pos);
  CHECK(bn_count == 0);
  pos += 8;
  return pos;
}

}  // namespace

int test_checkpoint() {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "nn-ckpt";
  std::filesystem::create_directories(dir);
  nn::Context ctx;
  std::mt19937 rng(1);

  {
    nn::MlpConfig config;
    config.input_size = 8;
    config.hidden = {6};
    config.output_size = 4;
    Batch train{{2, 8}, uniform(16, rng), {0, 1}};
    Batch probe{{2, 8}, uniform(16, rng), {1, 0}};
    round_trip(nn::build_mlp(ctx, config, 1), ctx, dir / "mlp.bin", train, probe, false);
  }
  {
    nn::MlpConfig config;
    config.input_size = 8;
    config.hidden = {6};
    config.output_size = 4;
    config.activation = nn::Activation::Gelu;
    Batch train{{2, 8}, uniform(16, rng), {0, 1}};
    Batch probe{{2, 8}, uniform(16, rng), {1, 0}};
    round_trip(nn::build_mlp(ctx, config, 1), ctx, dir / "mlp-gelu.bin", train, probe, false);
    nn::Network::Loaded loaded =
        nn::Network::load(ctx, dir / "mlp-gelu.bin", nn::LoadMode::Inference);
    CHECK(std::get<nn::MlpConfig>(loaded.network.config()).activation == nn::Activation::Gelu);
  }
  {
    nn::MlpConfig config;
    config.input_size = 8;
    config.hidden = {6};
    config.output_size = 4;
    config.activation = nn::Activation::SwiGLU;
    Batch train{{2, 8}, uniform(16, rng), {0, 1}};
    Batch probe{{2, 8}, uniform(16, rng), {1, 0}};
    round_trip(nn::build_mlp(ctx, config, 1), ctx, dir / "mlp-swiglu.bin", train, probe, false);
    nn::Network::Loaded loaded =
        nn::Network::load(ctx, dir / "mlp-swiglu.bin", nn::LoadMode::Inference);
    CHECK(std::get<nn::MlpConfig>(loaded.network.config()).activation == nn::Activation::SwiGLU);
    CHECK(loaded.network.parameters()[0].value->shape() == std::vector<int>({12, 8}));
  }
  {
    nn::RnnConfig config;
    config.input_size = 4;
    config.hidden = {3, 5};
    config.output_size = 4;
    Batch train{{2, 3, 4}, uniform(24, rng), {0, 1}};
    Batch probe{{2, 3, 4}, uniform(24, rng), {1, 0}};
    round_trip(nn::build_rnn(ctx, config, 1), ctx, dir / "rnn.bin", train, probe, false);
    nn::Network::Loaded loaded = nn::Network::load(ctx, dir / "rnn.bin", nn::LoadMode::Inference);
    CHECK(loaded.network.family() == nn::Family::Rnn);
    const nn::RnnConfig& restored = std::get<nn::RnnConfig>(loaded.network.config());
    CHECK(restored.hidden.size() == 2);
    CHECK(restored.hidden[1] == 5);
    CHECK(loaded.network.parameters()[0].value->shape() == std::vector<int>({3, 4}));
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
    config.tie_embeddings = false;
    config.activation = nn::TransformerActivation::SwiGLU;
    config.positional = nn::PositionalEncoding::LearnedAbsolute;
    nn::Network net = nn::build_transformer_network(ctx, config, 1);
    net.train();
    nn::Sgd optimizer(net.parameters(), 0.1f, 0.9f);
    net.zero_grad();
    nn::Tensor input(ctx, {1, 3, 8});
    input.copy_from_host(uniform(24, rng));
    const nn::Tensor& output = net.forward(input);
    CHECK(output.shape() == std::vector<int>({1, 3, 10}));
    nn::Tensor grad(ctx, output.shape());
    grad.copy_from_host(uniform(static_cast<int64_t>(output.numel()), rng));
    net.backward(grad);
    optimizer.step();
    CHECK(optimizer.step_count() == 1);
    const std::filesystem::path path = dir / "transformer.bin";
    net.save(path, &optimizer);

    nn::Context ctx_b;
    nn::Network::Loaded resumed = nn::Network::load(ctx_b, path, nn::LoadMode::Resume);
    CHECK(resumed.optimizer.has_value());
    CHECK(resumed.network.family() == nn::Family::Transformer);
    compare_state(net, resumed.network, optimizer, *resumed.optimizer, false);

    nn::Network::Loaded loaded = nn::Network::load(ctx, path, nn::LoadMode::Inference);
    CHECK(!loaded.optimizer);
    CHECK(loaded.network.family() == nn::Family::Transformer);
    const nn::TransformerConfig& restored = std::get<nn::TransformerConfig>(loaded.network.config());
    CHECK(restored.vocab_size == config.vocab_size);
    CHECK(restored.d_model == config.d_model);
    CHECK(restored.num_heads == config.num_heads);
    CHECK(restored.num_layers == config.num_layers);
    CHECK(restored.d_ff == config.d_ff);
    CHECK(restored.max_seq_len == config.max_seq_len);
    CHECK(restored.dropout == config.dropout);
    CHECK(restored.tie_embeddings == false);
    CHECK(restored.bias == config.bias);
    CHECK(restored.activation == config.activation);
    CHECK(restored.positional == config.positional);
    expect_throw([&] { nn::Transformer::load(ctx, path, nn::LoadMode::Resume); },
                 "checkpoint tensor mismatch: count");
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
    nn::Transformer model(ctx, config, 1);
    model.train();
    nn::Sgd optimizer(model.parameters(), 0.1f, 0.9f);
    model.zero_grad();
    nn::IndexTensor tokens(ctx, {1, 3});
    const std::vector<int64_t> ids = {1, 0, 1};
    tokens.copy_from_host(ids.data(), static_cast<int64_t>(ids.size()));
    const nn::Tensor& logits = model.forward(tokens);
    nn::Tensor grad(ctx, logits.shape());
    grad.copy_from_host(uniform(logits.numel(), rng));
    model.backward(grad);
    optimizer.step();
    CHECK(optimizer.step_count() == 1);

    const std::filesystem::path path = dir / "transformer-facade.bin";
    model.save(path, &optimizer);
    CHECK(read_u32(read_bytes(path), 4) == 2);
    expect_throw([&] { nn::Network::load(ctx, path, nn::LoadMode::Inference); },
                 "checkpoint tensor mismatch: count");

    nn::Context ctx_b;
    nn::Transformer::Loaded resumed = nn::Transformer::load(ctx_b, path, nn::LoadMode::Resume);
    CHECK(resumed.optimizer.has_value());
    CHECK(!resumed.adamw.has_value());
    CHECK(resumed.step == 1);
    CHECK(resumed.model.network().family() == nn::Family::Transformer);
    const std::vector<nn::NamedParam> left = model.parameters();
    const std::vector<nn::NamedParam> right = resumed.model.parameters();
    CHECK(left.size() == right.size());
    for (size_t i = 0; i < left.size(); ++i) {
      CHECK(left[i].name == right[i].name);
      expect_bits(left[i].value->to_host(), right[i].value->to_host(), left[i].name.c_str());
    }
    CHECK(optimizer.step_count() == resumed.optimizer->step_count());
    CHECK(optimizer.learning_rate() == resumed.optimizer->learning_rate());
    CHECK(optimizer.momentum() == resumed.optimizer->momentum());
    CHECK(optimizer.velocity().size() == resumed.optimizer->velocity().size());
    for (size_t i = 0; i < optimizer.velocity().size(); ++i) {
      expect_bits(optimizer.velocity()[i].to_host(), resumed.optimizer->velocity()[i].to_host(),
                  "velocity");
    }

    const std::filesystem::path infer_path = dir / "transformer-facade-inf.bin";
    model.save(infer_path, nullptr);
    CHECK(read_u32(read_bytes(infer_path), 4) == 1);
    expect_throw([&] { nn::Transformer::load(ctx, infer_path, nn::LoadMode::Resume); },
                 "optimizer block missing");
    expect_throw([&] { nn::Network::load(ctx, infer_path, nn::LoadMode::Inference); },
                 "checkpoint tensor mismatch: count");
    nn::Transformer::Loaded loaded =
        nn::Transformer::load(ctx, infer_path, nn::LoadMode::Inference);
    CHECK(!loaded.optimizer);
    CHECK(!loaded.adamw);
    CHECK(loaded.step == 0);
    const nn::TransformerConfig& restored =
        std::get<nn::TransformerConfig>(loaded.model.network().config());
    CHECK(restored.vocab_size == config.vocab_size);
    CHECK(restored.d_model == config.d_model);
    CHECK(restored.num_heads == config.num_heads);
    CHECK(restored.num_layers == config.num_layers);
    CHECK(restored.d_ff == config.d_ff);
    CHECK(restored.max_seq_len == config.max_seq_len);
    CHECK(restored.dropout == config.dropout);
    CHECK(restored.tie_embeddings == config.tie_embeddings);
    CHECK(restored.bias == config.bias);
    CHECK(restored.activation == config.activation);
    CHECK(restored.positional == config.positional);
    const std::vector<nn::NamedParam> inferred = loaded.model.parameters();
    CHECK(inferred.size() == left.size());
    for (size_t i = 0; i < left.size(); ++i) {
      CHECK(left[i].name == inferred[i].name);
      expect_bits(left[i].value->to_host(), inferred[i].value->to_host(), left[i].name.c_str());
    }
  }
  {
    nn::ConvConfig config;
    config.in_channels = 1;
    config.in_height = 8;
    config.in_width = 8;
    config.convs = {nn::ConvSpec{3, 3, 1, 1}};
    config.pool = nn::PoolSpec{2, 2};
    config.fc = {4, 4};
    Batch train{{2, 1, 8, 8}, uniform(2 * 64, rng), {0, 1}};
    Batch probe{{2, 1, 8, 8}, uniform(2 * 64, rng), {1, 2}};
    round_trip(nn::build_conv(ctx, config, 1), ctx, dir / "conv.bin", train, probe, false);
  }
  {
    nn::ResNetConfig config;
    config.in_channels = 1;
    config.in_height = 8;
    config.in_width = 8;
    config.stem_channels = 4;
    config.blocks_per_stage = {1, 1};
    config.stage_widths = {4, 8};
    config.output_size = 4;
    Batch train{{2, 1, 8, 8}, uniform(2 * 64, rng), {0, 1}};
    Batch probe{{2, 1, 8, 8}, uniform(2 * 64, rng), {2, 3}};
    round_trip(nn::build_resnet(ctx, config, 1), ctx, dir / "resnet.bin", train, probe, true);
  }
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
    Batch train{{2, 1, 8, 8}, uniform(2 * 64, rng), {0, 1}};
    Batch probe{{2, 1, 8, 8}, uniform(2 * 64, rng), {2, 3}};
    const std::filesystem::path path = dir / "convresnet.bin";
    round_trip(nn::build_conv_resnet(ctx, config, 1), ctx, path, train, probe, true);

    nn::ConvResNet::Loaded loaded = nn::ConvResNet::load(ctx, path, nn::LoadMode::Resume);
    CHECK(loaded.model.family() == nn::Family::ConvResNet);
    CHECK(loaded.optimizer.has_value());
    CHECK(!loaded.adamw.has_value());
    CHECK(loaded.step == 1);
    CHECK(loaded.model.config().fc == config.fc);
    CHECK(loaded.model.config().convs.size() == 1);
    CHECK(loaded.model.config().blocks_per_stage.size() == 2);
  }

  const std::filesystem::path mlp = dir / "mlp.bin";
  {
    const std::vector<uint8_t> bytes = read_bytes(mlp);
    write_bytes(dir / "trunc.bin", std::vector<uint8_t>(bytes.begin(), bytes.begin() + 16));
    expect_throw([&] { nn::Network::load(ctx, dir / "trunc.bin", nn::LoadMode::Inference); },
                 "truncated checkpoint");
  }
  {
    std::vector<uint8_t> bytes = read_bytes(mlp);
    write_u32(bytes, 4, 99);
    write_bytes(dir / "version.bin", bytes);
    expect_throw([&] { nn::Network::load(ctx, dir / "version.bin", nn::LoadMode::Inference); },
                 "unsupported checkpoint version 99");
  }
  {
    std::vector<uint8_t> bytes = read_bytes(mlp);
    const std::string name = "fc.0.weight";
    const auto found = std::search(bytes.begin(), bytes.end(), name.begin(), name.end());
    CHECK(found != bytes.end());
    const size_t name_at = static_cast<size_t>(found - bytes.begin());
    CHECK(read_u32(bytes, name_at - 4) == name.size());
    size_t pos = align8(name_at + name.size());
    CHECK(read_u32(bytes, pos) == 2);
    const int32_t dim = static_cast<int32_t>(read_u32(bytes, pos + 4));
    write_u32(bytes, pos + 4, static_cast<uint32_t>(dim + 1));
    write_bytes(dir / "shape.bin", bytes);
    expect_throw([&] { nn::Network::load(ctx, dir / "shape.bin", nn::LoadMode::Inference); },
                 "checkpoint tensor mismatch: fc.0.weight");
  }
  {
    std::vector<uint8_t> bytes = read_bytes(mlp);
    bytes.resize(optimizer_offset(bytes));
    write_u64(bytes, 16, 0);
    write_u32(bytes, 24, 0);
    write_bytes(dir / "noopt.bin", bytes);
    expect_throw([&] { nn::Network::load(ctx, dir / "noopt.bin", nn::LoadMode::Resume); },
                 "optimizer block missing");
  }
  {
    nn::Network::Loaded loaded = nn::Network::load(ctx, mlp, nn::LoadMode::Inference);
    CHECK(!loaded.optimizer.has_value());
    CHECK(!loaded.adamw.has_value());
    CHECK(loaded.step == 1);
    expect_throw([&] { nn::ConvResNet::load(ctx, mlp, nn::LoadMode::Inference); },
                 "bad checkpoint family");
    CHECK(!loaded.network.training());
  }

  {
    nn::MlpConfig config;
    config.input_size = 8;
    config.hidden = {6};
    config.output_size = 4;
    Batch train{{2, 8}, uniform(16, rng), {0, 1}};
    Batch probe{{2, 8}, uniform(16, rng), {1, 0}};
    nn::Network net = nn::build_mlp(ctx, config, 3);
    net.train();
    nn::AdamW optimizer = make_adamw(net);
    const std::vector<nn::NamedParam> params = net.parameters();
    nn::AdamWGroup swapped_decay;
    swapped_decay.weight_decay = 0.05f;
    swapped_decay.params = {params[2], params[3]};
    nn::AdamWGroup swapped_plain;
    swapped_plain.params = {params[0], params[1]};
    nn::AdamW swapped(std::vector<nn::AdamWGroup>{std::move(swapped_decay), std::move(swapped_plain)},
                      1e-2f, 0.9f, 0.999f, 1e-8f);
    expect_throw([&] { net.save(dir / "adamw-order.bin", &swapped); },
                 "optimizer does not belong to this network");

    train_step_adamw(net, optimizer, ctx, train);
    CHECK(optimizer.step_count() == 1);
    CHECK(optimizer.groups().size() == 2);
    CHECK(optimizer.groups()[0].weight_decay == 0.05f);
    CHECK(optimizer.groups()[1].weight_decay == 0.f);
    bool nonzero = false;
    for (float value : optimizer.m()[0].to_host()) nonzero = nonzero || value != 0.f;
    CHECK(nonzero);
    const std::vector<float> logits = eval_logits(net, ctx, probe);
    const std::filesystem::path path = dir / "adamw.bin";
    net.save(path, &optimizer);
    const std::vector<uint8_t> saved = read_bytes(path);
    CHECK(read_u32(saved, 4) == 2);
    CHECK(read_u32(saved, 24) == 1);
    CHECK(read_u32(saved, optimizer_offset(saved)) == 2);

    nn::Context ctx_b;
    nn::Network::Loaded loaded = nn::Network::load(ctx_b, path, nn::LoadMode::Resume);
    CHECK(loaded.adamw.has_value());
    CHECK(!loaded.optimizer.has_value());
    CHECK(loaded.step == 1);
    CHECK(!loaded.network.training());
    compare_adamw(net, loaded.network, optimizer, *loaded.adamw);
    expect_bits(logits, eval_logits(loaded.network, ctx_b, probe), "adamw logits");

    train_step_adamw(net, optimizer, ctx, train);
    train_step_adamw(loaded.network, *loaded.adamw, ctx_b, train);
    compare_adamw(net, loaded.network, optimizer, *loaded.adamw);

    nn::Network::Loaded inferred = nn::Network::load(ctx, path, nn::LoadMode::Inference);
    CHECK(!inferred.optimizer.has_value());
    CHECK(!inferred.adamw.has_value());
    CHECK(inferred.step == 1);

    net.save(dir / "adamw-null.bin", static_cast<const nn::AdamW*>(nullptr));
    const std::vector<uint8_t> empty_opt = read_bytes(dir / "adamw-null.bin");
    CHECK(read_u32(empty_opt, 4) == 2);
    CHECK(read_u32(empty_opt, 24) == 0);
    expect_throw([&] { nn::Network::load(ctx, dir / "adamw-null.bin", nn::LoadMode::Resume); },
                 "optimizer block missing");

    std::vector<uint8_t> truncated = saved;
    truncated.resize(optimizer_offset(truncated) + 2);
    write_bytes(dir / "adamw-trunc.bin", truncated);
    expect_throw([&] { nn::Network::load(ctx, dir / "adamw-trunc.bin", nn::LoadMode::Inference); },
                 "truncated checkpoint");
  }
  {
    // Kind is 4 bytes, so a v1 optimizer slice is not byte-identical: pad-to-8 is a file
    // offset. Re-encode the same fields after the kind tag.
    const std::vector<uint8_t> bytes = read_bytes(mlp);
    const size_t at = optimizer_offset(bytes);
    size_t pos = at;
    const float learning_rate = read_f32(bytes, pos);
    pos += 4;
    const float momentum = read_f32(bytes, pos);
    pos += 4;
    const uint32_t count = read_u32(bytes, pos);
    pos += 8;
    std::vector<std::vector<float>> velocities(count);
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t numel = read_u32(bytes, pos);
      pos += 8;
      velocities[i].resize(numel);
      for (uint32_t j = 0; j < numel; ++j) {
        velocities[i][j] = read_f32(bytes, pos);
        pos += 4;
      }
      pos = align8(pos);
    }
    std::vector<uint8_t> rebuilt(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(at));
    write_u32(rebuilt, 4, 2);
    append_u32(rebuilt, 1);
    append_f32(rebuilt, learning_rate);
    append_f32(rebuilt, momentum);
    append_u32(rebuilt, count);
    append_u32(rebuilt, 0);
    for (const std::vector<float>& velocity : velocities) {
      append_u32(rebuilt, static_cast<uint32_t>(velocity.size()));
      append_u32(rebuilt, 0);
      for (float value : velocity) append_f32(rebuilt, value);
      while (rebuilt.size() % 8 != 0) rebuilt.push_back(0);
    }
    write_bytes(dir / "v2sgd.bin", rebuilt);
    nn::Network::Loaded from_v2 = nn::Network::load(ctx, dir / "v2sgd.bin", nn::LoadMode::Resume);
    nn::Network::Loaded from_v1 = nn::Network::load(ctx, mlp, nn::LoadMode::Resume);
    CHECK(from_v2.optimizer.has_value());
    CHECK(!from_v2.adamw.has_value());
    CHECK(!from_v1.adamw.has_value());
    CHECK(from_v2.step == from_v1.step);
    CHECK(from_v2.optimizer->learning_rate() == from_v1.optimizer->learning_rate());
    CHECK(from_v2.optimizer->momentum() == from_v1.optimizer->momentum());
    CHECK(from_v2.optimizer->step_count() == from_v1.optimizer->step_count());
    CHECK(from_v2.optimizer->velocity().size() == from_v1.optimizer->velocity().size());
    for (size_t i = 0; i < from_v2.optimizer->velocity().size(); ++i) {
      expect_bits(from_v2.optimizer->velocity()[i].to_host(), from_v1.optimizer->velocity()[i].to_host(),
                  "v2 sgd velocity");
    }
    const std::vector<nn::NamedParam> a = from_v2.network.parameters();
    const std::vector<nn::NamedParam> b = from_v1.network.parameters();
    CHECK(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
      expect_bits(a[i].value->to_host(), b[i].value->to_host(), "v2 sgd weight");
    }

    write_u32(rebuilt, at, 7);
    write_bytes(dir / "kind.bin", rebuilt);
    expect_throw([&] { nn::Network::load(ctx, dir / "kind.bin", nn::LoadMode::Inference); },
                 "unsupported optimizer kind 7");

    std::vector<uint8_t> bare = read_bytes(mlp);
    bare.resize(optimizer_offset(bare));
    write_u32(bare, 4, 2);
    write_u64(bare, 16, 0);
    write_u32(bare, 24, 0);
    write_bytes(dir / "v2noopt.bin", bare);
    expect_throw([&] { nn::Network::load(ctx, dir / "v2noopt.bin", nn::LoadMode::Resume); },
                 "optimizer block missing");
    nn::Network::Loaded inferred = nn::Network::load(ctx, dir / "v2noopt.bin", nn::LoadMode::Inference);
    CHECK(!inferred.optimizer.has_value());
    CHECK(!inferred.adamw.has_value());
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
    nn::Transformer model(ctx, config, 1);
    {
      const std::vector<nn::NamedParam> params = model.parameters();
      nn::AdamWGroup late;
      late.weight_decay = 0.05f;
      late.params.assign(params.begin() + 1, params.end());
      nn::AdamWGroup early;
      early.weight_decay = 0.f;
      early.params.push_back(params.front());
      nn::AdamW swapped(
          std::vector<nn::AdamWGroup>{std::move(late), std::move(early)}, 1e-2f, 0.9f, 0.95f, 1e-8f);
      expect_throw([&] { model.save(dir / "facade-adamw-order.bin", &swapped); },
                   "optimizer does not belong to this network");
    }

    nn::AdamW optimizer = make_facade_adamw(model);
    CHECK(optimizer.groups().size() >= 2);
    step_facade_adamw(model, optimizer, ctx);
    CHECK(optimizer.step_count() == 1);
    bool nonzero = false;
    for (float value : optimizer.m()[0].to_host()) nonzero = nonzero || value != 0.f;
    CHECK(nonzero);

    const std::filesystem::path path = dir / "facade-adamw.bin";
    model.save(path, &optimizer);
    const std::vector<uint8_t> saved = read_bytes(path);
    CHECK(read_u32(saved, 4) == 2);
    CHECK(read_u32(saved, 8) == 6);
    CHECK(read_u32(saved, 24) == 1);

    nn::Context ctx_b;
    nn::Transformer::Loaded resumed = nn::Transformer::load(ctx_b, path, nn::LoadMode::Resume);
    CHECK(resumed.adamw.has_value());
    CHECK(!resumed.optimizer.has_value());
    CHECK(resumed.step == 1);
    compare_facade_adamw(model, resumed.model, optimizer, *resumed.adamw);

    step_facade_adamw(model, optimizer, ctx);
    step_facade_adamw(resumed.model, *resumed.adamw, ctx_b);
    CHECK(optimizer.step_count() == 2);
    compare_facade_adamw(model, resumed.model, optimizer, *resumed.adamw);

    nn::Transformer::Loaded inferred = nn::Transformer::load(ctx, path, nn::LoadMode::Inference);
    CHECK(!inferred.optimizer.has_value());
    CHECK(!inferred.adamw.has_value());
    CHECK(inferred.step == 1);

    model.save(dir / "facade-adamw-null.bin", static_cast<const nn::AdamW*>(nullptr));
    const std::vector<uint8_t> empty_opt = read_bytes(dir / "facade-adamw-null.bin");
    CHECK(read_u32(empty_opt, 4) == 2);
    CHECK(read_u32(empty_opt, 24) == 0);
    expect_throw([&] { nn::Transformer::load(ctx, dir / "facade-adamw-null.bin", nn::LoadMode::Resume); },
                 "optimizer block missing");
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
    config.positional = nn::PositionalEncoding::Rope;
    nn::Transformer model(ctx, config, 3);
    const std::vector<nn::NamedParam> params = model.parameters();
    CHECK(params[1].name == "blocks.0.ln1.weight");
    for (const nn::NamedParam& param : params) {
      CHECK(param.name.find("pos_emb") == std::string::npos);
      CHECK(param.name.size() < 5 || param.name.compare(param.name.size() - 5, 5, ".bias") != 0);
    }
    const std::filesystem::path path = dir / "facade-rope.bin";
    model.save(path, nullptr);
    nn::IndexTensor tokens(ctx, {1, 3});
    const std::vector<int64_t> ids = {1, 0, 2};
    tokens.copy_from_host(ids.data(), static_cast<int64_t>(ids.size()));
    const std::vector<float> logits = model.forward(tokens).to_host();
    nn::Transformer::Loaded loaded = nn::Transformer::load(ctx, path, nn::LoadMode::Inference);
    CHECK(loaded.model.parameters().size() == params.size());
    for (size_t i = 0; i < params.size(); ++i) {
      CHECK(loaded.model.parameters()[i].name == params[i].name);
      expect_bits(loaded.model.parameters()[i].value->to_host(), params[i].value->to_host(),
                  params[i].name.c_str());
    }
    expect_bits(loaded.model.forward(tokens).to_host(), logits, "rope logits");
  }

  std::printf("ckpt ok\n");
  return 0;
}
