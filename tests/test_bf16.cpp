#include "check.hpp"

#include "nn/builders.hpp"
#include "nn/dtype.hpp"
#include "nn/layers/multi_head_attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

namespace {

constexpr int kBatch = 2;
constexpr int kIn = 8;
constexpr int kHidden = 16;
constexpr int kOut = 4;

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

// atol 1e-2 and rtol 1e-2. The 1e-5 expect_close used by the Float32 tests is too tight for BF16.
void expect_loose(const std::vector<float>& got, const std::vector<float>& want, const char* name) {
  CHECK(got.size() == want.size());
  for (size_t i = 0; i < got.size(); ++i) {
    const double a = static_cast<double>(got[i]);
    const double b = static_cast<double>(want[i]);
    const double diff = std::abs(a - b);
    const double scale = std::max(std::abs(a), std::abs(b));
    if (diff <= 1e-2 + 1e-2 * scale) continue;
    std::fprintf(stderr, "%s[%zu] got %.8g want %.8g\n", name, i, a, b);
    CHECK(false);
  }
}

void expect_finite(const std::vector<float>& values, const char* name) {
  for (float value : values) {
    if (std::isfinite(value)) continue;
    std::fprintf(stderr, "%s has a non-finite value\n", name);
    CHECK(false);
  }
}

uint32_t read_u32(const std::vector<uint8_t>& bytes, size_t offset) {
  CHECK(offset + 4 <= bytes.size());
  return static_cast<uint32_t>(bytes[offset]) | (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
         (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
         (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

size_t align8(size_t offset) { return (offset + 7u) & ~size_t{7}; }

size_t skip_string(const std::vector<uint8_t>& bytes, size_t offset) {
  const uint32_t length = read_u32(bytes, offset);
  CHECK(length <= 256);
  const size_t next = align8(offset + 4 + length);
  CHECK(next <= bytes.size());
  return next;
}

size_t skip_shape(const std::vector<uint8_t>& bytes, size_t offset) {
  const uint32_t rank = read_u32(bytes, offset);
  CHECK(rank == 1 || rank == 2);
  const size_t next = align8(offset + 4 + 4 * static_cast<size_t>(rank));
  CHECK(next <= bytes.size());
  return next;
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

// Version-1 MLP layout. Each parameter record stores dtype tag 1 (float32) ahead of the floats.
void expect_float32_tags(const std::vector<uint8_t>& bytes, uint32_t expect_count) {
  size_t pos = 32;
  const uint32_t hidden = read_u32(bytes, pos + 12);
  CHECK(hidden == 1);
  pos = align8(pos + 16 + 4 * static_cast<size_t>(hidden));
  const uint32_t param_count = read_u32(bytes, pos);
  CHECK(param_count == expect_count);
  pos += 8;
  for (uint32_t i = 0; i < param_count; ++i) {
    pos = skip_string(bytes, pos);
    pos = skip_shape(bytes, pos);
    CHECK(read_u32(bytes, pos) == static_cast<uint32_t>(nn::DType::Float32));
    const uint32_t numel = read_u32(bytes, pos + 4);
    pos = align8(pos + 8 + 4 * static_cast<size_t>(numel));
    CHECK(pos <= bytes.size());
  }
}

std::vector<float> uniform(int64_t n, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> values(static_cast<size_t>(n));
  for (float& value : values) value = dist(rng);
  return values;
}

nn::MlpConfig mlp_config() {
  nn::MlpConfig config;
  config.input_size = kIn;
  config.hidden = {kHidden};
  config.output_size = kOut;
  config.activation = nn::Activation::Gelu;
  return config;
}

}  // namespace

int test_bf16() {
  std::mt19937 rng(1);
  nn::Context bf16;
  CHECK(bf16.activation_dtype() == nn::DType::BFloat16);

  nn::Network net = nn::build_mlp(bf16, mlp_config(), 1);
  const std::vector<nn::NamedParam> params = net.parameters();
  CHECK(params.size() == 4);
  std::vector<std::vector<float>> master;
  master.reserve(params.size());
  for (const nn::NamedParam& param : params) {
    CHECK(param.value->dtype() == nn::DType::Float32);
    CHECK(param.grad->dtype() == nn::DType::Float32);
    CHECK(param.value->data() != nullptr);
    master.push_back(param.value->to_host());
  }

  const std::vector<float> host_input = uniform(static_cast<int64_t>(kBatch) * kIn, rng);
  nn::Tensor input(bf16, {kBatch, kIn}, nn::DType::Float32);
  input.copy_from_host(host_input);
  const nn::Tensor& output = net.forward(input);
  CHECK(output.dtype() == nn::DType::BFloat16);
  const std::vector<float> bf16_out = output.to_host();
  expect_finite(bf16_out, "bf16 mlp");

  nn::Context fp32;
  fp32.set_activation_dtype(nn::DType::Float32);
  nn::Network ref = nn::build_mlp(fp32, mlp_config(), 2);
  const std::vector<nn::NamedParam> ref_params = ref.parameters();
  CHECK(ref_params.size() == params.size());
  for (size_t i = 0; i < params.size(); ++i) {
    CHECK(ref_params[i].name == params[i].name);
    ref_params[i].value->copy_from(*params[i].value);
  }
  nn::Tensor ref_input(fp32, {kBatch, kIn}, nn::DType::Float32);
  ref_input.copy_from_host(host_input);
  const std::vector<float> fp32_out = ref.forward(ref_input).to_host();
  expect_loose(bf16_out, fp32_out, "bf16 mlp");

  const std::vector<float> host_grad(bf16_out.size(), 0.25f);
  nn::Tensor grad_output(bf16, output.shape(), nn::DType::Float32);
  grad_output.copy_from_host(host_grad);
  net.zero_grad();
  net.backward(grad_output);

  nn::Tensor ref_grad(fp32, {kBatch, kOut}, nn::DType::Float32);
  ref_grad.copy_from_host(host_grad);
  ref.zero_grad();
  ref.backward(ref_grad);

  const std::vector<nn::NamedParam> after = net.parameters();
  const std::vector<nn::NamedParam> ref_after = ref.parameters();
  CHECK(after.size() == master.size());
  CHECK(ref_after.size() == master.size());
  for (size_t i = 0; i < after.size(); ++i) {
    CHECK(after[i].grad->dtype() == nn::DType::Float32);
    expect_loose(after[i].grad->to_host(), ref_after[i].grad->to_host(), after[i].name.c_str());
    expect_bits(master[i], after[i].value->to_host(), after[i].name.c_str());
  }

  const std::filesystem::path path = std::filesystem::temp_directory_path() / "nn-bf16" / "mlp.bin";
  std::filesystem::create_directories(path.parent_path());
  net.save(path, nullptr);
  expect_float32_tags(read_bytes(path), static_cast<uint32_t>(master.size()));

  nn::Context loaded_ctx;
  loaded_ctx.set_activation_dtype(nn::DType::Float32);
  nn::Network::Loaded loaded = nn::Network::load(loaded_ctx, path, nn::LoadMode::Inference);
  const std::vector<nn::NamedParam> loaded_params = loaded.network.parameters();
  CHECK(loaded_params.size() == master.size());
  for (size_t i = 0; i < master.size(); ++i) {
    expect_bits(master[i], loaded_params[i].value->to_host(), loaded_params[i].name.c_str());
  }

  // Plain causal attention. QKV projections take the BF16 GEMM path; scores stay on device.
  nn::MultiHeadAttention attention(bf16, 16, 4, true, false, false, rng);
  const std::vector<float> host_seq = uniform(2 * 4 * 16, rng);
  nn::Tensor sequence(bf16, {2, 4, 16}, nn::DType::Float32);
  sequence.copy_from_host(host_seq);
  const nn::Tensor& attn_out = attention.forward(sequence);
  CHECK(attn_out.shape() == std::vector<int>({2, 4, 16}));
  CHECK(attn_out.dtype() == nn::DType::BFloat16);
  expect_finite(attn_out.to_host(), "bf16 mha");

  std::printf("bf16 ok\n");
  return 0;
}
