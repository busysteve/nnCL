#pragma once

#include <cstdint>
#include <vector>

namespace nn {

enum class Activation : uint32_t { Relu = 1, Gelu = 2, SwiGLU = 3 };

struct MlpConfig {
  int input_size = 0;
  std::vector<int> hidden;
  int output_size = 0;
  Activation activation = Activation::Relu;
};

struct ConvSpec {
  int out_channels = 0;
  int kernel = 0;
  int stride = 0;
  int pad = 0;
};

struct PoolSpec {
  int kernel = 0;
  int stride = 0;
};

struct ConvConfig {
  int in_channels = 0;
  int in_height = 0;
  int in_width = 0;
  std::vector<ConvSpec> convs;
  PoolSpec pool;
  std::vector<int> fc;
};

struct ResNetConfig {
  int in_channels = 0;
  int in_height = 0;
  int in_width = 0;
  int stem_channels = 0;
  std::vector<int> blocks_per_stage;
  std::vector<int> stage_widths;
  int output_size = 0;
};

// One image network that can include a conv+pool trunk, a residual stem and
// stages, or both. The fully connected head is always present.
struct ConvResNetConfig {
  int in_channels = 0;
  int in_height = 0;
  int in_width = 0;
  std::vector<ConvSpec> convs;
  PoolSpec pool;
  int stem_channels = 0;
  std::vector<int> blocks_per_stage;
  std::vector<int> stage_widths;
  std::vector<int> fc;
};

struct RnnConfig {
  int input_size = 0;
  std::vector<int> hidden;
  int output_size = 0;
};

enum class TransformerActivation : uint32_t { Gelu = 1, SwiGLU = 2 };
enum class PositionalEncoding : uint32_t { LearnedAbsolute = 1, Rope = 2 };

struct TransformerConfig {
  int vocab_size = 0;
  int d_model = 0;
  int num_heads = 0;
  int num_layers = 0;
  int d_ff = 0;              // post-gate width H; SwiGLU fc1 writes 2H
  int max_seq_len = 0;
  float dropout = 0.f;
  bool tie_embeddings = false;
  bool bias = false;
  TransformerActivation activation = TransformerActivation::SwiGLU;
  PositionalEncoding positional = PositionalEncoding::LearnedAbsolute;
};

int conv_out_size(int in, int kernel, int stride, int pad);

void validate(const MlpConfig& config);
void validate(const ConvConfig& config);
void validate(const ResNetConfig& config);
void validate(const ConvResNetConfig& config);
void validate(const RnnConfig& config);
void validate(const TransformerConfig& config);

}  // namespace nn
