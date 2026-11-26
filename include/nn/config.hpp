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
  // 0 keeps head_dim = d_model / num_heads and one KV head per query head.
  int head_dim = 0;
  int num_kv_heads = 0;
  bool qk_norm = false;      // RMSNorm on Q and K, width head_dim, before RoPE
  float rope_theta = 10000.f;
  float rms_eps = 1e-5f;
  bool causal = true;        // false is bidirectional attention, as in BERT
  int type_vocab_size = 0;   // 0 omits the segment table. BERT uses 2.
  // Linear bias per projection. split_bias is not written to the file.
  // When it is false, construction sets the four flags equal to bias.
  bool qkv_bias = false;
  bool attn_out_bias = false;
  bool mlp_bias = false;
  bool lm_head_bias = false;
  bool split_bias = false;
  bool gelu_tanh = false;          // erf GELU until a later step builds the tanh form
  int rope_dims = 0;               // 0 rotates the whole head
  int sliding_window = 0;          // 0 is a strict causal mask
  uint32_t rope_scaling = 0;       // 0 none, 1 llama3
  float rope_factor = 0.f;
  float rope_low_freq_factor = 0.f;
  float rope_high_freq_factor = 0.f;
  int rope_original_context = 0;
  bool post_norm = false;
  bool embed_norm = false;
  float embed_norm_eps = 1e-5f;
  bool final_norm = true;
  bool mlm_head = false;
};

int conv_out_size(int in, int kernel, int stride, int pad);

void validate(const MlpConfig& config);
void validate(const ConvConfig& config);
void validate(const ResNetConfig& config);
void validate(const ConvResNetConfig& config);
void validate(const RnnConfig& config);
void validate(const TransformerConfig& config);
TransformerConfig apply_bias_flags(TransformerConfig config);

}  // namespace nn
