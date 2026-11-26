#pragma once

#include "nn/config.hpp"
#include "nn/layer.hpp"
#include "nn/layers/multi_head_attention.hpp"

#include <memory>
#include <optional>
#include <random>

namespace nn {

class Dropout;
class Linear;

struct TransformerBlockSpec {
  int d_model = 0;
  int num_heads = 0;
  int d_ff = 0;           // width AFTER the gate. SwiGLU's first linear writes 2*d_ff.
  float dropout = 0.f;    // in [0, 1)
  bool bias = false;
  bool use_rope = false;  // RMSNorm, and adjacent-pair RoPE on Q and K. Odd head dim throws.
  TransformerActivation activation = TransformerActivation::SwiGLU;
  int head_dim = 0;       // 0 => d_model / num_heads. Qwen3 uses a wider head.
  int num_kv_heads = 0;   // 0 => num_heads. Smaller means grouped-query attention.
  bool qk_norm = false;
  float rope_theta = 10000.f;
  float rms_eps = 1e-5f;
  bool causal = true;
  bool qkv_bias = false;
  bool attn_out_bias = false;
  bool mlp_bias = false;
  bool lm_head_bias = false;
  bool split_bias = false;
  bool gelu_tanh = false;
  int rope_dims = 0;
  int sliding_window = 0;
  uint32_t rope_scaling = 0;
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

inline void fill_transformer_block_spec(TransformerBlockSpec& spec, const TransformerConfig& config) {
  spec.d_model = config.d_model;
  spec.num_heads = config.num_heads;
  spec.d_ff = config.d_ff;
  spec.dropout = config.dropout;
  spec.bias = config.bias;
  spec.qkv_bias = config.qkv_bias;
  spec.attn_out_bias = config.attn_out_bias;
  spec.mlp_bias = config.mlp_bias;
  spec.lm_head_bias = config.lm_head_bias;
  spec.split_bias = config.split_bias;
  spec.use_rope = config.positional == PositionalEncoding::Rope;
  spec.activation = config.activation;
  spec.head_dim = config.head_dim;
  spec.num_kv_heads = config.num_kv_heads;
  spec.qk_norm = config.qk_norm;
  spec.rope_theta = config.rope_theta;
  spec.rms_eps = config.rms_eps;
  spec.causal = config.causal;
  spec.gelu_tanh = config.gelu_tanh;
  spec.rope_dims = config.rope_dims;
  spec.sliding_window = config.sliding_window;
  spec.rope_scaling = config.rope_scaling;
  spec.rope_factor = config.rope_factor;
  spec.rope_low_freq_factor = config.rope_low_freq_factor;
  spec.rope_high_freq_factor = config.rope_high_freq_factor;
  spec.rope_original_context = config.rope_original_context;
  spec.post_norm = config.post_norm;
  spec.embed_norm = config.embed_norm;
  spec.embed_norm_eps = config.embed_norm_eps;
  spec.final_norm = config.final_norm;
  spec.mlm_head = config.mlm_head;
}

class TransformerBlock : public Layer {
 public:
  TransformerBlock(Context& ctx, const TransformerBlockSpec& spec, std::mt19937& rng);
  ~TransformerBlock() override;
  const Tensor& forward(const Tensor& input) override;  // (N,T,D)
  const Tensor& forward_cached(const Tensor& x_t, MultiHeadAttention::Cache& cache);
  const Tensor& backward(const Tensor& grad_output) override;
  std::vector<NamedParam> parameters() const override;
  void zero_grad() override;
  void set_training(bool training) override;
  MultiHeadAttention& attn() { return *mha_; }
  const MultiHeadAttention& attn() const { return *mha_; }

 private:
  Context* ctx_;
  int d_model_;
  std::unique_ptr<Layer> ln1_;
  std::unique_ptr<MultiHeadAttention> mha_;
  std::unique_ptr<Dropout> drop1_;
  std::unique_ptr<Layer> ln2_;
  std::unique_ptr<Linear> fc1_;
  std::unique_ptr<Layer> act_;
  std::unique_ptr<Linear> fc2_;
  std::unique_ptr<Dropout> drop2_;
  std::optional<Tensor> hidden_;
  std::optional<Tensor> output_;
  std::optional<Tensor> grad_hidden_;
  std::optional<Tensor> grad_input_;
};

}  // namespace nn
