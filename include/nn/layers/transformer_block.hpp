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
};

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
