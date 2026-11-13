#pragma once

#include "nn/layer.hpp"
#include "nn/layers/linear.hpp"
#include "nn/layers/rms_norm.hpp"

#include <memory>
#include <optional>
#include <random>

namespace nn {

class MultiHeadAttention : public Layer {
 public:
  struct Cache {
    Tensor k;  // (N, H, T_max, Dh)
    Tensor v;  // (N, H, T_max, Dh)
    int t = 0;  // next write index
  };

  // head_dim 0 and num_kv_heads 0 keep Q, K, and V square at d_model.
  MultiHeadAttention(Context& ctx, int d_model, int num_heads, bool causal, bool use_bias,
                     bool use_rope, std::mt19937& rng, int head_dim = 0, int num_kv_heads = 0,
                     bool qk_norm = false, float rope_theta = 10000.f, float rms_eps = 1e-5f);
  const Tensor& forward(const Tensor& input) override;  // (N,T,D)
  const Tensor& forward_cached(const Tensor& x_t, Cache& cache);
  void reset_cache(Cache& cache, int batch, int t_max) const;
  const Tensor& backward(const Tensor& grad_output) override;
  std::vector<NamedParam> parameters() const override;
  void zero_grad() override;

 private:
  void lay_qkv(const Tensor& q, const Tensor& k, const Tensor& v, int time);
  Context* ctx_;
  int d_model_;
  int num_heads_;
  int num_kv_heads_;
  int head_dim_;
  int q_dim_;
  int kv_dim_;
  bool causal_;
  bool use_rope_ = false;
  bool qk_norm_ = false;
  float rope_theta_ = 10000.f;
  std::unique_ptr<Linear> q_;
  std::unique_ptr<Linear> k_;
  std::unique_ptr<Linear> v_;
  std::unique_ptr<Linear> out_;
  std::unique_ptr<RMSNorm> q_norm_;
  std::unique_ptr<RMSNorm> k_norm_;
  std::optional<Tensor> q_heads_;
  std::optional<Tensor> k_heads_;
  std::optional<Tensor> v_heads_;
  std::optional<Tensor> k_kv_;
  std::optional<Tensor> v_kv_;
  std::optional<Tensor> q_flat_grad_;
  std::optional<Tensor> kv_flat_grad_;
  std::optional<Tensor> scores_;
  std::optional<Tensor> attn_;
  std::optional<Tensor> context_;
  std::optional<Tensor> merged_;
  std::optional<Tensor> output_;
  std::optional<Tensor> d_context_;
  std::optional<Tensor> d_attn_;
  std::optional<Tensor> d_scores_;
  std::optional<Tensor> d_q_;
  std::optional<Tensor> d_k_;
  std::optional<Tensor> d_v_;
  std::optional<Tensor> merged_grad_;
  std::optional<Tensor> grad_pair_;
  std::optional<Tensor> grad_input_;
  bool cached_forward_ = false;
};

}  // namespace nn
