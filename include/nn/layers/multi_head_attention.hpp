#pragma once

#include "nn/layer.hpp"
#include "nn/layers/linear.hpp"

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

  MultiHeadAttention(Context& ctx, int d_model, int num_heads, bool causal, bool use_bias,
                     bool use_rope, std::mt19937& rng);
  const Tensor& forward(const Tensor& input) override;  // (N,T,D)
  const Tensor& forward_cached(const Tensor& x_t, Cache& cache);
  void reset_cache(Cache& cache, int batch, int t_max) const;
  const Tensor& backward(const Tensor& grad_output) override;
  std::vector<NamedParam> parameters() const override;
  void zero_grad() override;

 private:
  Context* ctx_;
  int d_model_;
  int num_heads_;
  int head_dim_;
  bool causal_;
  bool use_rope_ = false;
  std::unique_ptr<Linear> q_;
  std::unique_ptr<Linear> k_;
  std::unique_ptr<Linear> v_;
  std::unique_ptr<Linear> out_;
  std::optional<Tensor> q_heads_;
  std::optional<Tensor> k_heads_;
  std::optional<Tensor> v_heads_;
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
