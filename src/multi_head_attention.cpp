#include "nn/layers/multi_head_attention.hpp"

#include "ensure.hpp"
#include "nn/error.hpp"
#include "ocl.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace nn {
namespace {

void append_params(std::vector<NamedParam>& out, const Layer& layer, const char* prefix) {
  for (NamedParam param : layer.parameters()) {
    param.name = std::string(prefix) + param.name;
    out.push_back(param);
  }
}

Context* attention_context(Context& ctx, int d_model, int num_heads, bool use_rope) {
  if (d_model < 1 || num_heads < 1 || d_model % num_heads != 0) {
    throw Error("MultiHeadAttention size");
  }
  if (use_rope && ((d_model / num_heads) % 2 != 0)) throw std::invalid_argument("use_rope");
  return &ctx;
}

// S[i,j] = scale * sum_d Q[i,d] * K[j,d]. Per-head stride is time * dh.
void gemm_scores(Context& ctx, const float* q, const float* k, float* scores, int time, int dh,
                 int batch, float scale) {
  const int64_t stride_qk = static_cast<int64_t>(time) * dh;
  const int64_t stride_s = static_cast<int64_t>(time) * time;
  gemm_nt(ctx, scores, q, k, time, time, dh, batch, time, dh, dh, stride_s, stride_qk, stride_qk,
          scale, 0.f);
}

// dQ[i,d] = scale * sum_j dS[i,j] * K[j,d].
void gemm_dq(Context& ctx, const float* k, const float* ds, float* dq, int time, int dh, int batch,
             float scale) {
  const int64_t stride_qk = static_cast<int64_t>(time) * dh;
  const int64_t stride_s = static_cast<int64_t>(time) * time;
  gemm_nn(ctx, dq, ds, k, time, dh, time, batch, dh, time, dh, stride_qk, stride_s, stride_qk, scale,
          0.f);
}

// dK[j,d] = scale * sum_i dS[i,j] * Q[i,d].
void gemm_dk(Context& ctx, const float* q, const float* ds, float* dk, int time, int dh, int batch,
             float scale) {
  const int64_t stride_qk = static_cast<int64_t>(time) * dh;
  const int64_t stride_s = static_cast<int64_t>(time) * time;
  gemm_tn(ctx, dk, ds, q, time, dh, time, batch, dh, time, dh, stride_qk, stride_s, stride_qk, scale,
          0.f);
}

}  // namespace

MultiHeadAttention::MultiHeadAttention(Context& ctx, int d_model, int num_heads, bool causal,
                                       bool use_bias, bool use_rope, std::mt19937& rng)
    : ctx_(attention_context(ctx, d_model, num_heads, use_rope)),
      d_model_(d_model),
      num_heads_(num_heads),
      head_dim_(d_model / num_heads),
      causal_(causal),
      use_rope_(use_rope),
      q_(std::make_unique<Linear>(*ctx_, d_model, d_model, use_bias, rng)),
      k_(std::make_unique<Linear>(*ctx_, d_model, d_model, use_bias, rng)),
      v_(std::make_unique<Linear>(*ctx_, d_model, d_model, use_bias, rng)),
      out_(std::make_unique<Linear>(*ctx_, d_model, d_model, use_bias, rng)) {}

std::vector<NamedParam> MultiHeadAttention::parameters() const {
  std::vector<NamedParam> params;
  append_params(params, *q_, "q.");
  append_params(params, *k_, "k.");
  append_params(params, *v_, "v.");
  append_params(params, *out_, "out.");
  return params;
}

void MultiHeadAttention::zero_grad() {
  q_->zero_grad();
  k_->zero_grad();
  v_->zero_grad();
  out_->zero_grad();
}

const Tensor& MultiHeadAttention::forward(const Tensor& input) {
  const std::vector<int>& shape = input.shape();
  if (shape.size() != 3 || shape[2] != d_model_) throw Error("MultiHeadAttention input shape");
  cached_forward_ = false;
  const int n = shape[0];
  const int time = shape[1];
  const int heads = num_heads_;
  const int dh = head_dim_;
  const int batch = n * heads;
  const std::vector<int> heads_shape = {n, heads, time, dh};
  const std::vector<int> score_shape = {n, heads, time, time};

  ensure_tensor(q_heads_, *ctx_, heads_shape);
  ensure_tensor(k_heads_, *ctx_, heads_shape);
  ensure_tensor(v_heads_, *ctx_, heads_shape);
  ensure_tensor(scores_, *ctx_, score_shape);
  ensure_tensor(attn_, *ctx_, score_shape);
  ensure_tensor(context_, *ctx_, heads_shape);
  ensure_tensor(merged_, *ctx_, shape);
  ensure_tensor(output_, *ctx_, shape);

  const int64_t hidden = input.numel();
  launch_split_heads(*ctx_, q_heads_->data(), q_->forward(input).data(), time, heads, dh, hidden);
  launch_split_heads(*ctx_, k_heads_->data(), k_->forward(input).data(), time, heads, dh, hidden);
  launch_split_heads(*ctx_, v_heads_->data(), v_->forward(input).data(), time, heads, dh, hidden);
  if (use_rope_) {
    launch_rope(*ctx_, q_heads_->data(), k_heads_->data(), time, dh, 0, hidden / 2, 0);
  }

  const float scale = 1.f / std::sqrt(static_cast<float>(dh));
  gemm_scores(*ctx_, q_heads_->data(), k_heads_->data(), scores_->data(), time, dh, batch, scale);
  if (causal_) launch_causal_mask(*ctx_, scores_->data(), time, scores_->numel());
  launch_softmax_rows(*ctx_, attn_->data(), scores_->data(), batch * time, time);
  gemm_dq(*ctx_, v_heads_->data(), attn_->data(), context_->data(), time, dh, batch, 1.f);
  launch_merge_heads(*ctx_, merged_->data(), context_->data(), time, heads, dh, hidden);
  output_->copy_from(out_->forward(*merged_));
  return *output_;
}

void MultiHeadAttention::reset_cache(Cache& cache, int batch, int t_max) const {
  if (batch < 1) throw std::invalid_argument("batch");
  if (t_max < 1) throw std::invalid_argument("t_max");
  cache.k = Tensor(*ctx_, {batch, num_heads_, t_max, head_dim_});
  cache.v = Tensor(*ctx_, {batch, num_heads_, t_max, head_dim_});
  cache.k.zero();
  cache.v.zero();
  cache.t = 0;
}

const Tensor& MultiHeadAttention::forward_cached(const Tensor& x_t, Cache& cache) {
  if (&x_t.context() != ctx_ || x_t.rank() != 3 || x_t.shape()[1] != 1 || x_t.shape()[2] != d_model_) {
    throw std::invalid_argument("x_t");
  }
  const int n = x_t.shape()[0];
  const int heads = num_heads_;
  const int dh = head_dim_;
  const std::vector<int>& k_shape = cache.k.shape();
  if (&cache.k.context() != ctx_ || &cache.v.context() != ctx_ || cache.k.rank() != 4 ||
      cache.v.shape() != k_shape || k_shape[0] != n || k_shape[1] != heads || k_shape[3] != dh ||
      cache.t < 0 || cache.t >= k_shape[2]) {
    throw std::invalid_argument("cache");
  }
  cached_forward_ = true;

  const int t = cache.t;
  const int t_max = k_shape[2];
  const int tk = t + 1;
  constexpr int tq = 1;
  const int batch = n * heads;
  const std::vector<int> q_shape = {n, heads, tq, dh};
  const std::vector<int> score_shape = {n, heads, tq, tk};
  const std::vector<int> flat_shape = x_t.shape();

  ensure_tensor(q_heads_, *ctx_, q_shape);
  ensure_tensor(k_heads_, *ctx_, q_shape);
  ensure_tensor(v_heads_, *ctx_, q_shape);
  ensure_tensor(scores_, *ctx_, score_shape);
  ensure_tensor(attn_, *ctx_, score_shape);
  ensure_tensor(context_, *ctx_, q_shape);
  ensure_tensor(merged_, *ctx_, flat_shape);
  ensure_tensor(output_, *ctx_, flat_shape);

  const int64_t hidden = x_t.numel();
  launch_split_heads(*ctx_, q_heads_->data(), q_->forward(x_t).data(), tq, heads, dh, hidden);
  launch_split_heads(*ctx_, k_heads_->data(), k_->forward(x_t).data(), tq, heads, dh, hidden);
  launch_split_heads(*ctx_, v_heads_->data(), v_->forward(x_t).data(), tq, heads, dh, hidden);
  if (use_rope_) {
    launch_rope(*ctx_, q_heads_->data(), k_heads_->data(), tq, dh, t, hidden / 2, 0);
  }
  const int64_t kv_n = static_cast<int64_t>(batch) * dh;
  launch_write_kv(*ctx_, cache.k.data(), k_heads_->data(), t, t_max, dh, kv_n);
  launch_write_kv(*ctx_, cache.v.data(), v_heads_->data(), t, t_max, dh, kv_n);

  const float scale = 1.f / std::sqrt(static_cast<float>(dh));
  const int64_t stride_k = static_cast<int64_t>(t_max) * dh;
  const int64_t stride_q = static_cast<int64_t>(tq) * dh;
  const int64_t stride_s = static_cast<int64_t>(tq) * tk;
  gemm_nt(*ctx_, scores_->data(), q_heads_->data(), cache.k.data(), tq, tk, dh, batch, tk, dh, dh,
          stride_s, stride_q, stride_k, scale, 0.f);
  launch_softmax_rows(*ctx_, attn_->data(), scores_->data(), batch, tk);
  gemm_nn(*ctx_, context_->data(), attn_->data(), cache.v.data(), tq, dh, tk, batch, dh, tk, dh,
          stride_q, stride_s, stride_k, 1.f, 0.f);
  launch_merge_heads(*ctx_, merged_->data(), context_->data(), tq, heads, dh, hidden);
  output_->copy_from(out_->forward(*merged_));
  cache.t = tk;
  return *output_;
}

const Tensor& MultiHeadAttention::backward(const Tensor& grad_output) {
  if (cached_forward_) throw std::invalid_argument("forward_cached");
  if (!output_ || !attn_ || !q_heads_ || !k_heads_ || !v_heads_ ||
      grad_output.shape() != output_->shape()) {
    throw Error("MultiHeadAttention backward shape");
  }
  const int n = grad_output.shape()[0];
  const int time = grad_output.shape()[1];
  const int heads = num_heads_;
  const int dh = head_dim_;
  const int batch = n * heads;
  const std::vector<int> heads_shape = {n, heads, time, dh};
  const std::vector<int> score_shape = {n, heads, time, time};
  const std::vector<int>& flat_shape = grad_output.shape();

  ensure_tensor(d_context_, *ctx_, heads_shape);
  ensure_tensor(d_attn_, *ctx_, score_shape);
  ensure_tensor(d_scores_, *ctx_, score_shape);
  ensure_tensor(d_q_, *ctx_, heads_shape);
  ensure_tensor(d_k_, *ctx_, heads_shape);
  ensure_tensor(d_v_, *ctx_, heads_shape);
  ensure_tensor(merged_grad_, *ctx_, flat_shape);
  ensure_tensor(grad_pair_, *ctx_, flat_shape);
  ensure_tensor(grad_input_, *ctx_, flat_shape);

  const int64_t hidden = grad_output.numel();
  const float scale = 1.f / std::sqrt(static_cast<float>(dh));

  const Tensor& d_merged = out_->backward(grad_output);
  launch_split_heads(*ctx_, d_context_->data(), d_merged.data(), time, heads, dh, hidden);
  gemm_scores(*ctx_, d_context_->data(), v_heads_->data(), d_attn_->data(), time, dh, batch, 1.f);
  gemm_dk(*ctx_, d_context_->data(), attn_->data(), d_v_->data(), time, dh, batch, 1.f);
  launch_softmax_backward(*ctx_, d_scores_->data(), attn_->data(), d_attn_->data(), batch * time, time,
                          causal_ ? 1 : 0);
  gemm_dq(*ctx_, k_heads_->data(), d_scores_->data(), d_q_->data(), time, dh, batch, scale);
  gemm_dk(*ctx_, q_heads_->data(), d_scores_->data(), d_k_->data(), time, dh, batch, scale);
  if (use_rope_) {
    launch_rope(*ctx_, d_q_->data(), d_k_->data(), time, dh, 0, hidden / 2, 1);
  }

  launch_merge_heads(*ctx_, merged_grad_->data(), d_q_->data(), time, heads, dh, hidden);
  const Tensor& gq = q_->backward(*merged_grad_);
  launch_merge_heads(*ctx_, merged_grad_->data(), d_k_->data(), time, heads, dh, hidden);
  const Tensor& gk = k_->backward(*merged_grad_);
  launch_merge_heads(*ctx_, merged_grad_->data(), d_v_->data(), time, heads, dh, hidden);
  const Tensor& gv = v_->backward(*merged_grad_);

  launch_add2(*ctx_, grad_pair_->data(), gq.data(), gk.data(), hidden);
  launch_add2(*ctx_, grad_input_->data(), grad_pair_->data(), gv.data(), hidden);
  return *grad_input_;
}

}  // namespace nn
