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

Context* attention_context(Context& ctx, int d_model, int num_heads, int head_dim, int num_kv_heads,
                          bool use_rope, float rope_theta, float rms_eps) {
  if (d_model < 1 || num_heads < 1) throw Error("MultiHeadAttention size");
  const int dh = head_dim == 0 ? d_model / num_heads : head_dim;
  const int kv = num_kv_heads == 0 ? num_heads : num_kv_heads;
  if (head_dim == 0 && d_model % num_heads != 0) throw Error("MultiHeadAttention size");
  if (dh < 1 || kv < 1 || kv > num_heads || num_heads % kv != 0) throw Error("MultiHeadAttention size");
  if (dh > 0x7fffffff / num_heads || dh > 0x7fffffff / kv) throw Error("MultiHeadAttention size");
  if (use_rope && dh % 2 != 0) throw std::invalid_argument("use_rope");
  if (!(rope_theta > 0.f) || !std::isfinite(rope_theta)) throw std::invalid_argument("rope_theta");
  if (!(rms_eps > 0.f) || !std::isfinite(rms_eps)) throw std::invalid_argument("rms_eps");
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
                                       bool use_bias, bool use_rope, std::mt19937& rng, int head_dim,
                                       int num_kv_heads, bool qk_norm, float rope_theta, float rms_eps)
    : ctx_(attention_context(ctx, d_model, num_heads, head_dim, num_kv_heads, use_rope, rope_theta,
                             rms_eps)),
      d_model_(d_model),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads == 0 ? num_heads : num_kv_heads),
      head_dim_(head_dim == 0 ? d_model / num_heads : head_dim),
      q_dim_(num_heads * (head_dim == 0 ? d_model / num_heads : head_dim)),
      kv_dim_((num_kv_heads == 0 ? num_heads : num_kv_heads) *
              (head_dim == 0 ? d_model / num_heads : head_dim)),
      causal_(causal),
      use_rope_(use_rope),
      qk_norm_(qk_norm),
      rope_theta_(rope_theta),
      q_(std::make_unique<Linear>(*ctx_, d_model, num_heads * (head_dim == 0 ? d_model / num_heads : head_dim),
                                  use_bias, rng)),
      k_(std::make_unique<Linear>(
          *ctx_, d_model,
          (num_kv_heads == 0 ? num_heads : num_kv_heads) * (head_dim == 0 ? d_model / num_heads : head_dim),
          use_bias, rng)),
      v_(std::make_unique<Linear>(
          *ctx_, d_model,
          (num_kv_heads == 0 ? num_heads : num_kv_heads) * (head_dim == 0 ? d_model / num_heads : head_dim),
          use_bias, rng)),
      out_(std::make_unique<Linear>(*ctx_, num_heads * (head_dim == 0 ? d_model / num_heads : head_dim), d_model,
                                    use_bias, rng)) {
  if (qk_norm) {
    q_norm_ = std::make_unique<RMSNorm>(*ctx_, head_dim_, rms_eps);
    k_norm_ = std::make_unique<RMSNorm>(*ctx_, head_dim_, rms_eps);
  }
}

std::vector<NamedParam> MultiHeadAttention::parameters() const {
  std::vector<NamedParam> params;
  append_params(params, *q_, "q.");
  append_params(params, *k_, "k.");
  append_params(params, *v_, "v.");
  append_params(params, *out_, "out.");
  if (q_norm_) append_params(params, *q_norm_, "q_norm.");
  if (k_norm_) append_params(params, *k_norm_, "k_norm.");
  return params;
}

void MultiHeadAttention::zero_grad() {
  q_->zero_grad();
  k_->zero_grad();
  v_->zero_grad();
  out_->zero_grad();
  if (q_norm_) q_norm_->zero_grad();
  if (k_norm_) k_norm_->zero_grad();
}

void MultiHeadAttention::lay_qkv(const Tensor& q, const Tensor& k, const Tensor& v, int time) {
  const int n = q.shape()[0];
  const int heads = num_heads_;
  const int kv = num_kv_heads_;
  const int group = heads / kv;
  const int dh = head_dim_;
  const int64_t q_n = static_cast<int64_t>(n) * time * q_dim_;
  const int64_t kv_n = static_cast<int64_t>(n) * time * kv_dim_;
  const int64_t full_n = static_cast<int64_t>(n) * heads * time * dh;
  launch_split_heads(*ctx_, q_heads_->data(), q.data(), time, heads, dh, q_n);
  if (q_norm_) q_heads_->copy_from(q_norm_->forward(*q_heads_));
  if (group == 1) {
    launch_split_heads(*ctx_, k_heads_->data(), k.data(), time, kv, dh, kv_n);
    launch_split_heads(*ctx_, v_heads_->data(), v.data(), time, kv, dh, kv_n);
    if (k_norm_) k_heads_->copy_from(k_norm_->forward(*k_heads_));
  } else {
    ensure_tensor(k_kv_, *ctx_, {n, kv, time, dh});
    ensure_tensor(v_kv_, *ctx_, {n, kv, time, dh});
    launch_split_heads(*ctx_, k_kv_->data(), k.data(), time, kv, dh, kv_n);
    launch_split_heads(*ctx_, v_kv_->data(), v.data(), time, kv, dh, kv_n);
    if (k_norm_) k_kv_->copy_from(k_norm_->forward(*k_kv_));
    launch_repeat_kv(*ctx_, k_heads_->data(), k_kv_->data(), time, kv, group, dh, full_n);
    launch_repeat_kv(*ctx_, v_heads_->data(), v_kv_->data(), time, kv, group, dh, full_n);
  }
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
  const std::vector<int> merged_shape = {n, time, q_dim_};
  const int64_t full_n = static_cast<int64_t>(n) * heads * time * dh;

  ensure_tensor(q_heads_, *ctx_, heads_shape);
  ensure_tensor(k_heads_, *ctx_, heads_shape);
  ensure_tensor(v_heads_, *ctx_, heads_shape);
  ensure_tensor(scores_, *ctx_, score_shape);
  ensure_tensor(attn_, *ctx_, score_shape);
  ensure_tensor(context_, *ctx_, heads_shape);
  ensure_tensor(merged_, *ctx_, merged_shape);
  ensure_tensor(output_, *ctx_, shape);

  lay_qkv(q_->forward(input), k_->forward(input), v_->forward(input), time);
  if (use_rope_) {
    launch_rope(*ctx_, q_heads_->data(), k_heads_->data(), time, dh, 0, full_n / 2, 0, rope_theta_);
  }

  const float scale = 1.f / std::sqrt(static_cast<float>(dh));
  gemm_scores(*ctx_, q_heads_->data(), k_heads_->data(), scores_->data(), time, dh, batch, scale);
  if (causal_) launch_causal_mask(*ctx_, scores_->data(), time, scores_->numel());
  launch_softmax_rows(*ctx_, attn_->data(), scores_->data(), batch * time, time);
  gemm_dq(*ctx_, v_heads_->data(), attn_->data(), context_->data(), time, dh, batch, 1.f);
  launch_merge_heads(*ctx_, merged_->data(), context_->data(), time, heads, dh, full_n);
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
  const int64_t full_n = static_cast<int64_t>(n) * heads * dh;
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
  const std::vector<int> merged_shape = {n, tq, q_dim_};

  ensure_tensor(q_heads_, *ctx_, q_shape);
  ensure_tensor(k_heads_, *ctx_, q_shape);
  ensure_tensor(v_heads_, *ctx_, q_shape);
  ensure_tensor(scores_, *ctx_, score_shape);
  ensure_tensor(attn_, *ctx_, score_shape);
  ensure_tensor(context_, *ctx_, q_shape);
  ensure_tensor(merged_, *ctx_, merged_shape);
  ensure_tensor(output_, *ctx_, flat_shape);

  lay_qkv(q_->forward(x_t), k_->forward(x_t), v_->forward(x_t), tq);
  if (use_rope_) {
    launch_rope(*ctx_, q_heads_->data(), k_heads_->data(), tq, dh, t, full_n / 2, 0, rope_theta_);
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
  launch_merge_heads(*ctx_, merged_->data(), context_->data(), tq, heads, dh, full_n);
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
  const int kv = num_kv_heads_;
  const int group = heads / kv;
  const int dh = head_dim_;
  const int batch = n * heads;
  const std::vector<int> heads_shape = {n, heads, time, dh};
  const std::vector<int> score_shape = {n, heads, time, time};
  const std::vector<int>& flat_shape = grad_output.shape();
  const int64_t full_n = static_cast<int64_t>(n) * heads * time * dh;
  const int64_t q_n = static_cast<int64_t>(n) * time * q_dim_;
  const int64_t kv_n = static_cast<int64_t>(n) * time * kv_dim_;

  ensure_tensor(d_context_, *ctx_, heads_shape);
  ensure_tensor(d_attn_, *ctx_, score_shape);
  ensure_tensor(d_scores_, *ctx_, score_shape);
  ensure_tensor(d_q_, *ctx_, heads_shape);
  ensure_tensor(d_k_, *ctx_, heads_shape);
  ensure_tensor(d_v_, *ctx_, heads_shape);
  ensure_tensor(q_flat_grad_, *ctx_, {n, time, q_dim_});
  ensure_tensor(kv_flat_grad_, *ctx_, {n, time, kv_dim_});
  ensure_tensor(grad_pair_, *ctx_, flat_shape);
  ensure_tensor(grad_input_, *ctx_, flat_shape);

  const int64_t hidden = grad_output.numel();
  const float scale = 1.f / std::sqrt(static_cast<float>(dh));

  const Tensor& d_merged = out_->backward(grad_output);
  launch_split_heads(*ctx_, d_context_->data(), d_merged.data(), time, heads, dh, q_n);
  gemm_scores(*ctx_, d_context_->data(), v_heads_->data(), d_attn_->data(), time, dh, batch, 1.f);
  gemm_dk(*ctx_, d_context_->data(), attn_->data(), d_v_->data(), time, dh, batch, 1.f);
  launch_softmax_backward(*ctx_, d_scores_->data(), attn_->data(), d_attn_->data(), batch * time, time,
                          causal_ ? 1 : 0);
  gemm_dq(*ctx_, k_heads_->data(), d_scores_->data(), d_q_->data(), time, dh, batch, scale);
  gemm_dk(*ctx_, q_heads_->data(), d_scores_->data(), d_k_->data(), time, dh, batch, scale);
  if (use_rope_) {
    launch_rope(*ctx_, d_q_->data(), d_k_->data(), time, dh, 0, full_n / 2, 1, rope_theta_);
  }

  const float* dq = d_q_->data();
  if (q_norm_) dq = q_norm_->backward(*d_q_).data();
  launch_merge_heads(*ctx_, q_flat_grad_->data(), dq, time, heads, dh, q_n);
  const Tensor& gq = q_->backward(*q_flat_grad_);

  const float* dk = d_k_->data();
  if (group > 1) {
    ensure_tensor(k_kv_, *ctx_, {n, kv, time, dh});
    const int64_t kv_heads_n = static_cast<int64_t>(n) * kv * time * dh;
    launch_sum_kv(*ctx_, k_kv_->data(), d_k_->data(), time, kv, group, dh, kv_heads_n);
    dk = k_kv_->data();
    if (k_norm_) dk = k_norm_->backward(*k_kv_).data();
  } else if (k_norm_) {
    dk = k_norm_->backward(*d_k_).data();
  }
  launch_merge_heads(*ctx_, kv_flat_grad_->data(), dk, time, kv, dh, kv_n);
  const Tensor& gk = k_->backward(*kv_flat_grad_);

  const float* dv = d_v_->data();
  if (group > 1) {
    ensure_tensor(v_kv_, *ctx_, {n, kv, time, dh});
    const int64_t kv_heads_n = static_cast<int64_t>(n) * kv * time * dh;
    launch_sum_kv(*ctx_, v_kv_->data(), d_v_->data(), time, kv, group, dh, kv_heads_n);
    dv = v_kv_->data();
  }
  launch_merge_heads(*ctx_, kv_flat_grad_->data(), dv, time, kv, dh, kv_n);
  const Tensor& gv = v_->backward(*kv_flat_grad_);

  launch_add2(*ctx_, grad_pair_->data(), gq.data(), gk.data(), hidden);
  launch_add2(*ctx_, grad_input_->data(), grad_pair_->data(), gv.data(), hidden);
  return *grad_input_;
}

}  // namespace nn
