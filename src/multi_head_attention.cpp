#include "nn/layers/multi_head_attention.hpp"

#include "activate.hpp"
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

void gemm_scores_bf16(Context& ctx, const uint16_t* q, const uint16_t* k, uint16_t* scores, int time,
                      int dh, int batch, float scale) {
  const int64_t stride_qk = static_cast<int64_t>(time) * dh;
  const int64_t stride_s = static_cast<int64_t>(time) * time;
  gemm_nt_bf16(ctx, scores, true, q, k, time, time, dh, batch, time, dh, dh, stride_s, stride_qk,
               stride_qk, scale, 0.f);
}

void gemm_dq_bf16(Context& ctx, const uint16_t* k, const uint16_t* ds, uint16_t* dq, int time, int dh,
                  int batch, float scale) {
  const int64_t stride_qk = static_cast<int64_t>(time) * dh;
  const int64_t stride_s = static_cast<int64_t>(time) * time;
  gemm_nn_bf16(ctx, dq, true, ds, k, time, dh, time, batch, dh, time, dh, stride_qk, stride_s,
               stride_qk, scale, 0.f);
}

void gemm_dk_bf16(Context& ctx, const uint16_t* q, const uint16_t* ds, uint16_t* dk, int time, int dh,
                  int batch, float scale) {
  const int64_t stride_qk = static_cast<int64_t>(time) * dh;
  const int64_t stride_s = static_cast<int64_t>(time) * time;
  gemm_tn_bf16(ctx, dk, true, ds, q, time, dh, time, batch, dh, time, dh, stride_qk, stride_s,
               stride_qk, scale, 0.f);
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
  if (q_heads_->dtype() == DType::BFloat16) {
    if (group == 1) {
      arena_reset(*ctx_);
      float* qf = stage_f32(*ctx_, q);
      float* kf = stage_f32(*ctx_, k);
      float* vf = stage_f32(*ctx_, v);
      float* qh = arena_f32(*ctx_, q_heads_->numel());
      float* kh = arena_f32(*ctx_, k_heads_->numel());
      float* vh = arena_f32(*ctx_, v_heads_->numel());
      launch_split_heads(*ctx_, qh, qf, time, heads, dh, q_n);
      launch_split_heads(*ctx_, kh, kf, time, kv, dh, kv_n);
      launch_split_heads(*ctx_, vh, vf, time, kv, dh, kv_n);
      keep_f32(*ctx_, *q_heads_, qh);
      keep_f32(*ctx_, *k_heads_, kh);
      keep_f32(*ctx_, *v_heads_, vh);
      if (q_norm_) q_heads_->copy_from(q_norm_->forward(*q_heads_));
      if (k_norm_) k_heads_->copy_from(k_norm_->forward(*k_heads_));
      return;
    }
    ensure_activation(k_kv_, *ctx_, {n, kv, time, dh});
    ensure_activation(v_kv_, *ctx_, {n, kv, time, dh});
    arena_reset(*ctx_);
    float* qf = stage_f32(*ctx_, q);
    float* kf = stage_f32(*ctx_, k);
    float* vf = stage_f32(*ctx_, v);
    float* qh = arena_f32(*ctx_, q_heads_->numel());
    float* kkv = arena_f32(*ctx_, k_kv_->numel());
    float* vkv = arena_f32(*ctx_, v_kv_->numel());
    launch_split_heads(*ctx_, qh, qf, time, heads, dh, q_n);
    launch_split_heads(*ctx_, kkv, kf, time, kv, dh, kv_n);
    launch_split_heads(*ctx_, vkv, vf, time, kv, dh, kv_n);
    keep_f32(*ctx_, *q_heads_, qh);
    keep_f32(*ctx_, *k_kv_, kkv);
    keep_f32(*ctx_, *v_kv_, vkv);
    if (q_norm_) q_heads_->copy_from(q_norm_->forward(*q_heads_));
    if (k_norm_) k_kv_->copy_from(k_norm_->forward(*k_kv_));
    arena_reset(*ctx_);
    float* ks = stage_f32(*ctx_, *k_kv_);
    float* vs = stage_f32(*ctx_, *v_kv_);
    float* kh = arena_f32(*ctx_, k_heads_->numel());
    float* vh = arena_f32(*ctx_, v_heads_->numel());
    launch_repeat_kv(*ctx_, kh, ks, time, kv, group, dh, full_n);
    launch_repeat_kv(*ctx_, vh, vs, time, kv, group, dh, full_n);
    keep_f32(*ctx_, *k_heads_, kh);
    keep_f32(*ctx_, *v_heads_, vh);
    return;
  }
  launch_split_heads(*ctx_, q_heads_->data(), q.data(), time, heads, dh, q_n);
  if (q_norm_) q_heads_->copy_from(q_norm_->forward(*q_heads_));
  if (group == 1) {
    launch_split_heads(*ctx_, k_heads_->data(), k.data(), time, kv, dh, kv_n);
    launch_split_heads(*ctx_, v_heads_->data(), v.data(), time, kv, dh, kv_n);
    if (k_norm_) k_heads_->copy_from(k_norm_->forward(*k_heads_));
  } else {
    ensure_activation(k_kv_, *ctx_, {n, kv, time, dh});
    ensure_activation(v_kv_, *ctx_, {n, kv, time, dh});
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

  ensure_activation(q_heads_, *ctx_, heads_shape);
  ensure_activation(k_heads_, *ctx_, heads_shape);
  ensure_activation(v_heads_, *ctx_, heads_shape);
  ensure_activation(scores_, *ctx_, score_shape);
  ensure_activation(attn_, *ctx_, score_shape);
  ensure_activation(context_, *ctx_, heads_shape);
  ensure_activation(merged_, *ctx_, merged_shape);
  ensure_activation(output_, *ctx_, shape);

  lay_qkv(q_->forward(input), k_->forward(input), v_->forward(input), time);
  const float scale = 1.f / std::sqrt(static_cast<float>(dh));
  if (output_->dtype() == DType::BFloat16) {
    if (use_rope_) {
      arena_reset(*ctx_);
      float* q = stage_f32(*ctx_, *q_heads_);
      float* k = stage_f32(*ctx_, *k_heads_);
      launch_rope(*ctx_, q, k, time, dh, 0, full_n / 2, 0, rope_theta_);
      keep_f32(*ctx_, *q_heads_, q);
      keep_f32(*ctx_, *k_heads_, k);
    }
    gemm_scores_bf16(*ctx_, q_heads_->bf16_data(), k_heads_->bf16_data(), scores_->bf16_data(), time,
                     dh, batch, scale);
    arena_reset(*ctx_);
    float* scores = stage_f32(*ctx_, *scores_);
    if (causal_) launch_causal_mask(*ctx_, scores, time, scores_->numel());
    float* attn = arena_f32(*ctx_, attn_->numel());
    launch_softmax_rows(*ctx_, attn, scores, batch * time, time);
    if (causal_) keep_f32(*ctx_, *scores_, scores);
    keep_f32(*ctx_, *attn_, attn);
    gemm_dq_bf16(*ctx_, v_heads_->bf16_data(), attn_->bf16_data(), context_->bf16_data(), time, dh,
                 batch, 1.f);
    arena_reset(*ctx_);
    float* ctx_h = stage_f32(*ctx_, *context_);
    float* merged = arena_f32(*ctx_, merged_->numel());
    launch_merge_heads(*ctx_, merged, ctx_h, time, heads, dh, full_n);
    keep_f32(*ctx_, *merged_, merged);
    output_->copy_from(out_->forward(*merged_));
    return *output_;
  }
  if (use_rope_) {
    launch_rope(*ctx_, q_heads_->data(), k_heads_->data(), time, dh, 0, full_n / 2, 0, rope_theta_);
  }
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
  cache.k = Tensor(*ctx_, {batch, num_heads_, t_max, head_dim_}, ctx_->activation_dtype());
  cache.v = Tensor(*ctx_, {batch, num_heads_, t_max, head_dim_}, ctx_->activation_dtype());
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

  ensure_activation(q_heads_, *ctx_, q_shape);
  ensure_activation(k_heads_, *ctx_, q_shape);
  ensure_activation(v_heads_, *ctx_, q_shape);
  ensure_activation(scores_, *ctx_, score_shape);
  ensure_activation(attn_, *ctx_, score_shape);
  ensure_activation(context_, *ctx_, q_shape);
  ensure_activation(merged_, *ctx_, merged_shape);
  ensure_activation(output_, *ctx_, flat_shape);

  lay_qkv(q_->forward(x_t), k_->forward(x_t), v_->forward(x_t), tq);
  const int64_t kv_n = static_cast<int64_t>(batch) * dh;
  const float scale = 1.f / std::sqrt(static_cast<float>(dh));
  const int64_t stride_k = static_cast<int64_t>(t_max) * dh;
  const int64_t stride_q = static_cast<int64_t>(tq) * dh;
  const int64_t stride_s = static_cast<int64_t>(tq) * tk;
  if (output_->dtype() == DType::BFloat16) {
    if (use_rope_) {
      arena_reset(*ctx_);
      float* q = stage_f32(*ctx_, *q_heads_);
      float* k = stage_f32(*ctx_, *k_heads_);
      launch_rope(*ctx_, q, k, tq, dh, t, full_n / 2, 0, rope_theta_);
      keep_f32(*ctx_, *q_heads_, q);
      keep_f32(*ctx_, *k_heads_, k);
    }
    if (cache.k.dtype() == DType::BFloat16) {
      launch_write_kv_u16(*ctx_, cache.k.bf16_data(), k_heads_->bf16_data(), t, t_max, dh, kv_n);
      launch_write_kv_u16(*ctx_, cache.v.bf16_data(), v_heads_->bf16_data(), t, t_max, dh, kv_n);
    } else {
      arena_reset(*ctx_);
      float* k = stage_f32(*ctx_, *k_heads_);
      float* v = stage_f32(*ctx_, *v_heads_);
      launch_write_kv(*ctx_, cache.k.data(), k, t, t_max, dh, kv_n);
      launch_write_kv(*ctx_, cache.v.data(), v, t, t_max, dh, kv_n);
    }
    gemm_nt_bf16(*ctx_, scores_->bf16_data(), true, stage_bf16(*ctx_, *q_heads_),
                 stage_bf16(*ctx_, cache.k), tq, tk, dh, batch, tk, dh, dh, stride_s, stride_q,
                 stride_k, scale, 0.f);
    arena_reset(*ctx_);
    float* scores = stage_f32(*ctx_, *scores_);
    float* attn = arena_f32(*ctx_, attn_->numel());
    launch_softmax_rows(*ctx_, attn, scores, batch, tk);
    keep_f32(*ctx_, *attn_, attn);
    gemm_nn_bf16(*ctx_, context_->bf16_data(), true, stage_bf16(*ctx_, *attn_),
                 stage_bf16(*ctx_, cache.v), tq, dh, tk, batch, dh, tk, dh, stride_q, stride_s,
                 stride_k, 1.f, 0.f);
    arena_reset(*ctx_);
    float* ctx_h = stage_f32(*ctx_, *context_);
    float* merged = arena_f32(*ctx_, merged_->numel());
    launch_merge_heads(*ctx_, merged, ctx_h, tq, heads, dh, full_n);
    keep_f32(*ctx_, *merged_, merged);
    output_->copy_from(out_->forward(*merged_));
    cache.t = tk;
    return *output_;
  }
  if (use_rope_) {
    launch_rope(*ctx_, q_heads_->data(), k_heads_->data(), tq, dh, t, full_n / 2, 0, rope_theta_);
  }
  launch_write_kv(*ctx_, cache.k.data(), k_heads_->data(), t, t_max, dh, kv_n);
  launch_write_kv(*ctx_, cache.v.data(), v_heads_->data(), t, t_max, dh, kv_n);
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

  ensure_activation(d_context_, *ctx_, heads_shape);
  ensure_activation(d_attn_, *ctx_, score_shape);
  ensure_activation(d_scores_, *ctx_, score_shape);
  ensure_activation(d_q_, *ctx_, heads_shape);
  ensure_activation(d_k_, *ctx_, heads_shape);
  ensure_activation(d_v_, *ctx_, heads_shape);
  ensure_activation(q_flat_grad_, *ctx_, {n, time, q_dim_});
  ensure_activation(kv_flat_grad_, *ctx_, {n, time, kv_dim_});
  ensure_activation(grad_pair_, *ctx_, flat_shape);
  ensure_activation(grad_input_, *ctx_, flat_shape);

  const int64_t hidden = grad_output.numel();
  const float scale = 1.f / std::sqrt(static_cast<float>(dh));

  if (grad_input_->dtype() == DType::BFloat16) {
    const Tensor& d_merged = out_->backward(grad_output);
    arena_reset(*ctx_);
    float* merged_f = stage_f32(*ctx_, d_merged);
    float* d_context = arena_f32(*ctx_, d_context_->numel());
    launch_split_heads(*ctx_, d_context, merged_f, time, heads, dh, q_n);
    keep_f32(*ctx_, *d_context_, d_context);
    gemm_scores_bf16(*ctx_, d_context_->bf16_data(), v_heads_->bf16_data(), d_attn_->bf16_data(), time,
                     dh, batch, 1.f);
    gemm_dk_bf16(*ctx_, d_context_->bf16_data(), attn_->bf16_data(), d_v_->bf16_data(), time, dh,
                 batch, 1.f);
    arena_reset(*ctx_);
    float* probs = stage_f32(*ctx_, *attn_);
    float* dp = stage_f32(*ctx_, *d_attn_);
    float* ds = arena_f32(*ctx_, d_scores_->numel());
    launch_softmax_backward(*ctx_, ds, probs, dp, batch * time, time, causal_ ? 1 : 0);
    keep_f32(*ctx_, *d_scores_, ds);
    gemm_dq_bf16(*ctx_, k_heads_->bf16_data(), d_scores_->bf16_data(), d_q_->bf16_data(), time, dh,
                 batch, scale);
    gemm_dk_bf16(*ctx_, q_heads_->bf16_data(), d_scores_->bf16_data(), d_k_->bf16_data(), time, dh,
                 batch, scale);
    if (use_rope_) {
      arena_reset(*ctx_);
      float* dq = stage_f32(*ctx_, *d_q_);
      float* dk = stage_f32(*ctx_, *d_k_);
      launch_rope(*ctx_, dq, dk, time, dh, 0, full_n / 2, 1, rope_theta_);
      keep_f32(*ctx_, *d_q_, dq);
      keep_f32(*ctx_, *d_k_, dk);
    }

    const Tensor* dq_t = &*d_q_;
    if (q_norm_) dq_t = &q_norm_->backward(*d_q_);
    arena_reset(*ctx_);
    float* dq_f = stage_f32(*ctx_, *dq_t);
    float* q_flat = arena_f32(*ctx_, q_flat_grad_->numel());
    launch_merge_heads(*ctx_, q_flat, dq_f, time, heads, dh, q_n);
    keep_f32(*ctx_, *q_flat_grad_, q_flat);
    const Tensor& gq = q_->backward(*q_flat_grad_);

    const Tensor* dk_t = &*d_k_;
    if (group > 1) {
      ensure_activation(k_kv_, *ctx_, {n, kv, time, dh});
      const int64_t kv_heads_n = static_cast<int64_t>(n) * kv * time * dh;
      arena_reset(*ctx_);
      float* src = stage_f32(*ctx_, *d_k_);
      float* dst = arena_f32(*ctx_, k_kv_->numel());
      launch_sum_kv(*ctx_, dst, src, time, kv, group, dh, kv_heads_n);
      keep_f32(*ctx_, *k_kv_, dst);
      dk_t = &*k_kv_;
      if (k_norm_) dk_t = &k_norm_->backward(*k_kv_);
    } else if (k_norm_) {
      dk_t = &k_norm_->backward(*d_k_);
    }
    arena_reset(*ctx_);
    float* dk_f = stage_f32(*ctx_, *dk_t);
    float* kv_flat = arena_f32(*ctx_, kv_flat_grad_->numel());
    launch_merge_heads(*ctx_, kv_flat, dk_f, time, kv, dh, kv_n);
    keep_f32(*ctx_, *kv_flat_grad_, kv_flat);
    const Tensor& gk = k_->backward(*kv_flat_grad_);

    const Tensor* dv_t = &*d_v_;
    if (group > 1) {
      ensure_activation(v_kv_, *ctx_, {n, kv, time, dh});
      const int64_t kv_heads_n = static_cast<int64_t>(n) * kv * time * dh;
      arena_reset(*ctx_);
      float* src = stage_f32(*ctx_, *d_v_);
      float* dst = arena_f32(*ctx_, v_kv_->numel());
      launch_sum_kv(*ctx_, dst, src, time, kv, group, dh, kv_heads_n);
      keep_f32(*ctx_, *v_kv_, dst);
      dv_t = &*v_kv_;
    }
    arena_reset(*ctx_);
    float* dv_f = stage_f32(*ctx_, *dv_t);
    float* v_flat = arena_f32(*ctx_, kv_flat_grad_->numel());
    launch_merge_heads(*ctx_, v_flat, dv_f, time, kv, dh, kv_n);
    keep_f32(*ctx_, *kv_flat_grad_, v_flat);
    const Tensor& gv = v_->backward(*kv_flat_grad_);

    arena_reset(*ctx_);
    float* aq = stage_f32(*ctx_, gq);
    float* ak = stage_f32(*ctx_, gk);
    float* pair = arena_f32(*ctx_, hidden);
    launch_add2(*ctx_, pair, aq, ak, hidden);
    float* av = stage_f32(*ctx_, gv);
    float* sum = arena_f32(*ctx_, hidden);
    launch_add2(*ctx_, sum, pair, av, hidden);
    keep_f32(*ctx_, *grad_pair_, pair);
    keep_f32(*ctx_, *grad_input_, sum);
    return *grad_input_;
  }

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
    ensure_activation(k_kv_, *ctx_, {n, kv, time, dh});
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
    ensure_activation(v_kv_, *ctx_, {n, kv, time, dh});
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
