#include "nn/transformer.hpp"

#include "ensure.hpp"
#include "nn/error.hpp"
#include "nn/layers/layer_norm.hpp"
#include "nn/layers/linear.hpp"
#include "nn/layers/rms_norm.hpp"
#include "nn/layers/transformer_block.hpp"
#include "ocl.hpp"

#include <random>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace nn {
namespace {

std::string join_name(const std::string& prefix, const std::string& name) {
  if (prefix.empty()) return name;
  return prefix + "." + name;
}

const TransformerConfig& transformer_config(const Network& network) {
  return std::get<TransformerConfig>(network.config());
}

}  // namespace

Transformer::Transformer(Context& ctx, const TransformerConfig& config, uint32_t seed) : ctx_(&ctx) {
  validate(config);
  std::mt19937 rng(seed);
  tok_emb_ = std::make_unique<Embedding>(ctx, config.vocab_size, config.d_model, rng);
  tok_emb_->set_name("tok_emb");
  if (config.positional == PositionalEncoding::LearnedAbsolute) {
    pos_emb_ = std::make_unique<Embedding>(ctx, config.max_seq_len, config.d_model, rng);
    pos_emb_->set_name("pos_emb");
  }
  TransformerBlockSpec block_spec;
  block_spec.d_model = config.d_model;
  block_spec.num_heads = config.num_heads;
  block_spec.d_ff = config.d_ff;
  block_spec.dropout = config.dropout;
  block_spec.bias = config.bias;
  block_spec.activation = config.activation;
  const bool rope = config.positional == PositionalEncoding::Rope;
  block_spec.use_rope = rope;
  block_spec.head_dim = config.head_dim;
  block_spec.num_kv_heads = config.num_kv_heads;
  block_spec.qk_norm = config.qk_norm;
  block_spec.rope_theta = config.rope_theta;
  block_spec.rms_eps = config.rms_eps;

  std::vector<std::unique_ptr<Layer>> layers;
  for (int i = 0; i < config.num_layers; ++i) {
    auto block = std::make_unique<TransformerBlock>(ctx, block_spec, rng);
    block->set_name("blocks." + std::to_string(i));
    block_ptrs_.push_back(block.get());
    layers.push_back(std::move(block));
  }
  std::unique_ptr<Layer> final_ln;
  if (rope) {
    final_ln = std::make_unique<RMSNorm>(ctx, config.d_model, config.rms_eps);
  } else {
    final_ln = std::make_unique<LayerNorm>(ctx, config.d_model);
  }
  final_ln->set_name("final_ln");
  final_ln_ptr_ = final_ln.get();
  layers.push_back(std::move(final_ln));
  auto lm_head = std::make_unique<Linear>(ctx, config.d_model, config.vocab_size, config.bias, rng);
  lm_head->set_name("lm_head");
  lm_head_ptr_ = lm_head.get();
  layers.push_back(std::move(lm_head));
  network_ = Network(ctx, Family::Transformer, std::move(layers), config);
}

const Tensor& Transformer::forward(const IndexTensor& tokens) {
  if (tokens.shape().size() != 2) throw std::invalid_argument("tokens");
  const int n = tokens.shape()[0];
  const int t = tokens.shape()[1];
  if (t > transformer_config(*network_).max_seq_len) throw std::invalid_argument("max_seq_len");

  const Tensor& tok = tok_emb_->forward_indices(tokens);
  const Tensor* x = &tok;
  if (pos_emb_) {
    if (!pos_ids_ || pos_ids_->shape()[0] != n || pos_ids_->shape()[1] != t) {
      pos_ids_.emplace(*ctx_, std::vector<int>{n, t});
      std::vector<int64_t> host(static_cast<size_t>(pos_ids_->numel()));
      for (int row = 0; row < n; ++row) {
        for (int col = 0; col < t; ++col) {
          host[static_cast<size_t>(row) * static_cast<size_t>(t) + static_cast<size_t>(col)] = col;
        }
      }
      pos_ids_->copy_from_host(host.data(), pos_ids_->numel());
      ctx_->synchronize();
    }
    const Tensor& pos = pos_emb_->forward_indices(*pos_ids_);
    ensure_tensor(embedded_, *ctx_, tok.shape());
    launch_add2(*ctx_, embedded_->data(), tok.data(), pos.data(), tok.numel());
    x = &*embedded_;
  }
  const Tensor& logits = network_->forward(*x);
  forwarded_ = true;
  used_embeddings_ = true;
  return logits;
}

void Transformer::reset_generate_state(GenerateState& st, int batch, int t_max) {
  std::vector<MultiHeadAttention::Cache> caches;
  caches.reserve(block_ptrs_.size());
  for (TransformerBlock* block : block_ptrs_) {
    MultiHeadAttention::Cache cache{Tensor(*ctx_, {1, 1, 1, 1}), Tensor(*ctx_, {1, 1, 1, 1}), 0};
    block->attn().reset_cache(cache, batch, t_max);
    caches.push_back(std::move(cache));
  }
  st.layer_caches = std::move(caches);
  st.t = 0;
}

const Tensor& Transformer::decode_step(const IndexTensor& next_ids, GenerateState& st) {
  if (training()) throw std::invalid_argument("decode_step");
  if (&next_ids.context() != ctx_ || next_ids.shape().size() != 2 || next_ids.shape()[1] != 1) {
    throw std::invalid_argument("next_ids");
  }
  const int n = next_ids.shape()[0];
  if (st.layer_caches.size() != block_ptrs_.size()) throw std::invalid_argument("cache");
  for (const MultiHeadAttention::Cache& cache : st.layer_caches) {
    if (cache.k.rank() != 4 || cache.k.shape()[0] != n) throw std::invalid_argument("cache");
  }

  if (st.t >= transformer_config(*network_).max_seq_len) throw std::invalid_argument("max_seq_len");

  const Tensor& tok = tok_emb_->forward_indices(next_ids);
  const Tensor* x = &tok;
  if (pos_emb_) {
    if (!pos_step_ids_ || pos_step_ids_->shape()[0] != n || pos_step_ids_->shape()[1] != 1) {
      pos_step_ids_.emplace(*ctx_, std::vector<int>{n, 1});
    }
    std::vector<int64_t> host(static_cast<size_t>(n), static_cast<int64_t>(st.t));
    pos_step_ids_->copy_from_host(host.data(), pos_step_ids_->numel());
    ctx_->synchronize();
    const Tensor& pos = pos_emb_->forward_indices(*pos_step_ids_);
    ensure_tensor(embedded_step_, *ctx_, tok.shape());
    launch_add2(*ctx_, embedded_step_->data(), tok.data(), pos.data(), tok.numel());
    x = &*embedded_step_;
  }

  for (size_t i = 0; i < block_ptrs_.size(); ++i) {
    x = &block_ptrs_[i]->forward_cached(*x, st.layer_caches[i]);
  }
  const Tensor& y = final_ln_ptr_->forward(*x);
  const Tensor& logits = lm_head_ptr_->forward(y);
  st.t += 1;
  return logits;
}

const Tensor& Transformer::prefill(const IndexTensor& prompt_ids, GenerateState& st, int t_max) {
  if (prompt_ids.shape().size() != 2) throw std::invalid_argument("prompt_ids");
  const int n = prompt_ids.shape()[0];
  const int t_prompt = prompt_ids.shape()[1];
  if (t_prompt > t_max) throw std::invalid_argument("t_max");
  reset_generate_state(st, n, t_max);

  const std::vector<int64_t> host = prompt_ids.to_host();

  IndexTensor column(*ctx_, {n, 1});
  const Tensor* logits = nullptr;
  for (int t = 0; t < t_prompt; ++t) {
    std::vector<int64_t> ids(static_cast<size_t>(n));
    for (int row = 0; row < n; ++row) {
      ids[static_cast<size_t>(row)] =
          host[static_cast<size_t>(row) * static_cast<size_t>(t_prompt) + static_cast<size_t>(t)];
    }
    column.copy_from_host(ids.data(), column.numel());
    ctx_->synchronize();
    logits = &decode_step(column, st);
  }
  return *logits;
}

const Tensor& Transformer::forward(const Tensor& embeddings) {
  const TransformerConfig& config = transformer_config(*network_);
  if (embeddings.rank() != 3 || embeddings.shape().back() != config.d_model) {
    throw std::invalid_argument("embeddings");
  }
  const Tensor& logits = network_->forward(embeddings);
  forwarded_ = true;
  used_embeddings_ = false;
  return logits;
}

const Tensor& Transformer::backward(const Tensor& grad_logits) {
  if (!forwarded_) throw Error("Transformer backward without forward");
  const Tensor& grad_x = network_->backward(grad_logits);
  if (used_embeddings_) {
    tok_emb_->backward(grad_x);
    if (pos_emb_) pos_emb_->backward(grad_x);
  }
  return grad_x;
}

void Transformer::zero_grad() {
  tok_emb_->zero_grad();
  if (pos_emb_) pos_emb_->zero_grad();
  network_->zero_grad();
}

void Transformer::train() {
  tok_emb_->set_training(true);
  if (pos_emb_) pos_emb_->set_training(true);
  network_->train();
}

void Transformer::eval() {
  tok_emb_->set_training(false);
  if (pos_emb_) pos_emb_->set_training(false);
  network_->eval();
}

bool Transformer::training() const { return network_->training(); }

std::vector<NamedParam> Transformer::parameters() const {
  std::vector<NamedParam> params;
  for (NamedParam param : tok_emb_->parameters()) {
    param.name = join_name(tok_emb_->name(), param.name);
    params.push_back(param);
  }
  if (pos_emb_) {
    for (NamedParam param : pos_emb_->parameters()) {
      param.name = join_name(pos_emb_->name(), param.name);
      params.push_back(param);
    }
  }
  const std::vector<NamedParam> trunk = network_->parameters();
  params.insert(params.end(), trunk.begin(), trunk.end());
  return params;
}

const Network& Transformer::network() const { return *network_; }

}  // namespace nn
