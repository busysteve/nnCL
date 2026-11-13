#pragma once

#include "nn/layers/embedding.hpp"
#include "nn/layers/multi_head_attention.hpp"
#include "nn/network.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace nn {

class TransformerBlock;
class Layer;
class Linear;

class Transformer {
 public:
  Transformer(Context& ctx, const TransformerConfig& config, uint32_t seed);
  Transformer(Transformer&&) noexcept = default;
  Transformer& operator=(Transformer&&) noexcept = default;
  Transformer(const Transformer&) = delete;
  Transformer& operator=(const Transformer&) = delete;

  const Tensor& forward(const IndexTensor& tokens);
  const Tensor& forward(const Tensor& embeddings);
  const Tensor& backward(const Tensor& grad_logits);

  // One KV cache per block. Not part of the checkpoint.
  struct GenerateState {
    std::vector<MultiHeadAttention::Cache> layer_caches;
    int t = 0;  // next absolute position; tokens consumed so far
  };

  void reset_generate_state(GenerateState& st, int batch, int t_max);
  const Tensor& decode_step(const IndexTensor& next_ids, GenerateState& st);
  const Tensor& prefill(const IndexTensor& prompt_ids, GenerateState& st, int t_max);

  void zero_grad();
  void train();
  void eval();
  bool training() const;

  std::vector<NamedParam> parameters() const;
  const Network& network() const;

  void save(const std::filesystem::path& path, const Sgd* optimizer) const;
  void save(const std::filesystem::path& path, const AdamW* optimizer) const;
  // nullptr stays version 1, or version 3 when the block is Qwen-style.
  // An AdamW* (including null) is version 2, or version 4 for that block.
  void save(const std::filesystem::path& path, std::nullptr_t) const {
    save(path, static_cast<const Sgd*>(nullptr));
  }

  struct Loaded;
  static Loaded load(Context& ctx, const std::filesystem::path& path, LoadMode mode);

 private:
  Context* ctx_ = nullptr;
  std::unique_ptr<Embedding> tok_emb_;
  std::unique_ptr<Embedding> pos_emb_;
  std::vector<TransformerBlock*> block_ptrs_;
  Layer* final_ln_ptr_ = nullptr;
  Linear* lm_head_ptr_ = nullptr;
  std::optional<Network> network_;
  std::optional<Tensor> embedded_;
  // Full-sequence embedded_ stays (N, T, D). A decode step must not resize it.
  std::optional<Tensor> embedded_step_;
  std::optional<IndexTensor> pos_ids_;
  std::optional<IndexTensor> pos_step_ids_;
  bool forwarded_ = false;
  bool used_embeddings_ = false;
};

struct Transformer::Loaded {
  Transformer model;
  std::optional<Sgd> optimizer;
  uint64_t step = 0;
  std::optional<AdamW> adamw;
};

}  // namespace nn
