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
  // type_ids is (N, T). Required only when type_vocab_size > 0 and the caller
  // does not want segment 0 in every position. forward(tokens) uses segment 0.
  const Tensor& forward(const IndexTensor& tokens, const IndexTensor& type_ids);
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
  // nullptr stays version 1, version 3 for a Qwen-style block, or version 5
  // when attention is bidirectional or a segment table is present.
  // An AdamW* (including null) is version 2, version 4 for that Qwen block,
  // or version 6 for the bidirectional block.
  // A config whose projection bias differs from bias, or whose later graph
  // fields differ from the version 1–6 defaults, is version 7 with no
  // optimizer block and version 8 when an optimizer block is written. A null
  // AdamW* on that config writes version 7.
  void save(const std::filesystem::path& path, std::nullptr_t) const {
    save(path, static_cast<const Sgd*>(nullptr));
  }

  struct Loaded;
  static Loaded load(Context& ctx, const std::filesystem::path& path, LoadMode mode);

 private:
  Context* ctx_ = nullptr;
  const Tensor& forward_tokens(const IndexTensor& tokens, const IndexTensor* type_ids);

  std::unique_ptr<Embedding> tok_emb_;
  std::unique_ptr<Embedding> pos_emb_;
  std::unique_ptr<Embedding> type_emb_;
  std::vector<TransformerBlock*> block_ptrs_;
  Layer* final_ln_ptr_ = nullptr;
  Linear* lm_head_ptr_ = nullptr;
  std::optional<Network> network_;
  std::optional<Tensor> embedded_;
  std::optional<Tensor> embedded_pair_;
  // Full-sequence embedded_ stays (N, T, D). A decode step must not resize it.
  std::optional<Tensor> embedded_step_;
  std::optional<Tensor> embedded_step_pair_;
  std::optional<IndexTensor> pos_ids_;
  std::optional<IndexTensor> pos_step_ids_;
  std::optional<IndexTensor> type_zero_;
  std::optional<IndexTensor> type_step_ids_;
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
