#pragma once

#include "nn/index_tensor.hpp"
#include "nn/layer.hpp"

#include <optional>
#include <random>

namespace nn {

class Embedding : public Layer {
 public:
  Embedding(Context& ctx, int num_embeddings, int embedding_dim, std::mt19937& rng);
  const Tensor& forward_indices(const IndexTensor& indices);
  const Tensor& forward(const Tensor& input) override;
  const Tensor& backward(const Tensor& grad_output) override;
  std::vector<NamedParam> parameters() const override;
  void zero_grad() override;

 private:
  Context* ctx_;
  int num_embeddings_;
  int embedding_dim_;
  Tensor weight_;
  Tensor grad_weight_;
  std::optional<IndexTensor> cache_indices_;
  std::optional<Tensor> output_;
  std::optional<Tensor> grad_input_;
};

}  // namespace nn
