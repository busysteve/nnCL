#include "nn/layers/embedding.hpp"

#include "ensure.hpp"
#include "nn/error.hpp"
#include "nn/init.hpp"
#include "ocl.hpp"

#include <cstdint>
#include <vector>

namespace nn {
namespace {

std::vector<int> embedding_weight_shape(int num_embeddings, int embedding_dim) {
  if (num_embeddings < 1 || embedding_dim < 1) throw Error("Embedding size");
  return {num_embeddings, embedding_dim};
}

}  // namespace

Embedding::Embedding(Context& ctx, int num_embeddings, int embedding_dim, std::mt19937& rng)
    : ctx_(&ctx),
      num_embeddings_(num_embeddings),
      embedding_dim_(embedding_dim),
      weight_(ctx, embedding_weight_shape(num_embeddings, embedding_dim)),
      grad_weight_(ctx, {num_embeddings, embedding_dim}) {
  normal_(weight_, 0.f, 0.02f, rng);
  grad_weight_.zero();
}

std::vector<NamedParam> Embedding::parameters() const {
  return {NamedParam{"weight", const_cast<Tensor*>(&weight_), const_cast<Tensor*>(&grad_weight_)}};
}

void Embedding::zero_grad() { grad_weight_.zero(); }

const Tensor& Embedding::forward(const Tensor&) { throw Error("Embedding requires indices"); }

const Tensor& Embedding::forward_indices(const IndexTensor& indices) {
  if (&indices.context() != ctx_) throw Error("Embedding context");
  const std::vector<int64_t> host = indices.to_host();
  for (int64_t id : host) {
    if (id < 0 || id >= static_cast<int64_t>(num_embeddings_)) throw Error("index out of range");
  }
  if (!cache_indices_ || cache_indices_->shape() != indices.shape()) {
    cache_indices_.emplace(*ctx_, indices.shape());
  }
  cache_indices_->copy_from_host(host.data(), indices.numel());
  // copy_from_host is async; the local buffer has to stay alive until it finishes.
  ctx_->synchronize();

  std::vector<int> out_shape = indices.shape();
  out_shape.push_back(embedding_dim_);
  ensure_tensor(output_, *ctx_, out_shape);
  const int64_t rows = cache_indices_->numel();
  launch_embedding_gather(*ctx_, output_->data(), weight_.data(), cache_indices_->data(), rows,
                          embedding_dim_);
  return *output_;
}

const Tensor& Embedding::backward(const Tensor& grad_output) {
  if (!cache_indices_ || !output_) throw Error("Embedding backward without forward");
  if (grad_output.shape() != output_->shape()) throw Error("Embedding backward shape");
  const int64_t rows = cache_indices_->numel();
  launch_embedding_scatter(*ctx_, grad_weight_.data(), grad_output.data(), cache_indices_->data(),
                           rows, embedding_dim_);
  // Ids are discrete. Own a zero tensor so the returned reference follows the layer rule.
  ensure_tensor(grad_input_, *ctx_, grad_output.shape());
  grad_input_->zero();
  return *grad_input_;
}

}  // namespace nn
