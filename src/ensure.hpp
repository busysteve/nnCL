#pragma once

#include "nn/tensor.hpp"

#include <optional>
#include <vector>

namespace nn {

inline void ensure_tensor(std::optional<Tensor>& slot, Context& ctx, const std::vector<int>& shape,
                          DType dtype = DType::Float32) {
  if (!slot || slot->shape() != shape || slot->dtype() != dtype) slot.emplace(ctx, shape, dtype);
}

// Hidden states, scores, and the other tensors kept for backward.
inline void ensure_activation(std::optional<Tensor>& slot, Context& ctx, const std::vector<int>& shape) {
  ensure_tensor(slot, ctx, shape, ctx.activation_dtype());
}

}  // namespace nn
