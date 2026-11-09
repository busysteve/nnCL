#pragma once

#include "nn/tensor.hpp"

#include <optional>
#include <vector>

namespace nn {

inline void ensure_tensor(std::optional<Tensor>& slot, Context& ctx, const std::vector<int>& shape) {
  if (!slot || slot->shape() != shape) slot.emplace(ctx, shape);
}

}  // namespace nn
