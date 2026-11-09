#pragma once

#include "nn/tensor.hpp"

#include <random>

namespace nn {

void kaiming_normal(Tensor& weight, int fan_in, std::mt19937& rng);
void normal_(Tensor& tensor, float mean, float std, std::mt19937& rng);

}  // namespace nn
