#pragma once

#include "nn/network.hpp"

#include <cstdint>

namespace nn {

Network build_mlp(Context& ctx, const MlpConfig& config, uint32_t seed);
Network build_conv(Context& ctx, const ConvConfig& config, uint32_t seed);
Network build_resnet(Context& ctx, const ResNetConfig& config, uint32_t seed);
Network build_conv_resnet(Context& ctx, const ConvResNetConfig& config, uint32_t seed);
Network build_rnn(Context& ctx, const RnnConfig& config, uint32_t seed);
Network build_transformer_network(Context& ctx, const TransformerConfig& config, uint32_t seed);

}  // namespace nn
