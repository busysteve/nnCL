#pragma once

#include "nn/builders.hpp"
#include "nn/error.hpp"

#include <utility>

namespace nn {

// One image network that can include a conv+pool trunk, residual stages, or both.
class ConvResNet {
 public:
  ConvResNet(Context& ctx, const ConvResNetConfig& config, uint32_t seed);
  explicit ConvResNet(Network network);
  ConvResNet(ConvResNet&&) noexcept = default;
  ConvResNet& operator=(ConvResNet&&) noexcept = default;
  ConvResNet(const ConvResNet&) = delete;
  ConvResNet& operator=(const ConvResNet&) = delete;

  Network& network() { return network_; }
  const Network& network() const { return network_; }

  const Tensor& forward(const Tensor& input) { return network_.forward(input); }
  const Tensor& backward(const Tensor& grad_output) { return network_.backward(grad_output); }
  void zero_grad() { network_.zero_grad(); }
  void train() { network_.train(); }
  void eval() { network_.eval(); }
  bool training() const { return network_.training(); }
  Family family() const { return network_.family(); }
  const ConvResNetConfig& config() const { return std::get<ConvResNetConfig>(network_.config()); }

  std::vector<NamedParam> parameters() const { return network_.parameters(); }
  std::vector<NamedBuffer> buffers() const { return network_.buffers(); }

  void save(const std::filesystem::path& path, const Sgd* optimizer) const {
    network_.save(path, optimizer);
  }
  void save(const std::filesystem::path& path, const AdamW* optimizer) const {
    network_.save(path, optimizer);
  }
  void save(const std::filesystem::path& path, std::nullptr_t) const { network_.save(path, nullptr); }

  struct Loaded;
  static Loaded load(Context& ctx, const std::filesystem::path& path, LoadMode mode);

 private:
  Network network_;
};

struct ConvResNet::Loaded {
  ConvResNet model;
  std::optional<Sgd> optimizer;
  std::optional<AdamW> adamw;
  uint64_t step = 0;
};

inline ConvResNet::ConvResNet(Context& ctx, const ConvResNetConfig& config, uint32_t seed)
    : network_(build_conv_resnet(ctx, config, seed)) {}

inline ConvResNet::ConvResNet(Network network) : network_(std::move(network)) {
  if (network_.family() != Family::ConvResNet) throw Error("ConvResNet family");
}

inline ConvResNet::Loaded ConvResNet::load(Context& ctx, const std::filesystem::path& path,
                                           LoadMode mode) {
  Network::Loaded loaded = Network::load(ctx, path, mode);
  if (loaded.network.family() != Family::ConvResNet) throw Error("bad checkpoint family");
  return Loaded{ConvResNet(std::move(loaded.network)), std::move(loaded.optimizer),
                std::move(loaded.adamw), loaded.step};
}

}  // namespace nn
