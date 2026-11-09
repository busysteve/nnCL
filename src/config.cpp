#include "nn/config.hpp"

#include <climits>
#include <stdexcept>
#include <string>

namespace nn {
namespace {

[[noreturn]] void bad(const char* field) { throw std::invalid_argument(field); }

void require_positive(int value, const char* field) {
  if (value < 1) bad(field);
}

}  // namespace

int conv_out_size(int in, int kernel, int stride, int pad) {
  if (kernel < 1) bad("kernel");
  if (stride < 1) bad("stride");
  if (pad < 0) bad("pad");
  const int out = (in + 2 * pad - kernel) / stride + 1;
  if (out < 1) bad("spatial");
  return out;
}

void validate(const MlpConfig& config) {
  require_positive(config.input_size, "input_size");
  require_positive(config.output_size, "output_size");
  if (config.activation != Activation::Relu && config.activation != Activation::Gelu &&
      config.activation != Activation::SwiGLU) {
    bad("activation");
  }
  for (int width : config.hidden) {
    require_positive(width, "hidden");
    if (config.activation == Activation::SwiGLU && width > INT_MAX / 2) bad("hidden");
  }
}

void validate(const ConvConfig& config) {
  require_positive(config.in_channels, "in_channels");
  require_positive(config.in_height, "in_height");
  require_positive(config.in_width, "in_width");
  if (config.convs.empty()) bad("convs");
  if (config.fc.empty()) bad("fc");
  int height = config.in_height;
  int width = config.in_width;
  for (const ConvSpec& spec : config.convs) {
    require_positive(spec.out_channels, "out_channels");
    height = conv_out_size(height, spec.kernel, spec.stride, spec.pad);
    width = conv_out_size(width, spec.kernel, spec.stride, spec.pad);
    height = conv_out_size(height, config.pool.kernel, config.pool.stride, 0);
    width = conv_out_size(width, config.pool.kernel, config.pool.stride, 0);
  }
  for (int units : config.fc) require_positive(units, "fc");
}

void validate(const ResNetConfig& config) {
  require_positive(config.in_channels, "in_channels");
  require_positive(config.in_height, "in_height");
  require_positive(config.in_width, "in_width");
  require_positive(config.stem_channels, "stem_channels");
  require_positive(config.output_size, "output_size");
  if (config.blocks_per_stage.empty() ||
      config.blocks_per_stage.size() != config.stage_widths.size()) {
    bad("stage_widths");
  }
  int height = conv_out_size(config.in_height, 3, 1, 1);
  int width = conv_out_size(config.in_width, 3, 1, 1);
  for (size_t stage = 0; stage < config.blocks_per_stage.size(); ++stage) {
    require_positive(config.stage_widths[stage], "stage_widths");
    if (config.blocks_per_stage[stage] < 1) bad("blocks_per_stage");
    const int first_stride = stage == 0 ? 1 : 2;
    height = conv_out_size(height, 3, first_stride, 1);
    width = conv_out_size(width, 3, first_stride, 1);
  }
}

void validate(const ConvResNetConfig& config) {
  require_positive(config.in_channels, "in_channels");
  require_positive(config.in_height, "in_height");
  require_positive(config.in_width, "in_width");
  if (config.fc.empty()) bad("fc");
  for (int units : config.fc) require_positive(units, "fc");
  if (config.convs.empty() && config.blocks_per_stage.empty()) bad("convs");
  if (config.stem_channels < 0) bad("stem_channels");
  if (config.stem_channels > 0 && config.blocks_per_stage.empty()) bad("blocks_per_stage");
  if (!config.blocks_per_stage.empty() &&
      config.blocks_per_stage.size() != config.stage_widths.size()) {
    bad("stage_widths");
  }

  int height = config.in_height;
  int width = config.in_width;
  for (const ConvSpec& spec : config.convs) {
    require_positive(spec.out_channels, "out_channels");
    height = conv_out_size(height, spec.kernel, spec.stride, spec.pad);
    width = conv_out_size(width, spec.kernel, spec.stride, spec.pad);
    height = conv_out_size(height, config.pool.kernel, config.pool.stride, 0);
    width = conv_out_size(width, config.pool.kernel, config.pool.stride, 0);
  }
  if (config.stem_channels > 0) {
    height = conv_out_size(height, 3, 1, 1);
    width = conv_out_size(width, 3, 1, 1);
  }
  for (size_t stage = 0; stage < config.blocks_per_stage.size(); ++stage) {
    require_positive(config.stage_widths[stage], "stage_widths");
    if (config.blocks_per_stage[stage] < 1) bad("blocks_per_stage");
    const int first_stride = stage == 0 ? 1 : 2;
    height = conv_out_size(height, 3, first_stride, 1);
    width = conv_out_size(width, 3, first_stride, 1);
  }
}

void validate(const RnnConfig& config) {
  require_positive(config.input_size, "input_size");
  require_positive(config.output_size, "output_size");
  if (config.hidden.empty()) bad("hidden");
  for (int width : config.hidden) require_positive(width, "hidden");
}

void validate(const TransformerConfig& config) {
  require_positive(config.vocab_size, "vocab_size");
  require_positive(config.d_model, "d_model");
  require_positive(config.num_heads, "num_heads");
  require_positive(config.num_layers, "num_layers");
  require_positive(config.d_ff, "d_ff");
  require_positive(config.max_seq_len, "max_seq_len");
  if (config.d_model % config.num_heads != 0) bad("num_heads");
  if (!(config.dropout >= 0.f) || !(config.dropout < 1.f)) bad("dropout");
  if (config.tie_embeddings) bad("tie_embeddings");
  if (config.activation != TransformerActivation::Gelu &&
      config.activation != TransformerActivation::SwiGLU) {
    bad("activation");
  }
  if (config.positional == PositionalEncoding::Rope) {
    if ((config.d_model / config.num_heads) % 2 != 0) bad("use_rope");
  } else if (config.positional != PositionalEncoding::LearnedAbsolute) {
    bad("positional");
  }
  if (config.activation == TransformerActivation::SwiGLU && config.d_ff > INT_MAX / 2) bad("d_ff");
}

}  // namespace nn
