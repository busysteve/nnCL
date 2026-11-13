#include "nn/builders.hpp"
#include "nn/error.hpp"
#include "nn/transformer.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nn {
namespace {

constexpr uint32_t kMagic = 0x314E4E43u;
constexpr uint32_t kVersion = 1u;
constexpr uint32_t kVersionAdamW = 2u;
constexpr uint32_t kVersionExt = 3u;
constexpr uint32_t kVersionExtAdamW = 4u;
constexpr uint32_t kKindSgd = 1u;
constexpr uint32_t kKindAdamW = 2u;
constexpr uint32_t kFloat32 = 1u;

class ByteWriter {
 public:
  void u32(uint32_t value) {
    bytes_.push_back(static_cast<uint8_t>(value));
    bytes_.push_back(static_cast<uint8_t>(value >> 8));
    bytes_.push_back(static_cast<uint8_t>(value >> 16));
    bytes_.push_back(static_cast<uint8_t>(value >> 24));
  }
  void u64(uint64_t value) {
    u32(static_cast<uint32_t>(value));
    u32(static_cast<uint32_t>(value >> 32));
  }
  void i32(int32_t value) { u32(static_cast<uint32_t>(value)); }
  void f32(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    u32(bits);
  }
  void pad8() {
    while (bytes_.size() % 8 != 0) bytes_.push_back(0);
  }
  void str(const std::string& value) {
    if (value.size() > 256) throw Error("checkpoint tensor mismatch: name");
    u32(static_cast<uint32_t>(value.size()));
    bytes_.insert(bytes_.end(), value.begin(), value.end());
    pad8();
  }
  void shape(const std::vector<int>& dims) {
    u32(static_cast<uint32_t>(dims.size()));
    for (int dim : dims) i32(dim);
    pad8();
  }
  void f32s(const std::vector<float>& values) {
    for (float value : values) f32(value);
  }
  const std::vector<uint8_t>& bytes() const { return bytes_; }

 private:
  std::vector<uint8_t> bytes_;
};

class ByteReader {
 public:
  explicit ByteReader(std::vector<uint8_t> bytes) : bytes_(std::move(bytes)) {}

  uint32_t u32() {
    need(4);
    uint32_t value = static_cast<uint32_t>(bytes_[pos_]) |
                     (static_cast<uint32_t>(bytes_[pos_ + 1]) << 8) |
                     (static_cast<uint32_t>(bytes_[pos_ + 2]) << 16) |
                     (static_cast<uint32_t>(bytes_[pos_ + 3]) << 24);
    pos_ += 4;
    return value;
  }
  uint64_t u64() {
    const uint64_t low = u32();
    const uint64_t high = u32();
    return low | (high << 32);
  }
  int32_t i32() { return static_cast<int32_t>(u32()); }
  float f32() {
    const uint32_t bits = u32();
    float value = 0.f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  void pad8() {
    while (pos_ % 8 != 0) {
      need(1);
      ++pos_;
    }
  }
  std::string str() {
    const uint32_t len = u32();
    if (len > 256) throw Error("checkpoint tensor mismatch: name");
    need(len);
    std::string value(reinterpret_cast<const char*>(bytes_.data() + pos_), len);
    pos_ += len;
    pad8();
    return value;
  }
  std::vector<int> shape() {
    const uint32_t rank = u32();
    if (rank != 1 && rank != 2 && rank != 4) throw Error("checkpoint tensor mismatch: rank");
    if (rank > (bytes_.size() - pos_) / 4) throw Error("truncated checkpoint");
    std::vector<int> dims(rank);
    for (uint32_t i = 0; i < rank; ++i) dims[i] = i32();
    pad8();
    return dims;
  }
  std::vector<float> f32s(uint32_t count) {
    if (static_cast<uint64_t>(count) > (bytes_.size() - pos_) / 4) throw Error("truncated checkpoint");
    std::vector<float> values(count);
    for (uint32_t i = 0; i < count; ++i) values[i] = f32();
    return values;
  }

 private:
  void need(size_t n) const {
    if (pos_ + n > bytes_.size()) throw Error("truncated checkpoint");
  }
  std::vector<uint8_t> bytes_;
  size_t pos_ = 0;
};

struct ParsedParam {
  std::string name;
  std::vector<int> shape;
  std::vector<float> data;
};

struct ParsedBn {
  std::string prefix;
  uint64_t batches = 0;
  std::vector<float> mean;
  std::vector<float> var;
};

struct ParsedOpt {
  float learning_rate = 0.f;
  float momentum = 0.f;
  std::vector<std::vector<float>> velocity;
};

struct ParsedAdamGroup {
  float weight_decay = 0.f;
  std::vector<std::vector<float>> m;
  std::vector<std::vector<float>> v;
};

struct ParsedAdamW {
  float lr = 0.f;
  float beta1 = 0.f;
  float beta2 = 0.f;
  float eps = 0.f;
  std::vector<ParsedAdamGroup> groups;
};

struct Header {
  uint32_t version = 0;
  uint32_t family = 0;
  uint32_t training = 0;
  uint64_t step = 0;
  uint32_t optimizer_present = 0;
};

int64_t product(const std::vector<int>& shape) {
  int64_t n = 1;
  for (int dim : shape) {
    if (dim < 1) return -1;
    n *= dim;
  }
  return n;
}

std::string strip_suffix(const std::string& name, const std::string& suffix) {
  if (name.size() < suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
    return {};
  }
  return name.substr(0, name.size() - suffix.size());
}

void write_all(int fd, const uint8_t* data, size_t n) {
  while (n > 0) {
    const ssize_t wrote = ::write(fd, data, n);
    if (wrote < 0) {
      if (errno == EINTR) continue;
      throw Error(std::string("checkpoint write: ") + std::strerror(errno));
    }
    data += wrote;
    n -= static_cast<size_t>(wrote);
  }
}

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw Error("truncated checkpoint");
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  if (size < 32) throw Error("truncated checkpoint");
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  in.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!in) throw Error("truncated checkpoint");
  return bytes;
}

Header read_header(ByteReader& reader, uint32_t max_version = kVersionExtAdamW) {
  const uint32_t magic = reader.u32();
  if (magic != kMagic) throw Error("bad checkpoint magic");
  const uint32_t version = reader.u32();
  if (version < 1u || version > max_version) {
    throw Error("unsupported checkpoint version " + std::to_string(version));
  }
  Header header;
  header.version = version;
  header.family = reader.u32();
  header.training = reader.u32();
  header.step = reader.u64();
  header.optimizer_present = reader.u32();
  const uint32_t reserved = reader.u32();
  if (reserved != 0 || header.training > 1 || header.optimizer_present > 1) {
    throw Error("bad checkpoint header");
  }
  if (header.family < 1 || header.family > 6) throw Error("bad checkpoint family");
  return header;
}

bool transformer_extended(const TransformerConfig& config) {
  const bool wide = config.head_dim > 0 &&
                    (config.d_model % config.num_heads != 0 || config.head_dim != config.d_model / config.num_heads);
  const bool grouped = config.num_kv_heads > 0 && config.num_kv_heads != config.num_heads;
  return wide || grouped || config.qk_norm || config.rope_theta != 10000.f || config.rms_eps != 1e-5f;
}

uint32_t transformer_file_version(const TransformerConfig& config, bool kind_word) {
  if (!transformer_extended(config)) return kind_word ? kVersionAdamW : kVersion;
  return kind_word ? kVersionExtAdamW : kVersionExt;
}

std::variant<MlpConfig, ConvConfig, ResNetConfig, ConvResNetConfig, RnnConfig, TransformerConfig>
read_config(ByteReader& reader, uint32_t family, uint32_t version) {
  if (family == 1) {
    MlpConfig config;
    config.input_size = reader.i32();
    config.output_size = reader.i32();
    config.activation = static_cast<Activation>(reader.u32());
    const uint32_t hidden_count = reader.u32();
    if (hidden_count > 1000000u) throw Error("truncated checkpoint");
    config.hidden.resize(hidden_count);
    for (uint32_t i = 0; i < hidden_count; ++i) config.hidden[i] = reader.i32();
    reader.pad8();
    validate(config);
    return config;
  }
  if (family == 2) {
    ConvConfig config;
    config.in_channels = reader.i32();
    config.in_height = reader.i32();
    config.in_width = reader.i32();
    config.pool.kernel = reader.i32();
    config.pool.stride = reader.i32();
    const uint32_t conv_count = reader.u32();
    if (conv_count > 1000000u) throw Error("truncated checkpoint");
    config.convs.resize(conv_count);
    for (uint32_t i = 0; i < conv_count; ++i) {
      config.convs[i].out_channels = reader.i32();
      config.convs[i].kernel = reader.i32();
      config.convs[i].stride = reader.i32();
      config.convs[i].pad = reader.i32();
    }
    const uint32_t fc_count = reader.u32();
    if (fc_count > 1000000u) throw Error("truncated checkpoint");
    config.fc.resize(fc_count);
    for (uint32_t i = 0; i < fc_count; ++i) config.fc[i] = reader.i32();
    reader.pad8();
    validate(config);
    return config;
  }
  if (family == 3) {
    ResNetConfig config;
    config.in_channels = reader.i32();
    config.in_height = reader.i32();
    config.in_width = reader.i32();
    config.stem_channels = reader.i32();
    config.output_size = reader.i32();
    const uint32_t stage_count = reader.u32();
    if (stage_count > 1000000u) throw Error("truncated checkpoint");
    config.blocks_per_stage.resize(stage_count);
    config.stage_widths.resize(stage_count);
    for (uint32_t i = 0; i < stage_count; ++i) {
      config.blocks_per_stage[i] = reader.i32();
      config.stage_widths[i] = reader.i32();
    }
    reader.pad8();
    validate(config);
    return config;
  }
  if (family == 4) {
    ConvResNetConfig config;
    config.in_channels = reader.i32();
    config.in_height = reader.i32();
    config.in_width = reader.i32();
    config.stem_channels = reader.i32();
    config.pool.kernel = reader.i32();
    config.pool.stride = reader.i32();
    const uint32_t conv_count = reader.u32();
    if (conv_count > 1000000u) throw Error("truncated checkpoint");
    config.convs.resize(conv_count);
    for (uint32_t i = 0; i < conv_count; ++i) {
      config.convs[i].out_channels = reader.i32();
      config.convs[i].kernel = reader.i32();
      config.convs[i].stride = reader.i32();
      config.convs[i].pad = reader.i32();
    }
    const uint32_t stage_count = reader.u32();
    if (stage_count > 1000000u) throw Error("truncated checkpoint");
    config.blocks_per_stage.resize(stage_count);
    config.stage_widths.resize(stage_count);
    for (uint32_t i = 0; i < stage_count; ++i) {
      config.blocks_per_stage[i] = reader.i32();
      config.stage_widths[i] = reader.i32();
    }
    const uint32_t fc_count = reader.u32();
    if (fc_count > 1000000u) throw Error("truncated checkpoint");
    config.fc.resize(fc_count);
    for (uint32_t i = 0; i < fc_count; ++i) config.fc[i] = reader.i32();
    reader.pad8();
    validate(config);
    return config;
  }
  if (family == 5) {
    RnnConfig config;
    config.input_size = reader.i32();
    config.output_size = reader.i32();
    const uint32_t hidden_count = reader.u32();
    if (hidden_count > 1000000u) throw Error("truncated checkpoint");
    config.hidden.resize(hidden_count);
    for (uint32_t i = 0; i < hidden_count; ++i) config.hidden[i] = reader.i32();
    reader.pad8();
    validate(config);
    return config;
  }
  if (family == 6) {
    TransformerConfig config;
    config.vocab_size = reader.i32();
    config.d_model = reader.i32();
    config.num_heads = reader.i32();
    config.num_layers = reader.i32();
    config.d_ff = reader.i32();
    config.max_seq_len = reader.i32();
    config.dropout = reader.f32();
    config.tie_embeddings = reader.u32() != 0;
    config.bias = reader.u32() != 0;
    config.activation = static_cast<TransformerActivation>(reader.u32());
    config.positional = static_cast<PositionalEncoding>(reader.u32());
    if (version >= kVersionExt) {
      config.head_dim = reader.i32();
      config.num_kv_heads = reader.i32();
      config.qk_norm = reader.u32() != 0;
      config.rope_theta = reader.f32();
      config.rms_eps = reader.f32();
    }
    reader.pad8();
    validate(config);
    return config;
  }
  throw Error("bad checkpoint family");
}

std::vector<ParsedParam> read_params(ByteReader& reader, uint32_t* count_out) {
  const uint32_t count = reader.u32();
  reader.u32();
  if (count > 1000000u) throw Error("truncated checkpoint");
  std::vector<ParsedParam> params(count);
  for (uint32_t i = 0; i < count; ++i) {
    params[i].name = reader.str();
    params[i].shape = reader.shape();
    const uint32_t dtype = reader.u32();
    const uint32_t numel = reader.u32();
    if (dtype != kFloat32 || product(params[i].shape) != static_cast<int64_t>(numel)) {
      throw Error("checkpoint tensor mismatch: " + params[i].name);
    }
    params[i].data = reader.f32s(numel);
    reader.pad8();
  }
  *count_out = count;
  return params;
}

std::vector<ParsedBn> read_bn(ByteReader& reader) {
  const uint32_t count = reader.u32();
  reader.u32();
  if (count > 1000000u) throw Error("truncated checkpoint");
  std::vector<ParsedBn> layers(count);
  for (uint32_t i = 0; i < count; ++i) {
    layers[i].prefix = reader.str();
    layers[i].batches = reader.u64();
    const uint32_t channels = reader.u32();
    reader.u32();
    if (channels == 0 || channels > 1000000u) throw Error("checkpoint tensor mismatch: " + layers[i].prefix);
    layers[i].mean = reader.f32s(channels);
    layers[i].var = reader.f32s(channels);
    reader.pad8();
  }
  return layers;
}

ParsedOpt read_opt(ByteReader& reader, const std::vector<ParsedParam>& params) {
  ParsedOpt opt;
  opt.learning_rate = reader.f32();
  opt.momentum = reader.f32();
  const uint32_t count = reader.u32();
  reader.u32();
  if (count != params.size()) throw Error("optimizer state mismatch");
  opt.velocity.resize(count);
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t numel = reader.u32();
    reader.u32();
    if (numel != params[i].data.size()) throw Error("optimizer state mismatch");
    opt.velocity[i] = reader.f32s(numel);
    reader.pad8();
  }
  return opt;
}

ParsedAdamW read_adamw(ByteReader& reader, const std::vector<ParsedParam>& params) {
  ParsedAdamW opt;
  opt.lr = reader.f32();
  opt.beta1 = reader.f32();
  opt.beta2 = reader.f32();
  opt.eps = reader.f32();
  const uint32_t group_count = reader.u32();
  if (reader.u32() != 0) throw Error("optimizer state mismatch");
  if (group_count == 0 || group_count > 1000000u) throw Error("optimizer state mismatch");
  opt.groups.resize(group_count);
  size_t seen = 0;
  for (uint32_t g = 0; g < group_count; ++g) {
    opt.groups[g].weight_decay = reader.f32();
    const uint32_t param_count = reader.u32();
    if (param_count == 0 || param_count > 1000000u) throw Error("optimizer state mismatch");
    if (seen + param_count > params.size()) throw Error("optimizer state mismatch");
    opt.groups[g].m.resize(param_count);
    opt.groups[g].v.resize(param_count);
    for (uint32_t i = 0; i < param_count; ++i) {
      const uint32_t numel_m = reader.u32();
      if (reader.u32() != 0) throw Error("optimizer state mismatch");
      if (seen >= params.size() || numel_m != params[seen].data.size()) {
        throw Error("optimizer state mismatch");
      }
      opt.groups[g].m[i] = reader.f32s(numel_m);
      reader.pad8();
      const uint32_t numel_v = reader.u32();
      if (reader.u32() != 0 || numel_v != numel_m) throw Error("optimizer state mismatch");
      opt.groups[g].v[i] = reader.f32s(numel_v);
      reader.pad8();
      ++seen;
    }
  }
  if (seen != params.size()) throw Error("optimizer state mismatch");
  return opt;
}

void require_match(const ParsedParam& parsed, const NamedParam& param) {
  if (parsed.name != param.name || parsed.shape != param.value->shape() ||
      static_cast<int64_t>(parsed.data.size()) != param.value->numel()) {
    throw Error("checkpoint tensor mismatch: " + parsed.name);
  }
}

void write_transformer_config(ByteWriter& writer, const TransformerConfig& config, bool extended) {
  writer.i32(config.vocab_size);
  writer.i32(config.d_model);
  writer.i32(config.num_heads);
  writer.i32(config.num_layers);
  writer.i32(config.d_ff);
  writer.i32(config.max_seq_len);
  writer.f32(config.dropout);
  writer.u32(config.tie_embeddings ? 1u : 0u);
  writer.u32(config.bias ? 1u : 0u);
  writer.u32(static_cast<uint32_t>(config.activation));
  writer.u32(static_cast<uint32_t>(config.positional));
  if (extended) {
    writer.i32(config.head_dim);
    writer.i32(config.num_kv_heads);
    writer.u32(config.qk_norm ? 1u : 0u);
    writer.f32(config.rope_theta);
    writer.f32(config.rms_eps);
  }
}

void write_params(ByteWriter& writer, const std::vector<NamedParam>& params) {
  writer.u32(static_cast<uint32_t>(params.size()));
  writer.u32(0);
  for (const NamedParam& param : params) {
    const std::vector<float> values = param.value->to_host();
    writer.str(param.name);
    writer.shape(param.value->shape());
    writer.u32(kFloat32);
    writer.u32(static_cast<uint32_t>(values.size()));
    writer.f32s(values);
    writer.pad8();
  }
}

void write_bn(ByteWriter& writer, const std::vector<NamedBuffer>& buffers) {
  if (buffers.size() % 2 != 0) throw Error("checkpoint tensor mismatch: batch-norm buffers");
  writer.u32(static_cast<uint32_t>(buffers.size() / 2));
  writer.u32(0);
  for (size_t i = 0; i < buffers.size(); i += 2) {
    const std::string mean_prefix = strip_suffix(buffers[i].name, ".running_mean");
    const std::string var_prefix = strip_suffix(buffers[i + 1].name, ".running_var");
    if (mean_prefix.empty() || mean_prefix != var_prefix) {
      throw Error("checkpoint tensor mismatch: " + buffers[i].name);
    }
    const std::vector<float> mean = buffers[i].value->to_host();
    const std::vector<float> var = buffers[i + 1].value->to_host();
    writer.str(mean_prefix);
    writer.u64(*buffers[i].batches);
    writer.u32(static_cast<uint32_t>(mean.size()));
    writer.u32(0);
    writer.f32s(mean);
    writer.f32s(var);
    writer.pad8();
  }
}

void write_sgd(ByteWriter& writer, const Sgd& optimizer) {
  writer.f32(optimizer.learning_rate());
  writer.f32(optimizer.momentum());
  writer.u32(static_cast<uint32_t>(optimizer.velocity().size()));
  writer.u32(0);
  for (const Tensor& velocity : optimizer.velocity()) {
    const std::vector<float> values = velocity.to_host();
    writer.u32(static_cast<uint32_t>(values.size()));
    writer.u32(0);
    writer.f32s(values);
    writer.pad8();
  }
}

void commit_checkpoint(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
  const std::filesystem::path tmp = path.string() + ".tmp";
  int fd = ::open(tmp.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
  if (fd < 0) throw Error(std::string("checkpoint write: ") + std::strerror(errno));
  try {
    write_all(fd, bytes.data(), bytes.size());
    if (::fsync(fd) != 0) throw Error(std::string("checkpoint write: ") + std::strerror(errno));
    if (::close(fd) != 0) throw Error(std::string("checkpoint write: ") + std::strerror(errno));
    fd = -1;
    std::filesystem::rename(tmp, path);
  } catch (...) {
    if (fd >= 0) ::close(fd);
    ::unlink(tmp.c_str());
    throw;
  }
}

void require_owned_by(const Network& network, const std::vector<NamedParam>& params) {
  const std::vector<NamedParam> mine = network.parameters();
  if (mine.size() != params.size()) throw Error("optimizer does not belong to this network");
  for (size_t i = 0; i < mine.size(); ++i) {
    if (mine[i].value != params[i].value) throw Error("optimizer does not belong to this network");
  }
}

std::vector<NamedParam> flat_params(const AdamW& optimizer) {
  std::vector<NamedParam> params;
  for (const AdamWGroup& group : optimizer.groups()) {
    params.insert(params.end(), group.params.begin(), group.params.end());
  }
  return params;
}

void write_optimizer_sgd(ByteWriter& writer, const Sgd& optimizer) {
  writer.f32(optimizer.learning_rate());
  writer.f32(optimizer.momentum());
  writer.u32(static_cast<uint32_t>(optimizer.velocity().size()));
  writer.u32(0);
  for (const Tensor& velocity : optimizer.velocity()) {
    const std::vector<float> values = velocity.to_host();
    writer.u32(static_cast<uint32_t>(values.size()));
    writer.u32(0);
    writer.f32s(values);
    writer.pad8();
  }
}

void write_optimizer_adamw(ByteWriter& writer, const AdamW& optimizer) {
  writer.u32(kKindAdamW);
  writer.f32(optimizer.learning_rate());
  writer.f32(optimizer.beta1());
  writer.f32(optimizer.beta2());
  writer.f32(optimizer.eps());
  const std::vector<AdamWGroup>& groups = optimizer.groups();
  writer.u32(static_cast<uint32_t>(groups.size()));
  writer.u32(0);
  size_t flat = 0;
  for (const AdamWGroup& group : groups) {
    writer.f32(group.weight_decay);
    writer.u32(static_cast<uint32_t>(group.params.size()));
    for (const NamedParam& param : group.params) {
      if (flat >= optimizer.m().size() || flat >= optimizer.v().size() || param.value == nullptr) {
        throw Error("optimizer state mismatch");
      }
      const std::vector<float> m = optimizer.m()[flat].to_host();
      const std::vector<float> v = optimizer.v()[flat].to_host();
      if (m.size() != static_cast<size_t>(param.value->numel()) || v.size() != m.size()) {
        throw Error("optimizer state mismatch");
      }
      writer.u32(static_cast<uint32_t>(m.size()));
      writer.u32(0);
      writer.f32s(m);
      writer.pad8();
      writer.u32(static_cast<uint32_t>(v.size()));
      writer.u32(0);
      writer.f32s(v);
      writer.pad8();
      ++flat;
    }
  }
  if (flat != optimizer.m().size() || flat != optimizer.v().size()) throw Error("optimizer state mismatch");
}

void write_model(ByteWriter& writer, const Network& network, uint32_t version, uint64_t step,
                 uint32_t optimizer_present) {
  const std::vector<NamedParam> params = network.parameters();
  const std::vector<NamedBuffer> buffers = network.buffers();
  if (buffers.size() % 2 != 0) throw Error("checkpoint tensor mismatch: batch-norm buffers");

  writer.u32(kMagic);
  writer.u32(version);
  writer.u32(static_cast<uint32_t>(network.family()));
  writer.u32(network.training() ? 1u : 0u);
  writer.u64(step);
  writer.u32(optimizer_present);
  writer.u32(0);

  std::visit(
      [&](const auto& config) {
        using T = std::decay_t<decltype(config)>;
        if constexpr (std::is_same_v<T, MlpConfig>) {
          writer.i32(config.input_size);
          writer.i32(config.output_size);
          writer.u32(static_cast<uint32_t>(config.activation));
          writer.u32(static_cast<uint32_t>(config.hidden.size()));
          for (int width : config.hidden) writer.i32(width);
        } else if constexpr (std::is_same_v<T, ConvConfig>) {
          writer.i32(config.in_channels);
          writer.i32(config.in_height);
          writer.i32(config.in_width);
          writer.i32(config.pool.kernel);
          writer.i32(config.pool.stride);
          writer.u32(static_cast<uint32_t>(config.convs.size()));
          for (const ConvSpec& spec : config.convs) {
            writer.i32(spec.out_channels);
            writer.i32(spec.kernel);
            writer.i32(spec.stride);
            writer.i32(spec.pad);
          }
          writer.u32(static_cast<uint32_t>(config.fc.size()));
          for (int width : config.fc) writer.i32(width);
        } else if constexpr (std::is_same_v<T, ResNetConfig>) {
          writer.i32(config.in_channels);
          writer.i32(config.in_height);
          writer.i32(config.in_width);
          writer.i32(config.stem_channels);
          writer.i32(config.output_size);
          writer.u32(static_cast<uint32_t>(config.blocks_per_stage.size()));
          for (size_t i = 0; i < config.blocks_per_stage.size(); ++i) {
            writer.i32(config.blocks_per_stage[i]);
            writer.i32(config.stage_widths[i]);
          }
        } else if constexpr (std::is_same_v<T, ConvResNetConfig>) {
          writer.i32(config.in_channels);
          writer.i32(config.in_height);
          writer.i32(config.in_width);
          writer.i32(config.stem_channels);
          writer.i32(config.pool.kernel);
          writer.i32(config.pool.stride);
          writer.u32(static_cast<uint32_t>(config.convs.size()));
          for (const ConvSpec& spec : config.convs) {
            writer.i32(spec.out_channels);
            writer.i32(spec.kernel);
            writer.i32(spec.stride);
            writer.i32(spec.pad);
          }
          writer.u32(static_cast<uint32_t>(config.blocks_per_stage.size()));
          for (size_t i = 0; i < config.blocks_per_stage.size(); ++i) {
            writer.i32(config.blocks_per_stage[i]);
            writer.i32(config.stage_widths[i]);
          }
          writer.u32(static_cast<uint32_t>(config.fc.size()));
          for (int width : config.fc) writer.i32(width);
        } else if constexpr (std::is_same_v<T, RnnConfig>) {
          writer.i32(config.input_size);
          writer.i32(config.output_size);
          writer.u32(static_cast<uint32_t>(config.hidden.size()));
          for (int width : config.hidden) writer.i32(width);
        } else {
          static_assert(std::is_same_v<T, TransformerConfig>);
          write_transformer_config(writer, config, version >= kVersionExt);
        }
      },
      network.config());
  writer.pad8();

  write_params(writer, params);
  write_bn(writer, buffers);
}

AdamW resume_adamw(const std::vector<NamedParam>& live, const ParsedAdamW& parsed, uint64_t step) {
  std::vector<AdamWGroup> groups;
  groups.reserve(parsed.groups.size());
  size_t cursor = 0;
  for (const ParsedAdamGroup& group : parsed.groups) {
    AdamWGroup built;
    built.weight_decay = group.weight_decay;
    built.params.reserve(group.m.size());
    for (size_t i = 0; i < group.m.size(); ++i) built.params.push_back(live[cursor + i]);
    cursor += group.m.size();
    groups.push_back(std::move(built));
  }
  AdamW adamw(std::move(groups), parsed.lr, parsed.beta1, parsed.beta2, parsed.eps);
  size_t flat = 0;
  for (const ParsedAdamGroup& group : parsed.groups) {
    for (size_t i = 0; i < group.m.size(); ++i, ++flat) {
      adamw.m()[flat].copy_from_host(group.m[i]);
      adamw.v()[flat].copy_from_host(group.v[i]);
    }
  }
  adamw.set_step_count(step);
  return adamw;
}

}  // namespace

void Network::save(const std::filesystem::path& path, const Sgd* optimizer) const {
  if (optimizer) require_owned_by(*this, optimizer->params());
  ctx_->synchronize();
  ByteWriter writer;
  const uint32_t version = std::holds_alternative<TransformerConfig>(config())
                               ? transformer_file_version(std::get<TransformerConfig>(config()), false)
                               : kVersion;
  write_model(writer, *this, version, optimizer ? optimizer->step_count() : 0, optimizer ? 1u : 0u);
  if (optimizer) write_optimizer_sgd(writer, *optimizer);
  commit_checkpoint(path, writer.bytes());
}

void Network::save(const std::filesystem::path& path, const AdamW* optimizer) const {
  // Resume splits parameters() by stored group sizes, so groups must be that order.
  if (optimizer) require_owned_by(*this, flat_params(*optimizer));
  ctx_->synchronize();
  ByteWriter writer;
  const uint32_t version = std::holds_alternative<TransformerConfig>(config())
                               ? transformer_file_version(std::get<TransformerConfig>(config()), true)
                               : kVersionAdamW;
  write_model(writer, *this, version, optimizer ? optimizer->step_count() : 0, optimizer ? 1u : 0u);
  if (optimizer) write_optimizer_adamw(writer, *optimizer);
  commit_checkpoint(path, writer.bytes());
}

Network::Loaded Network::load(Context& ctx, const std::filesystem::path& path, LoadMode mode) {
  ByteReader reader(read_file(path));
  const Header header = read_header(reader);
  auto config = read_config(reader, header.family, header.version);
  uint32_t param_count = 0;
  const std::vector<ParsedParam> parsed_params = read_params(reader, &param_count);
  const std::vector<ParsedBn> parsed_bn = read_bn(reader);
  std::optional<ParsedOpt> parsed_sgd;
  std::optional<ParsedAdamW> parsed_adamw;
  if (header.optimizer_present) {
    if (header.version == kVersion || header.version == kVersionExt) {
      parsed_sgd = read_opt(reader, parsed_params);
    } else {
      const uint32_t kind = reader.u32();
      if (kind == kKindSgd) parsed_sgd = read_opt(reader, parsed_params);
      else if (kind == kKindAdamW) parsed_adamw = read_adamw(reader, parsed_params);
      else throw Error("unsupported optimizer kind " + std::to_string(kind));
    }
  }
  if (mode == LoadMode::Resume && !parsed_sgd && !parsed_adamw) throw Error("optimizer block missing");

  Network network = std::visit(
      [&](const auto& cfg) -> Network {
        using T = std::decay_t<decltype(cfg)>;
        if constexpr (std::is_same_v<T, MlpConfig>) return build_mlp(ctx, cfg, 0);
        else if constexpr (std::is_same_v<T, ConvConfig>) return build_conv(ctx, cfg, 0);
        else if constexpr (std::is_same_v<T, ResNetConfig>) return build_resnet(ctx, cfg, 0);
        else if constexpr (std::is_same_v<T, ConvResNetConfig>) return build_conv_resnet(ctx, cfg, 0);
        else if constexpr (std::is_same_v<T, RnnConfig>) return build_rnn(ctx, cfg, 0);
        else {
          static_assert(std::is_same_v<T, TransformerConfig>);
          return build_transformer_network(ctx, cfg, 0);
        }
      },
      config);

  const std::vector<NamedParam> params = network.parameters();
  if (params.size() != parsed_params.size()) throw Error("checkpoint tensor mismatch: count");
  for (size_t i = 0; i < params.size(); ++i) require_match(parsed_params[i], params[i]);

  const std::vector<NamedBuffer> buffers = network.buffers();
  if (buffers.size() != parsed_bn.size() * 2) throw Error("checkpoint tensor mismatch: batch-norm buffers");
  for (size_t i = 0; i < parsed_bn.size(); ++i) {
    const std::string mean_prefix = strip_suffix(buffers[2 * i].name, ".running_mean");
    const std::string var_prefix = strip_suffix(buffers[2 * i + 1].name, ".running_var");
    if (mean_prefix != parsed_bn[i].prefix || var_prefix != parsed_bn[i].prefix ||
        buffers[2 * i].value->shape() != std::vector<int>{static_cast<int>(parsed_bn[i].mean.size())} ||
        buffers[2 * i + 1].value->shape() != std::vector<int>{static_cast<int>(parsed_bn[i].var.size())}) {
      throw Error("checkpoint tensor mismatch: " + parsed_bn[i].prefix);
    }
  }

  for (size_t i = 0; i < params.size(); ++i) params[i].value->copy_from_host(parsed_params[i].data);
  for (size_t i = 0; i < parsed_bn.size(); ++i) {
    buffers[2 * i].value->copy_from_host(parsed_bn[i].mean);
    buffers[2 * i + 1].value->copy_from_host(parsed_bn[i].var);
    *buffers[2 * i].batches = parsed_bn[i].batches;
  }
  if (header.training) network.train();
  else network.eval();

  Loaded loaded{std::move(network), std::nullopt, std::nullopt, header.step};
  if (mode == LoadMode::Resume && parsed_sgd) {
    Sgd optimizer(loaded.network.parameters(), parsed_sgd->learning_rate, parsed_sgd->momentum);
    for (size_t i = 0; i < parsed_sgd->velocity.size(); ++i) {
      optimizer.velocity()[i].copy_from_host(parsed_sgd->velocity[i]);
    }
    optimizer.set_step_count(header.step);
    loaded.optimizer = std::move(optimizer);
  } else if (mode == LoadMode::Resume && parsed_adamw) {
    loaded.adamw = resume_adamw(loaded.network.parameters(), *parsed_adamw, header.step);
  }
  ctx.synchronize();
  return loaded;
}

void Transformer::save(const std::filesystem::path& path, const Sgd* optimizer) const {
  if (optimizer) {
    const std::vector<NamedParam> mine = parameters();
    const std::vector<NamedParam>& theirs = optimizer->params();
    if (mine.size() != theirs.size()) throw Error("optimizer does not belong to this network");
    for (size_t i = 0; i < mine.size(); ++i) {
      if (mine[i].value != theirs[i].value) throw Error("optimizer does not belong to this network");
    }
  }
  ctx_->synchronize();
  const std::vector<NamedParam> params = parameters();
  const TransformerConfig& config = std::get<TransformerConfig>(network().config());

  ByteWriter writer;
  writer.u32(kMagic);
  writer.u32(transformer_file_version(config, optimizer != nullptr));
  writer.u32(static_cast<uint32_t>(network().family()));
  writer.u32(training() ? 1u : 0u);
  writer.u64(optimizer ? optimizer->step_count() : 0);
  writer.u32(optimizer ? 1u : 0u);
  writer.u32(0);
  write_transformer_config(writer, config, transformer_extended(config));
  writer.pad8();
  write_params(writer, params);
  write_bn(writer, {});
  if (optimizer) {
    writer.u32(kKindSgd);
    write_sgd(writer, *optimizer);
  }
  commit_checkpoint(path, writer.bytes());
}

void Transformer::save(const std::filesystem::path& path, const AdamW* optimizer) const {
  // Resume splits parameters() by stored group sizes, so groups must be that order.
  if (optimizer) {
    const std::vector<NamedParam> mine = parameters();
    const std::vector<NamedParam> theirs = flat_params(*optimizer);
    if (mine.size() != theirs.size()) throw Error("optimizer does not belong to this network");
    for (size_t i = 0; i < mine.size(); ++i) {
      if (mine[i].value != theirs[i].value) throw Error("optimizer does not belong to this network");
    }
  }
  ctx_->synchronize();
  const std::vector<NamedParam> params = parameters();
  const TransformerConfig& config = std::get<TransformerConfig>(network().config());

  ByteWriter writer;
  writer.u32(kMagic);
  writer.u32(transformer_file_version(config, true));
  writer.u32(static_cast<uint32_t>(network().family()));
  writer.u32(training() ? 1u : 0u);
  writer.u64(optimizer ? optimizer->step_count() : 0);
  writer.u32(optimizer ? 1u : 0u);
  writer.u32(0);
  write_transformer_config(writer, config, transformer_extended(config));
  writer.pad8();
  write_params(writer, params);
  write_bn(writer, {});
  if (optimizer) write_optimizer_adamw(writer, *optimizer);
  commit_checkpoint(path, writer.bytes());
}

Transformer::Loaded Transformer::load(Context& ctx, const std::filesystem::path& path, LoadMode mode) {
  ByteReader reader(read_file(path));
  const Header header = read_header(reader, kVersionExtAdamW);
  if (header.family != static_cast<uint32_t>(Family::Transformer)) throw Error("bad checkpoint family");
  const auto config_var = read_config(reader, header.family, header.version);
  const TransformerConfig& config = std::get<TransformerConfig>(config_var);
  uint32_t param_count = 0;
  const std::vector<ParsedParam> parsed_params = read_params(reader, &param_count);
  const std::vector<ParsedBn> parsed_bn = read_bn(reader);
  if (!parsed_bn.empty()) throw Error("checkpoint tensor mismatch: batch-norm buffers");
  std::optional<ParsedOpt> parsed_opt;
  std::optional<ParsedAdamW> parsed_adamw;
  if (header.optimizer_present) {
    if (header.version == kVersionAdamW || header.version == kVersionExtAdamW) {
      const uint32_t kind = reader.u32();
      if (kind == kKindSgd) parsed_opt = read_opt(reader, parsed_params);
      else if (kind == kKindAdamW) parsed_adamw = read_adamw(reader, parsed_params);
      else throw Error("unsupported optimizer kind " + std::to_string(kind));
    } else {
      parsed_opt = read_opt(reader, parsed_params);
    }
  }
  if (mode == LoadMode::Resume && !parsed_opt && !parsed_adamw) throw Error("optimizer block missing");

  Transformer model(ctx, config, 0);
  const std::vector<NamedParam> params = model.parameters();
  if (params.size() != parsed_params.size()) throw Error("checkpoint tensor mismatch: count");
  for (size_t i = 0; i < params.size(); ++i) require_match(parsed_params[i], params[i]);
  for (size_t i = 0; i < params.size(); ++i) params[i].value->copy_from_host(parsed_params[i].data);
  if (header.training) model.train();
  else model.eval();

  Loaded loaded{std::move(model), std::nullopt, header.step};
  if (mode == LoadMode::Resume && parsed_opt) {
    Sgd optimizer(loaded.model.parameters(), parsed_opt->learning_rate, parsed_opt->momentum);
    for (size_t i = 0; i < parsed_opt->velocity.size(); ++i) {
      optimizer.velocity()[i].copy_from_host(parsed_opt->velocity[i]);
    }
    optimizer.set_step_count(header.step);
    loaded.optimizer = std::move(optimizer);
  } else if (mode == LoadMode::Resume && parsed_adamw) {
    loaded.adamw = resume_adamw(loaded.model.parameters(), *parsed_adamw, header.step);
  }
  ctx.synchronize();
  return loaded;
}

}  // namespace nn
