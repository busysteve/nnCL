#include "check.hpp"
#include "mnist.hpp"

#include "nn/builders.hpp"
#include "nn/loss.hpp"
#include "nn/conv_resnet.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kTrainBatch = 128;
constexpr int kEvalBatch = 256;
constexpr int kImages = 60000;
constexpr int kTests = 10000;
constexpr int kPixels = 784;
constexpr uint32_t kSeed = 1;

struct BatchTensors {
  nn::Tensor images;
  nn::Tensor flat;
  nn::DeviceIndices labels;
  bool rank2 = false;

  BatchTensors(nn::Context& ctx, int n, bool as_rank2)
      : images(ctx, {n, 1, 28, 28}),
        flat(ctx, {n, kPixels}),
        labels(ctx, n),
        rank2(as_rank2) {}

  nn::Tensor& input() { return rank2 ? flat : images; }
};

void fill_batch(BatchTensors& batch, const std::vector<float>& images, const std::vector<int64_t>& labels,
                const std::vector<int>& indices, int offset, int count) {
  std::vector<float> host_x(static_cast<size_t>(count) * kPixels);
  std::vector<int64_t> host_y(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    const int index = indices[offset + i];
    const float* src = images.data() + static_cast<size_t>(index) * kPixels;
    std::copy(src, src + kPixels, host_x.begin() + static_cast<size_t>(i) * kPixels);
    host_y[i] = labels[index];
  }
  if (batch.rank2) {
    batch.images.copy_from_host(host_x);
    batch.flat.copy_from(batch.images);
  } else {
    batch.images.copy_from_host(host_x);
  }
  batch.labels.copy_from_host(host_y.data(), count);
}

double train_epoch(nn::Network& net, nn::Sgd& optimizer, BatchTensors& full, BatchTensors& tail,
                   const Mnist& data, std::mt19937& rng) {
  net.train();
  std::vector<int> indices(kImages);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), rng);
  double total = 0.0;
  int seen = 0;
  for (int offset = 0; offset < kImages;) {
    const int count = std::min(kTrainBatch, kImages - offset);
    BatchTensors& batch = count == kTrainBatch ? full : tail;
    fill_batch(batch, data.train_images, data.train_labels, indices, offset, count);
    net.zero_grad();
    const nn::Tensor& logits = net.forward(batch.input());
    nn::LossResult loss = nn::softmax_cross_entropy(logits.context(), logits, batch.labels);
    total += static_cast<double>(loss.loss) * count;
    seen += count;
    net.backward(loss.grad_logits);
    optimizer.step();
    offset += count;
    if ((offset / kTrainBatch) % 100 == 0) {
      std::printf("  seen %d loss %.4f\n", seen, total / static_cast<double>(seen));
      std::fflush(stdout);
    }
  }
  return total / static_cast<double>(seen);
}

int count_correct(nn::Network& net, BatchTensors& full, BatchTensors& tail, const std::vector<float>& images,
                  const std::vector<int64_t>& labels, int total) {
  std::vector<int> indices(static_cast<size_t>(total));
  std::iota(indices.begin(), indices.end(), 0);
  int correct = 0;
  for (int offset = 0; offset < total;) {
    const int count = std::min(kEvalBatch, total - offset);
    BatchTensors& batch = count == kEvalBatch ? full : tail;
    fill_batch(batch, images, labels, indices, offset, count);
    const std::vector<float> logits = net.forward(batch.input()).to_host();
    const int classes = static_cast<int>(logits.size() / static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
      int best = 0;
      float best_value = logits[static_cast<size_t>(i) * classes];
      for (int c = 1; c < classes; ++c) {
        const float value = logits[static_cast<size_t>(i) * classes + c];
        if (value > best_value) {
          best_value = value;
          best = c;
        }
      }
      if (best == labels[indices[offset + i]]) ++correct;
    }
    offset += count;
  }
  return correct;
}

}  // namespace

int test_mnist() {
  const char* directory = std::getenv("MNIST_DIR");
  if (directory == nullptr || directory[0] == '\0') {
    std::fprintf(stderr,
                 "MNIST_DIR is not set. Expected files:\n"
                 "train-images-idx3-ubyte\n"
                 "train-labels-idx1-ubyte\n"
                 "t10k-images-idx3-ubyte\n"
                 "t10k-labels-idx1-ubyte\n");
    return 1;
  }
  Mnist data;
  try {
    data = load_mnist(directory);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\nExpected files:\ntrain-images-idx3-ubyte\ntrain-labels-idx1-ubyte\n"
                         "t10k-images-idx3-ubyte\n"
                         "t10k-labels-idx1-ubyte\n",
                 error.what());
    return 1;
  }

  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "nn-mnist";
  std::filesystem::create_directories(dir);

  auto finish = [&](const char* title, nn::Network net, bool rank2, int epochs, int floor,
                    float learning_rate, const std::filesystem::path& path) {
    nn::Context& ctx = net.parameters().front().value->context();
    nn::Sgd optimizer(net.parameters(), learning_rate, 0.9f);
    BatchTensors train_full(ctx, kTrainBatch, rank2);
    BatchTensors train_tail(ctx, kImages % kTrainBatch, rank2);
    BatchTensors eval_full(ctx, kEvalBatch, rank2);
    BatchTensors eval_tail(ctx, kTests % kEvalBatch, rank2);
    for (int epoch = 0; epoch < epochs; ++epoch) {
      if (epoch + 1 == epochs) {
        for (nn::NamedBuffer buffer : net.buffers()) {
          if (buffer.name.size() >= 12 &&
              buffer.name.compare(buffer.name.size() - 12, 12, "running_mean") == 0) {
            buffer.value->zero();
          }
          if (buffer.name.size() >= 11 &&
              buffer.name.compare(buffer.name.size() - 11, 11, "running_var") == 0) {
            buffer.value->fill(1.f);
            *buffer.batches = 0;
          }
        }
        if (!net.buffers().empty()) std::printf("%s reset batch-norm running stats\n", title);
      }
      std::mt19937 rng(kSeed + static_cast<uint32_t>(epoch));
      const double loss = train_epoch(net, optimizer, train_full, train_tail, data, rng);
      std::printf("%s epoch %d loss %.6f\n", title, epoch + 1, loss);
      std::fflush(stdout);
      if (epoch == 0) CHECK(std::isfinite(loss));
    }
    net.eval();
    const int correct = count_correct(net, eval_full, eval_tail, data.test_images, data.test_labels, kTests);
    std::printf("%s test correct %d / %d\n", title, correct, kTests);
    CHECK(correct >= floor);
    net.save(path, nullptr);
    nn::Context loaded_ctx;
    nn::Network::Loaded loaded = nn::Network::load(loaded_ctx, path, nn::LoadMode::Inference);
    CHECK(!loaded.optimizer.has_value());
    CHECK(!loaded.network.training());
    BatchTensors loaded_full(loaded_ctx, kEvalBatch, rank2);
    BatchTensors loaded_tail(loaded_ctx, kTests % kEvalBatch, rank2);
    const int again =
        count_correct(loaded.network, loaded_full, loaded_tail, data.test_images, data.test_labels, kTests);
    std::printf("%s reloaded correct %d / %d\n", title, again, kTests);
    CHECK(again == correct);
    CHECK(again >= floor);
  };

  if(0){
    nn::Context ctx;
    nn::MlpConfig config;
    config.input_size = 784;
    config.hidden = {256, 128};
    config.output_size = 10;
    finish("mlp", nn::build_mlp(ctx, config, kSeed), true, 5, 9600, 0.1f, dir / "mlp.bin");
  }
  if(0){
    nn::Context ctx;
    nn::ConvConfig config;
    config.in_channels = 1;
    config.in_height = 28;
    config.in_width = 28;
    config.convs = {nn::ConvSpec{32, 5, 1, 2}, nn::ConvSpec{64, 5, 1, 2}};
    config.pool = nn::PoolSpec{2, 2};
    config.fc = {128, 10};
    finish("conv", nn::build_conv(ctx, config, kSeed), false, 5, 9800, 0.1f, dir / "conv.bin");
  }
  if(0){
    nn::Context ctx;
    nn::ResNetConfig config;
    config.in_channels = 1;
    config.in_height = 28;
    config.in_width = 28;
    config.stem_channels = 16;
    config.blocks_per_stage = {2, 2, 2};
    config.stage_widths = {16, 32, 64};
    config.output_size = 10;
    finish("resnet", nn::build_resnet(ctx, config, kSeed), false, 8, 9800, 0.1f, dir / "resnet.bin");
  }
  if(1){
    nn::Context ctx;
    nn::ConvResNetConfig config;
    config.in_channels = 1;
    config.in_height = 28;
    config.in_width = 28;
    config.convs = {nn::ConvSpec{64, 5, 1, 2}, nn::ConvSpec{64, 5, 1, 2}};
    config.pool = nn::PoolSpec{2, 2};
    config.stem_channels = 8;
    config.blocks_per_stage = {2, 2, 2};
    config.stage_widths = {32, 64, 128};
    config.fc = {128, 10};
    finish("convresnet", nn::build_conv_resnet(ctx, config, kSeed), false, 5, 9800, 0.1f, dir / "convresnet.bin");
  }


  std::printf("mnist ok\n");
  return 0;
}
