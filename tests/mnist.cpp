#include "mnist.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

uint32_t read_be_u32(std::ifstream& in, const std::string& path) {
  unsigned char bytes[4];
  in.read(reinterpret_cast<char*>(bytes), 4);
  if (!in) throw std::runtime_error("truncated idx file: " + path);
  return (static_cast<uint32_t>(bytes[0]) << 24) | (static_cast<uint32_t>(bytes[1]) << 16) |
         (static_cast<uint32_t>(bytes[2]) << 8) | static_cast<uint32_t>(bytes[3]);
}

std::vector<float> read_images(const std::string& path, uint32_t expected) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("missing idx file: " + path);
  const uint32_t magic = read_be_u32(in, path);
  const uint32_t count = read_be_u32(in, path);
  const uint32_t rows = read_be_u32(in, path);
  const uint32_t cols = read_be_u32(in, path);
  if (magic != 2051 || count != expected || rows != 28 || cols != 28) {
    throw std::runtime_error("bad idx images: " + path);
  }
  std::vector<unsigned char> raw(static_cast<size_t>(count) * 28 * 28);
  in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
  if (!in) throw std::runtime_error("truncated idx file: " + path);
  std::vector<float> images(raw.size());
  for (size_t i = 0; i < raw.size(); ++i) images[i] = static_cast<float>(raw[i]) / 255.f;
  return images;
}

std::vector<int64_t> read_labels(const std::string& path, uint32_t expected) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("missing idx file: " + path);
  const uint32_t magic = read_be_u32(in, path);
  const uint32_t count = read_be_u32(in, path);
  if (magic != 2049 || count != expected) throw std::runtime_error("bad idx labels: " + path);
  std::vector<unsigned char> raw(count);
  in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
  if (!in) throw std::runtime_error("truncated idx file: " + path);
  std::vector<int64_t> labels(count);
  for (uint32_t i = 0; i < count; ++i) labels[i] = static_cast<int64_t>(raw[i]);
  return labels;
}

}  // namespace

Mnist load_mnist(const std::string& directory) {
  const std::string root = directory.back() == '/' ? directory : directory + "/";
  Mnist data;
  data.train_images = read_images(root + "train-images-idx3-ubyte", 60000);
  data.train_labels = read_labels(root + "train-labels-idx1-ubyte", 60000);
  data.test_images = read_images(root + "t10k-images-idx3-ubyte", 10000);
  data.test_labels = read_labels(root + "t10k-labels-idx1-ubyte", 10000);
  return data;
}
