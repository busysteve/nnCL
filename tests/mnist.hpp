#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct Mnist {
  std::vector<float> train_images;
  std::vector<int64_t> train_labels;
  std::vector<float> test_images;
  std::vector<int64_t> test_labels;
};

Mnist load_mnist(const std::string& directory);
