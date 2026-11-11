#include "ocl.hpp"

#include "nn/error.hpp"

#include <cctype>
#include <string>
#include <vector>

namespace nn {
namespace {

std::string trimmed(std::string text) {
  auto space = [](unsigned char c) { return std::isspace(c) != 0; };
  while (!text.empty() && space(static_cast<unsigned char>(text.front()))) text.erase(text.begin());
  while (!text.empty() && space(static_cast<unsigned char>(text.back()))) text.pop_back();
  return text;
}

bool is_auto(const std::string& text) {
  if (text.size() != 4) return false;
  return std::tolower(static_cast<unsigned char>(text[0])) == 'a' &&
         std::tolower(static_cast<unsigned char>(text[1])) == 'u' &&
         std::tolower(static_cast<unsigned char>(text[2])) == 't' &&
         std::tolower(static_cast<unsigned char>(text[3])) == 'o';
}

bool parse_index(const std::string& text, int& index) {
  if (text.empty()) return false;
  int value = 0;
  for (char ch : text) {
    if (ch < '0' || ch > '9') return false;
    const int digit = ch - '0';
    if (value > (2147483647 - digit) / 10) return false;
    value = value * 10 + digit;
  }
  index = value;
  return true;
}

}  // namespace

int device_count() { return opencl_device_count(); }

std::string device_name(int device) { return opencl_device_name(device); }

std::vector<DeviceInfo> detect_devices() {
  const int count = device_count();
  std::vector<DeviceInfo> found;
  found.reserve(static_cast<size_t>(count));
  for (int device = 0; device < count; ++device) {
    found.push_back(DeviceInfo{device, device_name(device)});
  }
  return found;
}

int select_device(const std::string& choice) {
  const std::vector<DeviceInfo> found = detect_devices();
  if (found.empty()) throw Error("device");
  const std::string text = trimmed(choice);
  if (is_auto(text)) return found.front().index;
  int index = 0;
  if (!parse_index(text, index) || index >= static_cast<int>(found.size())) throw Error("device");
  return index;
}

Context::Context() : impl_(std::make_unique<Impl>()) { impl_->open(0); }

Context::Context(int device) : impl_(std::make_unique<Impl>()) { impl_->open(device); }

Context::Context(const std::string& choice) : impl_(std::make_unique<Impl>()) {
  impl_->open(select_device(choice));
}

Context::~Context() = default;

void Context::synchronize() const {
  check_cl(clFinish(impl_->queue), "clFinish");
}

}  // namespace nn
