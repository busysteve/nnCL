#pragma once

#include <memory>
#include <string>

namespace nn {

// Devices this build accepts. auto counts GPUs. intel can count a CPU device.
int device_count();
std::string device_name(int device);

class Context {
 public:
  Context();
  // Index into device_count()'s list. Throws device when the index is outside it.
  explicit Context(int device);
  ~Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  Context(Context&&) = delete;
  Context& operator=(Context&&) = delete;

  void synchronize() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend struct ContextAccess;
};

}  // namespace nn
