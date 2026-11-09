#pragma once

#include <memory>

namespace nn {

class Context {
 public:
  Context();
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
