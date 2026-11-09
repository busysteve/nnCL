#include "ocl.hpp"

namespace nn {

Context::Context() : impl_(std::make_unique<Impl>()) { impl_->open(); }

Context::~Context() = default;

void Context::synchronize() const {
  check_cl(clFinish(impl_->queue), "clFinish");
}

}  // namespace nn
