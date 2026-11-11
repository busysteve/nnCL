#include "ocl.hpp"

namespace nn {

int device_count() { return opencl_device_count(); }

std::string device_name(int device) { return opencl_device_name(device); }

Context::Context() : impl_(std::make_unique<Impl>()) { impl_->open(0); }

Context::Context(int device) : impl_(std::make_unique<Impl>()) { impl_->open(device); }

Context::~Context() = default;

void Context::synchronize() const {
  check_cl(clFinish(impl_->queue), "clFinish");
}

}  // namespace nn
