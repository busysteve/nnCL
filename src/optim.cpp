#include "nn/optim.hpp"

#include "nn/error.hpp"
#include "ocl.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nn {
namespace {

void require_finite(float value, const char* name) {
  if (!std::isfinite(value)) throw std::invalid_argument(name);
}

}  // namespace

Sgd::Sgd(std::vector<NamedParam> params, float learning_rate, float momentum)
    : params_(std::move(params)), lr_(learning_rate), momentum_(momentum) {
  if (params_.empty()) throw Error("SGD has no parameters");
  Context& ctx = params_.front().value->context();
  velocity_.reserve(params_.size());
  for (const NamedParam& param : params_) {
    velocity_.emplace_back(ctx, param.value->shape());
    velocity_.back().zero();
  }
}

void Sgd::step() {
  Context& ctx = params_.front().value->context();
  for (size_t i = 0; i < params_.size(); ++i) {
    launch_sgd(ctx, params_[i].value->data(), velocity_[i].data(), params_[i].grad->data(),
               params_[i].value->numel(), lr_, momentum_);
  }
  ++step_;
}

AdamW::AdamW(std::vector<NamedParam> params, float lr, float beta1, float beta2, float eps,
             float weight_decay)
    : AdamW(std::vector<AdamWGroup>{AdamWGroup{std::move(params), weight_decay}}, lr, beta1, beta2,
            eps) {}

AdamW::AdamW(std::vector<AdamWGroup> groups, float lr, float beta1, float beta2, float eps)
    : groups_(std::move(groups)), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {
  if (groups_.empty()) throw std::invalid_argument("groups");
  require_finite(lr_, "lr");
  require_finite(beta1_, "beta1");
  require_finite(beta2_, "beta2");
  require_finite(eps_, "eps");
  Context* ctx = nullptr;
  size_t count = 0;
  std::vector<const Tensor*> seen;
  for (const AdamWGroup& group : groups_) {
    if (group.params.empty()) throw std::invalid_argument("params");
    require_finite(group.weight_decay, "weight_decay");
    for (const NamedParam& param : group.params) {
      if (!param.value || !param.grad || param.grad->numel() != param.value->numel()) {
        throw std::invalid_argument("params");
      }
      for (const Tensor* previous : seen) {
        if (previous == param.value) throw std::invalid_argument("params");
      }
      seen.push_back(param.value);
      if (!ctx) ctx = &param.value->context();
      ++count;
    }
  }
  m_.reserve(count);
  v_.reserve(count);
  for (const AdamWGroup& group : groups_) {
    for (const NamedParam& param : group.params) {
      m_.emplace_back(*ctx, param.value->shape());
      v_.emplace_back(*ctx, param.value->shape());
      m_.back().zero();
      v_.back().zero();
    }
  }
}

void AdamW::set_learning_rate(float lr) {
  require_finite(lr, "lr");
  lr_ = lr;
}

void AdamW::step() {
  ++step_;
  const float bc1 =
      1.f - static_cast<float>(std::pow(static_cast<double>(beta1_), static_cast<double>(step_)));
  const float bc2 =
      1.f - static_cast<float>(std::pow(static_cast<double>(beta2_), static_cast<double>(step_)));
  Context& ctx = groups_.front().params.front().value->context();
  size_t index = 0;
  for (const AdamWGroup& group : groups_) {
    for (const NamedParam& param : group.params) {
      launch_adamw(ctx, param.value->data(), m_[index].data(), v_[index].data(), param.grad->data(),
                   param.value->numel(), lr_, beta1_, beta2_, eps_, group.weight_decay, bc1, bc2);
      ++index;
    }
  }
}

float clip_grad_norm(const std::vector<NamedParam>& params, float max_norm) {
  if (!(max_norm > 0.f)) throw std::invalid_argument("max_norm");
  if (params.empty()) return 0.f;
  double sum_sq = 0.0;
  for (const NamedParam& param : params) {
    if (!param.grad) continue;
    sum_sq += launch_sum_squares(param.grad->context(), param.grad->data(), param.grad->numel());
  }
  const float total = static_cast<float>(std::sqrt(sum_sq));
  const float scale = std::min(1.f, max_norm / (total + 1e-6f));
  for (const NamedParam& param : params) {
    if (!param.grad) continue;
    launch_scale(param.grad->context(), param.grad->data(), param.grad->numel(), scale);
  }
  return total;
}

}  // namespace nn
