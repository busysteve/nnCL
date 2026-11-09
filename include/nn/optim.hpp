#pragma once

#include "nn/layer.hpp"

#include <cstdint>
#include <vector>

namespace nn {

class Sgd {
 public:
  Sgd(std::vector<NamedParam> params, float learning_rate, float momentum);
  Sgd(Sgd&&) noexcept = default;
  Sgd& operator=(Sgd&&) noexcept = default;
  Sgd(const Sgd&) = delete;
  Sgd& operator=(const Sgd&) = delete;

  void step();
  void set_learning_rate(float learning_rate) { lr_ = learning_rate; }
  float learning_rate() const { return lr_; }
  float momentum() const { return momentum_; }
  uint64_t step_count() const { return step_; }
  void set_step_count(uint64_t step) { step_ = step; }

  const std::vector<Tensor>& velocity() const { return velocity_; }
  std::vector<Tensor>& velocity() { return velocity_; }
  const std::vector<NamedParam>& params() const { return params_; }

 private:
  std::vector<NamedParam> params_;
  std::vector<Tensor> velocity_;
  float lr_ = 0.f;
  float momentum_ = 0.f;
  uint64_t step_ = 0;
};

struct AdamWGroup {
  std::vector<NamedParam> params;
  float weight_decay = 0.f;
};

class AdamW {
 public:
  AdamW(std::vector<AdamWGroup> groups, float lr, float beta1, float beta2, float eps);
  AdamW(std::vector<NamedParam> params, float lr, float beta1, float beta2, float eps,
        float weight_decay);
  AdamW(AdamW&&) noexcept = default;
  AdamW& operator=(AdamW&&) noexcept = default;
  AdamW(const AdamW&) = delete;
  AdamW& operator=(const AdamW&) = delete;

  void step();
  void set_learning_rate(float lr);
  float learning_rate() const { return lr_; }
  float beta1() const { return beta1_; }
  float beta2() const { return beta2_; }
  float eps() const { return eps_; }
  uint64_t step_count() const { return step_; }
  void set_step_count(uint64_t step) { step_ = step; }

  const std::vector<AdamWGroup>& groups() const { return groups_; }
  const std::vector<Tensor>& m() const { return m_; }
  std::vector<Tensor>& m() { return m_; }
  const std::vector<Tensor>& v() const { return v_; }
  std::vector<Tensor>& v() { return v_; }

 private:
  std::vector<AdamWGroup> groups_;
  std::vector<Tensor> m_;
  std::vector<Tensor> v_;
  float lr_ = 0.f;
  float beta1_ = 0.f;
  float beta2_ = 0.f;
  float eps_ = 0.f;
  uint64_t step_ = 0;
};

// Total L2 norm of grads before scaling. Null grads are skipped.
float clip_grad_norm(const std::vector<NamedParam>& params, float max_norm);

}  // namespace nn
