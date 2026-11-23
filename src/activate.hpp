#pragma once

#include "nn/tensor.hpp"

#include <cstdint>

namespace nn {

// Bump allocators. Pointers stay valid until the next arena_reset on this context.
// Reset does not free device memory, so a queued kernel may still read the previous bump.
void arena_reset(Context& ctx);
float* arena_f32(Context& ctx, int64_t n);
uint16_t* arena_bf16(Context& ctx, int64_t n);
void zero_f32(Context& ctx, float* dst, int64_t n);

void cast_f32_to_bf16(Context& ctx, uint16_t* dst, const float* src, int64_t n);
void cast_bf16_to_f32(Context& ctx, float* dst, const uint16_t* src, int64_t n);
void load_f32(Context& ctx, float* dst, const Tensor& src);
void store_tensor(Context& ctx, Tensor& dst, const float* src);

// Row-major products. A and B are BFloat16. C is BFloat16 when c_bf16 is set, otherwise Float32.
// Accumulation is Float32. beta == 0 does not read C.
void gemm_nt_bf16(Context& ctx, void* C, bool c_bf16, const uint16_t* A, const uint16_t* B, int M, int N,
                  int K, int batches, int ldc, int lda, int ldb, int64_t stride_c, int64_t stride_a,
                  int64_t stride_b, float alpha, float beta);
void gemm_nn_bf16(Context& ctx, void* C, bool c_bf16, const uint16_t* A, const uint16_t* B, int M, int N,
                  int K, int batches, int ldc, int lda, int ldb, int64_t stride_c, int64_t stride_a,
                  int64_t stride_b, float alpha, float beta);
void gemm_tn_bf16(Context& ctx, void* C, bool c_bf16, const uint16_t* A, const uint16_t* B, int M, int N,
                  int K, int batches, int ldc, int lda, int ldb, int64_t stride_c, int64_t stride_a,
                  int64_t stride_b, float alpha, float beta);

void launch_add_bias_bf16(Context& ctx, uint16_t* y, const float* bias, int rows, int cols);
void launch_bias_grad_bf16(Context& ctx, const uint16_t* grad_y, float* grad_bias, int rows, int cols);

inline float* stage_f32(Context& ctx, const Tensor& src) {
  float* dst = arena_f32(ctx, src.numel());
  load_f32(ctx, dst, src);
  return dst;
}

inline void keep_f32(Context& ctx, Tensor& dst, const float* src) { store_tensor(ctx, dst, src); }

// BFloat16 view. A Float32 tensor is rounded into the BF16 arena.
inline uint16_t* stage_bf16(Context& ctx, const Tensor& src) {
  if (src.dtype() == DType::BFloat16) return const_cast<Tensor&>(src).bf16_data();
  uint16_t* dst = arena_bf16(ctx, src.numel());
  cast_f32_to_bf16(ctx, dst, src.data(), src.numel());
  return dst;
}

}  // namespace nn
