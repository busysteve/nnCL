#pragma once

#include <cstdint>
#include <cstring>

namespace nn {

// Float32 is checkpoint dtype tag 1. Activations may be stored as BFloat16.
enum class DType : uint32_t { Float32 = 1, BFloat16 = 2 };

inline uint16_t f32_to_bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t exp = bits & 0x7f800000u;
  if (exp == 0x7f800000u) return static_cast<uint16_t>(bits >> 16);
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

inline float bf16_to_f32(uint16_t bits) {
  const uint32_t wide = static_cast<uint32_t>(bits) << 16;
  float value = 0.f;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

inline int dtype_size(DType dtype) { return dtype == DType::BFloat16 ? 2 : 4; }

}  // namespace nn
