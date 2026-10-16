#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace margelo::nitro::facerecognizer {

template <typename T>
std::size_t vectorCapacityByteSize(const std::vector<T>& values) noexcept {
  return values.capacity() * sizeof(T);
}

template <typename T, std::size_t Size>
std::size_t vectorArrayCapacityByteSize(
    const std::array<std::vector<T>, Size>& values) noexcept {
  std::size_t byteSize = 0;
  for (const std::vector<T>& entry : values) {
    byteSize += vectorCapacityByteSize(entry);
  }
  return byteSize;
}

// A loaded model is resident roughly twice over: ORT keeps the initializer
// weights it read from the file plus its own optimized in-memory graph. This
// estimate feeds getExternalMemorySize() so the JS GC senses that pressure.
inline std::size_t modelResidencyEstimate(
    std::size_t modelFileByteSize) noexcept {
  return modelFileByteSize * 2;
}

void clearFloatVector(std::vector<float>& values) noexcept;

}  // namespace margelo::nitro::facerecognizer
