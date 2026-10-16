#include "BufferMemory.hpp"

namespace margelo::nitro::facerecognizer {

void clearFloatVector(std::vector<float>& values) noexcept {
  std::vector<float>().swap(values);
}

}  // namespace margelo::nitro::facerecognizer
