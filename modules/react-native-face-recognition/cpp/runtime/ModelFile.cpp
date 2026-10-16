#include "ModelFile.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

namespace margelo::nitro::facerecognizer {

std::size_t checkedModelFileByteSize(const std::string& path,
                                     const char* modelName) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error(
        "FaceRecognizer could not open bundled " + std::string(modelName) +
        " model for memory accounting at path '" + path + "'.");
  }
  const std::ifstream::pos_type position = file.tellg();
  if (position < 0) {
    throw std::runtime_error("FaceRecognizer could not read bundled " +
                             std::string(modelName) + " model size at path '" +
                             path + "'.");
  }
  return static_cast<std::size_t>(position);
}

}  // namespace margelo::nitro::facerecognizer
