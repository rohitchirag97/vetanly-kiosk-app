#pragma once

#include <string>

namespace margelo::nitro::facerecognizer {

std::string bundledDetectorModelPath();
std::string bundledRecognizerModelPath();
std::string bundledLivenessModelPath(const std::string& baseName);

}  // namespace margelo::nitro::facerecognizer
