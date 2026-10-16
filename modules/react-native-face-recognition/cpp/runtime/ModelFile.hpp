#pragma once

#include <cstddef>
#include <string>

namespace margelo::nitro::facerecognizer {

std::size_t checkedModelFileByteSize(const std::string& path,
                                     const char* modelName);

}  // namespace margelo::nitro::facerecognizer
