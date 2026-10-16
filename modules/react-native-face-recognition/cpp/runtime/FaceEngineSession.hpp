#pragma once

#include "FaceExecutionProvider.hpp"

#include "onnxruntime_cxx_api.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace margelo::nitro::facerecognizer {

struct SessionDimensionOverride final {
  std::string name;
  std::int64_t value;
};

std::unique_ptr<Ort::Session> createSession(
    const std::string& modelPath,
    int inferenceThreads,
    FaceExecutionProvider provider,
    bool useSharedArena,
    std::span<const SessionDimensionOverride> dimensionOverrides);

}  // namespace margelo::nitro::facerecognizer
