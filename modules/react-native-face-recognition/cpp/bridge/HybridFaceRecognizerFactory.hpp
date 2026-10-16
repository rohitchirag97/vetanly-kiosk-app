#pragma once

#include "HybridFaceRecognizerFactorySpec.hpp"

namespace margelo::nitro::facerecognizer {

class HybridFaceRecognizerFactory final
    : public HybridFaceRecognizerFactorySpec {
public:
  HybridFaceRecognizerFactory();

  std::shared_ptr<Promise<std::shared_ptr<HybridFaceRecognizerSpec>>> create(
      const FaceRecognizerOptions& options) override;
};

}  // namespace margelo::nitro::facerecognizer
