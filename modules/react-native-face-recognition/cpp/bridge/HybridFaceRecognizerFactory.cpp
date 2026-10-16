#include "HybridFaceRecognizerFactory.hpp"

#include "HybridFaceRecognizer.hpp"

namespace margelo::nitro::facerecognizer {

HybridFaceRecognizerFactory::HybridFaceRecognizerFactory()
    : HybridObject(TAG) {}

std::shared_ptr<Promise<std::shared_ptr<HybridFaceRecognizerSpec>>>
HybridFaceRecognizerFactory::create(const FaceRecognizerOptions& options) {
  return Promise<std::shared_ptr<HybridFaceRecognizerSpec>>::async(
      [options]() -> std::shared_ptr<HybridFaceRecognizerSpec> {
        return std::make_shared<HybridFaceRecognizer>(options);
      });
}

}  // namespace margelo::nitro::facerecognizer
