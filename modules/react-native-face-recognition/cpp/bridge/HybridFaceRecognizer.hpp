#pragma once

#include "FaceEngine.hpp"
#include "HybridFaceRecognizerSpec.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace margelo::nitro::facerecognizer {

class HybridFaceRecognizer final : public HybridFaceRecognizerSpec {
public:
  explicit HybridFaceRecognizer(const FaceRecognizerOptions& options);

  EnrollFaceResult enrollFace(
      const std::string& enrollmentId,
      const std::shared_ptr<margelo::nitro::camera::HybridFrameSpec>& frame)
      override;
  void addEnrollment(
      const std::string& enrollmentId,
      const std::shared_ptr<margelo::nitro::ArrayBuffer>& enrollment) override;
  std::shared_ptr<margelo::nitro::ArrayBuffer> getEnrollment(
      const std::string& enrollmentId) override;
  std::vector<std::string> enrollmentIds() override;
  void removeEnrollment(const std::string& enrollmentId) override;
  std::vector<NativeRecognizedFace> recognizeFaces(
      const std::shared_ptr<margelo::nitro::camera::HybridFrameSpec>& frame)
      override;
  void clearEnrollments() override;
  void dispose() override;

protected:
  size_t getExternalMemorySize() noexcept override;

private:
  FaceEngine _engine;
};

}  // namespace margelo::nitro::facerecognizer
