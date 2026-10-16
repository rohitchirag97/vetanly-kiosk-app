#include "HybridFaceRecognizer.hpp"

namespace margelo::nitro::facerecognizer {

HybridFaceRecognizer::HybridFaceRecognizer(const FaceRecognizerOptions& options)
    : HybridObject(TAG), _engine(options) {}

EnrollFaceResult HybridFaceRecognizer::enrollFace(
    const std::string& enrollmentId,
    const std::shared_ptr<margelo::nitro::camera::HybridFrameSpec>& frame) {
  return _engine.enrollFace(enrollmentId, frame);
}

void HybridFaceRecognizer::addEnrollment(
    const std::string& enrollmentId,
    const std::shared_ptr<margelo::nitro::ArrayBuffer>& enrollment) {
  _engine.addEnrollment(enrollmentId, enrollment);
}

std::shared_ptr<margelo::nitro::ArrayBuffer>
HybridFaceRecognizer::getEnrollment(const std::string& enrollmentId) {
  return _engine.getEnrollment(enrollmentId);
}

std::vector<std::string> HybridFaceRecognizer::enrollmentIds() {
  return _engine.enrollmentIds();
}

void HybridFaceRecognizer::removeEnrollment(const std::string& enrollmentId) {
  _engine.removeEnrollment(enrollmentId);
}

std::vector<NativeRecognizedFace> HybridFaceRecognizer::recognizeFaces(
    const std::shared_ptr<margelo::nitro::camera::HybridFrameSpec>& frame) {
  return _engine.recognizeFaces(frame);
}

void HybridFaceRecognizer::clearEnrollments() {
  _engine.clearEnrollments();
}

void HybridFaceRecognizer::dispose() {
  _engine.dispose();
}

size_t HybridFaceRecognizer::getExternalMemorySize() noexcept {
  return _engine.externalMemorySize();
}

}  // namespace margelo::nitro::facerecognizer
