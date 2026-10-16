#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "FaceEngine.hpp"
#include "FaceGeometry.hpp"
#include "FrameSampler.hpp"

namespace margelo::nitro::facerecognizer {

EnrollFaceResult FaceEngine::enrollFace(
    const std::string& enrollmentId,
    const std::shared_ptr<margelo::nitro::camera::HybridFrameSpec>& frame) {
  validateEnrollmentId(enrollmentId);

  std::lock_guard<std::mutex> lock(_inferenceMutex);
  ensureActiveLocked();
  const FrameSampler sampler(frame);
  std::vector<DetectedFace> detectedFaces =
      _detector.detect(sampler, _config, std::max(_config.maxFaces, 2));
  if (_config.trackingEnabled) {
    _tracker.resetWithFrameFaces(sampler, detectedFaces);
  }

  if (detectedFaces.empty()) {
    releaseIdleModelsLocked();
    return EnrollFaceResult{EnrollFaceStatus::NOFACE, std::nullopt};
  }
  if (detectedFaces.size() != 1) {
    releaseIdleModelsLocked();
    return EnrollFaceResult{EnrollFaceStatus::MULTIPLEFACES, std::nullopt};
  }

  const DetectedFace& detectedFace = detectedFaces.front();
  const Rect uprightBounds = toUprightBounds(sampler, detectedFace.bounds);
  std::array<Point, kFaceLandmarkCount> uprightLandmarks{};
  for (std::size_t index = 0; index < kFaceLandmarkCount; ++index) {
    uprightLandmarks[index] =
        sampler.frameToUpright(detectedFace.landmarks[index]);
  }
  if (!hasUsableRecognitionGeometry(uprightBounds, uprightLandmarks)) {
    releaseIdleModelsLocked();
    return EnrollFaceResult{EnrollFaceStatus::LOWQUALITY, std::nullopt};
  }

  // Never enroll a spoof: a poisoned template would let the attack pass
  // recognition forever. Gate before computing the embedding so we don't store
  // anything for a rejected frame.
  if (_config.livenessEnabled) {
    _liveness.ensureLoaded(_config, _memoryInfo);
    if (_liveness.score(sampler, uprightBounds) < _config.livenessThreshold) {
      releaseIdleModelsLocked();
      return EnrollFaceResult{EnrollFaceStatus::SPOOF, std::nullopt};
    }
  }

  _recognizer.ensureLoaded(_config, _memoryInfo, _enrollments.index());
  FaceEmbedding faceEmbedding =
      _recognizer.embed(sampler, detectedFaces.front());

  const EnrollFaceStatus status =
      _enrollments.addSample(enrollmentId, std::move(faceEmbedding));
  _recentMatches.clear();
  _pendingRecognitions.clear();
  _framesSinceRecognition = _config.recognitionInterval;
  return EnrollFaceResult{status, _enrollments.exportSerialized(enrollmentId)};
}

void FaceEngine::addEnrollment(
    const std::string& enrollmentId,
    const std::shared_ptr<margelo::nitro::ArrayBuffer>& enrollment) {
  validateEnrollmentId(enrollmentId);

  std::lock_guard<std::mutex> lock(_inferenceMutex);
  ensureActiveLocked();
  // When the recognizer is already resident we can validate the template length
  // eagerly for a clearer error. Otherwise recognizer loading validates later,
  // so app startup does not force the large model resident just to rehydrate
  // templates.
  const std::optional<std::size_t> expectedEmbeddingLength =
      _recognizer.isLoaded()
          ? std::optional<std::size_t>(_recognizer.embeddingLength())
          : std::nullopt;
  _enrollments.importSerialized(
      enrollmentId, enrollment, expectedEmbeddingLength);
  _recentMatches.clear();
  _pendingRecognitions.clear();
  _framesSinceRecognition = _config.recognitionInterval;
}

std::shared_ptr<margelo::nitro::ArrayBuffer> FaceEngine::getEnrollment(
    const std::string& enrollmentId) {
  std::lock_guard<std::mutex> lock(_inferenceMutex);
  ensureActiveLocked();
  return _enrollments.exportSerialized(enrollmentId);
}

std::vector<std::string> FaceEngine::enrollmentIds() {
  std::lock_guard<std::mutex> lock(_inferenceMutex);
  ensureActiveLocked();
  return _enrollments.ids();
}

void FaceEngine::removeEnrollment(const std::string& enrollmentId) {
  std::lock_guard<std::mutex> lock(_inferenceMutex);
  ensureActiveLocked();
  _enrollments.remove(enrollmentId);
  _recentMatches.clear();
  _pendingRecognitions.clear();
  _framesSinceRecognition = _config.recognitionInterval;
  releaseIdleModelsLocked();
}

void FaceEngine::clearEnrollments() {
  std::lock_guard<std::mutex> lock(_inferenceMutex);
  ensureActiveLocked();
  _enrollments.clear();
  _recentMatches.clear();
  _pendingRecognitions.clear();
  _framesSinceRecognition = _config.recognitionInterval;
  releaseIdleModelsLocked();
}

}  // namespace margelo::nitro::facerecognizer
