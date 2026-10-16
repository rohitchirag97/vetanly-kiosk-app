#include "FaceEngineConfig.hpp"

#include "FaceConstants.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace margelo::nitro::facerecognizer {
namespace {

/** The detector is cheap per frame; a wider pool costs more than it returns. */
constexpr int kMaxDetectorThreads = 2;

int checkedInteger(double value, int minimum, int maximum, const char* name) {
  if (!std::isfinite(value) || std::trunc(value) != value || value < minimum ||
      value > maximum) {
    throw std::invalid_argument(
        "FaceRecognizer option '" + std::string(name) +
        "' must be an integer between " + std::to_string(minimum) + " and " +
        std::to_string(maximum) + "; received " + std::to_string(value) + ".");
  }
  return static_cast<int>(value);
}

int checkedDetectorInputSize(double requestedSize, const char* name) {
  if (!std::isfinite(requestedSize) ||
      std::trunc(requestedSize) != requestedSize ||
      requestedSize < kMinDetectorInputSize ||
      requestedSize > kMaxDetectorInputSize ||
      static_cast<int>(requestedSize) % kDetectorInputMultiple != 0) {
    throw std::invalid_argument(
        "FaceRecognizer option '" + std::string(name) +
        "' must be a multiple of " + std::to_string(kDetectorInputMultiple) +
        " between " + std::to_string(kMinDetectorInputSize) + " and " +
        std::to_string(kMaxDetectorInputSize) + "; received " +
        std::to_string(requestedSize) + ".");
  }
  return static_cast<int>(requestedSize);
}

float checkedUnitInterval(double value, const char* name) {
  if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
    throw std::invalid_argument("FaceRecognizer option '" + std::string(name) +
                                "' must be between 0 and 1; received " +
                                std::to_string(value) + ".");
  }
  return static_cast<float>(value);
}

CandidateRetrievalMode checkedCandidateRetrieval(CandidateRetrieval value) {
  switch (value) {
    case CandidateRetrieval::CENTROID:
      return CandidateRetrievalMode::CENTROID;
    case CandidateRetrieval::TEMPLATEMAX:
      return CandidateRetrievalMode::TEMPLATE_MAX;
  }
  throw std::invalid_argument(
      "FaceRecognizer option 'recognition.candidateRetrieval' is invalid.");
}

}  // namespace

FaceEngineConfig validateFaceRecognizerOptions(
    const FaceRecognizerOptions& options) {
  const int inferenceThreads = checkedInteger(options.threads, 1, 6, "threads");
  const int maxFaces =
      checkedInteger(options.detection.maxFaces, 1, 100, "detection.maxFaces");
  if (!std::isfinite(options.detection.minFaceSize) ||
      options.detection.minFaceSize < 1.0) {
    throw std::invalid_argument(
        "FaceRecognizer option 'detection.minFaceSize' must be at least 1 "
        "pixel; received " +
        std::to_string(options.detection.minFaceSize) + ".");
  }
  if (!std::isfinite(options.recognition.minFaceSize) ||
      options.recognition.minFaceSize < 1.0) {
    throw std::invalid_argument(
        "FaceRecognizer option 'recognition.minFaceSize' must be at least 1 "
        "pixel; received " +
        std::to_string(options.recognition.minFaceSize) + ".");
  }
  return FaceEngineConfig{
      .executionProvider = options.provider,
      .detectorThreads = std::min(inferenceThreads, kMaxDetectorThreads),
      .inferenceThreads = inferenceThreads,
      .maxFaces = maxFaces,
      .detectionThreshold = checkedUnitInterval(options.detection.threshold,
                                                "detection.threshold"),
      .detectionMinFaceSize = static_cast<float>(options.detection.minFaceSize),
      .detectorInputSize = checkedDetectorInputSize(options.detection.inputSize,
                                                    "detection.inputSize"),
      .recognitionThreshold = checkedUnitInterval(options.recognition.threshold,
                                                  "recognition.threshold"),
      .recognitionMinimumMargin = checkedUnitInterval(
          options.recognition.minimumMargin, "recognition.minimumMargin"),
      .candidateRetrieval =
          checkedCandidateRetrieval(options.recognition.candidateRetrieval),
      .recognitionSamplesPerDecision =
          checkedInteger(options.recognition.samplesPerDecision,
                         1,
                         5,
                         "recognition.samplesPerDecision"),
      .recognitionInterval = checkedInteger(
          options.recognition.interval, 1, 300, "recognition.interval"),
      .matchedRecognitionInterval =
          checkedInteger(options.recognition.matchedInterval,
                         1,
                         600,
                         "recognition.matchedInterval"),
      .recognitionMinFaceSize =
          static_cast<float>(options.recognition.minFaceSize),
      .trackingEnabled = options.tracking.enabled,
      .detectionInterval = checkedInteger(options.tracking.detectionInterval,
                                          1,
                                          300,
                                          "tracking.detectionInterval"),
      .livenessEnabled = options.liveness.enabled,
      .livenessThreshold =
          checkedUnitInterval(options.liveness.threshold, "liveness.threshold"),
  };
}

}  // namespace margelo::nitro::facerecognizer
