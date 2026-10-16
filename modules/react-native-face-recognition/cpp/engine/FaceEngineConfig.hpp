#pragma once

#include "CandidateRetrievalMode.hpp"
#include "FaceExecutionProvider.hpp"
#include "FaceRecognizerOptions.hpp"

namespace margelo::nitro::facerecognizer {

/**
 * Validated native runtime configuration.
 *
 * Values in this struct are safe for hot-path use: numeric ranges are checked,
 * optional JS settings are resolved, and integer intervals are already clamped
 * before inference starts.
 */
struct FaceEngineConfig final {
  FaceExecutionProvider executionProvider;
  /**
   * Intra-op threads for the detector, which runs on most frames but is cheap.
   * Kept narrow so it does not compete with the embedding pool.
   */
  int detectorThreads;
  /**
   * Intra-op threads for embedding and liveness — the stages whose latency
   * decides how many camera frames the pipeline has to drop.
   */
  int inferenceThreads;
  int maxFaces;
  float detectionThreshold;
  float detectionMinFaceSize;
  int detectorInputSize;
  float recognitionThreshold;
  float recognitionMinimumMargin;
  CandidateRetrievalMode candidateRetrieval;
  int recognitionSamplesPerDecision;
  int recognitionInterval;
  int matchedRecognitionInterval;
  float recognitionMinFaceSize;
  bool trackingEnabled;
  int detectionInterval;
  bool livenessEnabled;
  float livenessThreshold;
};

FaceEngineConfig validateFaceRecognizerOptions(
    const FaceRecognizerOptions& options);

}  // namespace margelo::nitro::facerecognizer
