#pragma once

#include "FaceConstants.hpp"
#include "Point.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace margelo::nitro::facerecognizer {

/** Decoded YuNet candidate before NMS and frame-coordinate mapping. */
struct DetectionCandidate final {
  float left;
  float top;
  float width;
  float height;
  float confidence;
  std::array<Point, kFaceLandmarkCount> landmarks;
};

/** Runtime parameters needed to decode one YuNet output set. */
struct YuNetDecodeParams final {
  int inputSize;
  float detectionThreshold;
  float minFaceSize;
  double letterboxScale;
};

/** Decodes raw YuNet output tensors into detector-space candidates. */
std::vector<DetectionCandidate> decodeYuNetOutputs(
    const std::array<const float*, kYuNetOutputCount>& outputs,
    const YuNetDecodeParams& params);

/** Applies NMS, sorts by confidence, and caps the returned face count. */
std::vector<DetectionCandidate> suppressOverlaps(
    std::vector<DetectionCandidate> candidates,
    float nmsThreshold,
    int maxFaces);

}  // namespace margelo::nitro::facerecognizer
