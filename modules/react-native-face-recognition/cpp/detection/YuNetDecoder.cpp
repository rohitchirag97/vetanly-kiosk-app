#include "YuNetDecoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace margelo::nitro::facerecognizer {
namespace {

constexpr std::size_t kBoxValueCount = 4;
constexpr std::size_t kLandmarkValueCount = kFaceLandmarkCount * 2;

float intersectionOverUnion(const DetectionCandidate& candidate,
                            const DetectionCandidate& keptCandidate) {
  const float left = std::max(candidate.left, keptCandidate.left);
  const float top = std::max(candidate.top, keptCandidate.top);
  const float right = std::min(candidate.left + candidate.width,
                               keptCandidate.left + keptCandidate.width);
  const float bottom = std::min(candidate.top + candidate.height,
                                keptCandidate.top + keptCandidate.height);
  const float intersectionWidth = std::max(0.0f, right - left);
  const float intersectionHeight = std::max(0.0f, bottom - top);
  const float intersection = intersectionWidth * intersectionHeight;
  const float unionArea = candidate.width * candidate.height +
                          keptCandidate.width * keptCandidate.height -
                          intersection;
  return unionArea > 0.0f ? intersection / unionArea : 0.0f;
}

}  // namespace

std::vector<DetectionCandidate> decodeYuNetOutputs(
    const std::array<const float*, kYuNetOutputCount>& outputs,
    const YuNetDecodeParams& params) {
  std::vector<DetectionCandidate> candidates;
  candidates.reserve(32);

  const double minFaceSizeInInput =
      static_cast<double>(params.minFaceSize) * params.letterboxScale;
  constexpr std::size_t kOutputsPerGroup = kYuNetStrides.size();
  for (std::size_t level = 0; level < kYuNetStrides.size(); ++level) {
    const int stride = kYuNetStrides[level];
    const int grid = params.inputSize / stride;
    const float* classScores = outputs[level];
    const float* objectnessScores = outputs[level + kOutputsPerGroup];
    const float* boxDeltas = outputs[level + 2 * kOutputsPerGroup];
    const float* landmarkDeltas = outputs[level + 3 * kOutputsPerGroup];

    for (int row = 0; row < grid; ++row) {
      for (int col = 0; col < grid; ++col) {
        const std::size_t location =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(grid) +
            static_cast<std::size_t>(col);
        // YuNet combines class and objectness scores with a geometric mean.
        const float confidence =
            std::sqrt(std::clamp(classScores[location], 0.0f, 1.0f) *
                      std::clamp(objectnessScores[location], 0.0f, 1.0f));
        if (confidence < params.detectionThreshold) {
          continue;
        }

        // YuNet encodes boxes relative to the grid cell and stride.
        const float* box = boxDeltas + location * kBoxValueCount;
        const float centerX = (col + box[0]) * stride;
        const float centerY = (row + box[1]) * stride;
        const float width = std::exp(box[2]) * stride;
        const float height = std::exp(box[3]) * stride;
        if (static_cast<double>(width) < minFaceSizeInInput ||
            static_cast<double>(height) < minFaceSizeInInput) {
          continue;
        }

        DetectionCandidate candidate{
            centerX - width * 0.5f,
            centerY - height * 0.5f,
            width,
            height,
            confidence,
            {},
        };
        const float* landmarkValues =
            landmarkDeltas + location * kLandmarkValueCount;
        for (std::size_t landmark = 0; landmark < kFaceLandmarkCount;
             ++landmark) {
          candidate.landmarks[landmark] =
              Point{(landmarkValues[landmark * 2] + col) * stride,
                    (landmarkValues[landmark * 2 + 1] + row) * stride};
        }
        candidates.push_back(candidate);
      }
    }
  }
  return candidates;
}

std::vector<DetectionCandidate> suppressOverlaps(
    std::vector<DetectionCandidate> candidates,
    float nmsThreshold,
    int maxFaces) {
  std::sort(candidates.begin(),
            candidates.end(),
            [](const DetectionCandidate& candidate,
               const DetectionCandidate& otherCandidate) {
              return candidate.confidence > otherCandidate.confidence;
            });

  std::vector<DetectionCandidate> kept;
  kept.reserve(std::min(candidates.size(), static_cast<std::size_t>(maxFaces)));
  for (const DetectionCandidate& candidate : candidates) {
    const bool overlaps = std::any_of(
        kept.begin(), kept.end(), [&](const DetectionCandidate& keptCandidate) {
          return intersectionOverUnion(candidate, keptCandidate) >=
                 nmsThreshold;
        });
    if (overlaps) {
      continue;
    }
    kept.push_back(candidate);
    if (kept.size() == static_cast<std::size_t>(maxFaces)) {
      break;
    }
  }
  return kept;
}

}  // namespace margelo::nitro::facerecognizer
