#pragma once

#include "YuNetDecoder.hpp"
#include "FrameSampler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

namespace margelo::nitro::facerecognizer {

inline Rect axisAlignedBounds(const std::array<Point, 4>& corners) {
  double minX = corners[0].x;
  double minY = corners[0].y;
  double maxX = corners[0].x;
  double maxY = corners[0].y;
  for (size_t index = 1; index < corners.size(); ++index) {
    minX = std::min(minX, corners[index].x);
    minY = std::min(minY, corners[index].y);
    maxX = std::max(maxX, corners[index].x);
    maxY = std::max(maxY, corners[index].y);
  }
  return Rect{minX, minY, maxX - minX, maxY - minY};
}

/**
 * Maps a detector-space YuNet candidate back into frame pixel coordinates.
 *
 * The detector runs on a letterboxed square tensor. This function reverses the
 * letterbox transform, accounts for frame orientation/mirroring through
 * `FrameSampler`, then clamps the result to the frame.
 */
inline Rect mapCandidateBounds(const DetectionCandidate& candidate,
                               const FrameSampler& sampler,
                               const LetterboxTransform& letterbox) {
  const std::array<Point, 4> corners{
      sampler.detectorToFrame(Point{candidate.left, candidate.top}, letterbox),
      sampler.detectorToFrame(
          Point{candidate.left + candidate.width, candidate.top}, letterbox),
      sampler.detectorToFrame(
          Point{candidate.left, candidate.top + candidate.height}, letterbox),
      sampler.detectorToFrame(Point{candidate.left + candidate.width,
                                    candidate.top + candidate.height},
                              letterbox),
  };
  Rect bounds = axisAlignedBounds(corners);
  const double frameWidth = static_cast<double>(sampler.getFrameWidth());
  const double frameHeight = static_cast<double>(sampler.getFrameHeight());
  const double minX = std::clamp(bounds.x, 0.0, frameWidth);
  const double minY = std::clamp(bounds.y, 0.0, frameHeight);
  const double maxX = std::clamp(bounds.x + bounds.width, 0.0, frameWidth);
  const double maxY = std::clamp(bounds.y + bounds.height, 0.0, frameHeight);
  return Rect{minX, minY, maxX - minX, maxY - minY};
}

inline Rect toUprightBounds(const FrameSampler& sampler, const Rect& bounds) {
  const std::array<Point, 4> corners{
      sampler.frameToUpright(Point{bounds.x, bounds.y}),
      sampler.frameToUpright(Point{bounds.x + bounds.width, bounds.y}),
      sampler.frameToUpright(Point{bounds.x, bounds.y + bounds.height}),
      sampler.frameToUpright(
          Point{bounds.x + bounds.width, bounds.y + bounds.height}),
  };
  return axisAlignedBounds(corners);
}

inline void toUprightFace(const FrameSampler& sampler, DetectedFace& face) {
  face.bounds = toUprightBounds(sampler, face.bounds);
  for (Point& landmark : face.landmarks) {
    landmark = sampler.frameToUpright(landmark);
  }
}

inline double pointDistance(const Point& point, const Point& otherPoint) {
  const double dx = point.x - otherPoint.x;
  const double dy = point.y - otherPoint.y;
  return std::sqrt(dx * dx + dy * dy);
}

inline Point midpoint(const Point& point, const Point& otherPoint) {
  return Point{(point.x + otherPoint.x) * 0.5, (point.y + otherPoint.y) * 0.5};
}

/**
 * Returns clockwise in-plane head roll in upright display degrees.
 *
 * The eye line is undirected, so the result is normalized to [-90, 90]. This
 * keeps front-camera mirroring from changing a small tilt into a near-180
 * degree rotation.
 */
inline double headRollAngleDegrees(std::span<const Point> landmarks) {
  if (landmarks.size() != kFaceLandmarkCount) {
    throw std::invalid_argument(
        "FaceRecognizer requires exactly five landmarks for head roll.");
  }

  const Point& leftEye = landmarks[0];
  const Point& rightEye = landmarks[1];
  const double deltaX = rightEye.x - leftEye.x;
  const double deltaY = rightEye.y - leftEye.y;
  if (deltaX * deltaX + deltaY * deltaY <= 1e-12) {
    throw std::invalid_argument(
        "FaceRecognizer cannot calculate head roll from coincident eyes.");
  }

  constexpr double kRadiansToDegrees = 180.0 / std::numbers::pi;
  return std::remainder(std::atan2(deltaY, deltaX) * kRadiansToDegrees, 180.0);
}

/**
 * Rejects landmark sets that cannot produce a reliable aligned face crop.
 *
 * These are detector-quality gates, not identity or demographic rules. A valid
 * face can fail them when the frame is blurred, heavily occluded, strongly
 * turned, or when the five landmarks are misplaced by the detector.
 */
inline bool hasUsableRecognitionGeometry(const Rect& bounds,
                                         std::span<const Point> landmarks) {
  if (landmarks.size() != kFaceLandmarkCount || bounds.width <= 0.0 ||
      bounds.height <= 0.0) {
    return false;
  }

  const Point& leftEye = landmarks[0];
  const Point& rightEye = landmarks[1];
  const Point& nose = landmarks[2];
  const Point& leftMouth = landmarks[3];
  const Point& rightMouth = landmarks[4];
  const double eyeDistance = pointDistance(leftEye, rightEye);
  const double mouthDistance = pointDistance(leftMouth, rightMouth);
  const double faceWidth = bounds.width;
  const double faceHeight = bounds.height;
  if (eyeDistance < faceWidth * kMinRecognitionEyeDistanceRatio ||
      mouthDistance < faceWidth * kMinRecognitionMouthDistanceRatio) {
    return false;
  }

  const Point eyeCenter = midpoint(leftEye, rightEye);
  const Point mouthCenter = midpoint(leftMouth, rightMouth);
  const double verticalSpread = mouthCenter.y - eyeCenter.y;
  if (verticalSpread < faceHeight * kMinRecognitionVerticalSpreadRatio ||
      nose.y <= eyeCenter.y || nose.y >= mouthCenter.y) {
    return false;
  }

  const double mouthEyeRatio = mouthDistance / eyeDistance;
  if (mouthEyeRatio < kMinRecognitionMouthEyeRatio ||
      mouthEyeRatio > kMaxRecognitionMouthEyeRatio) {
    return false;
  }

  const double noseOffsetRatio = std::abs(nose.x - eyeCenter.x) / eyeDistance;
  const double mouthCenterOffsetRatio =
      std::abs(mouthCenter.x - eyeCenter.x) / eyeDistance;
  return noseOffsetRatio <= kMaxRecognitionNoseOffsetRatio &&
         mouthCenterOffsetRatio <= kMaxRecognitionMouthCenterOffsetRatio;
}

inline bool hasUsableRecognitionGeometry(const DetectedFace& uprightFace) {
  return hasUsableRecognitionGeometry(uprightFace.bounds,
                                      uprightFace.landmarks);
}

}  // namespace margelo::nitro::facerecognizer
