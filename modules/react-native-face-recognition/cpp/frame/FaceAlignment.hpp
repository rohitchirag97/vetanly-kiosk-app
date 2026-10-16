#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>

#include "DetectedFace.hpp"
#include "FaceConstants.hpp"

namespace margelo::nitro::facerecognizer {

/**
 * Inverse similarity transform from aligned recognizer pixels to frame
 * coordinates.
 */
struct SimilarityTransform final {
  double inverseScaleCosine;
  double inverseScaleSine;
  double translateX;
  double translateY;
};

/**
 * Builds an ArcFace-style alignment transform from five frame-space landmarks.
 *
 * The returned transform maps aligned 112x112 crop pixels back to frame
 * coordinates. Mirroring flips the target template so front-camera output stays
 * consistent with the displayed preview.
 */
inline SimilarityTransform createFaceAlignmentTransform(
    std::span<const Point> landmarks, bool mirrored) {
  if (landmarks.size() != kTargetLandmarks.size()) {
    throw std::invalid_argument(
        "FaceRecognizer requires exactly five landmarks for alignment.");
  }

  const auto targetXForLandmark = [mirrored](std::size_t index) {
    const double x = kTargetLandmarks[index][0];
    return mirrored ? kAlignedFaceSize - x : x;
  };

  double frameMeanX = 0.0;
  double frameMeanY = 0.0;
  double targetMeanX = 0.0;
  double targetMeanY = 0.0;
  for (std::size_t index = 0; index < landmarks.size(); ++index) {
    frameMeanX += landmarks[index].x;
    frameMeanY += landmarks[index].y;
    targetMeanX += targetXForLandmark(index);
    targetMeanY += kTargetLandmarks[index][1];
  }
  frameMeanX /= static_cast<double>(landmarks.size());
  frameMeanY /= static_cast<double>(landmarks.size());
  targetMeanX /= static_cast<double>(landmarks.size());
  targetMeanY /= static_cast<double>(landmarks.size());

  double cosineNumerator = 0.0;
  double sineNumerator = 0.0;
  double frameSquaredDistanceSum = 0.0;
  for (std::size_t index = 0; index < landmarks.size(); ++index) {
    const double frameX = landmarks[index].x - frameMeanX;
    const double frameY = landmarks[index].y - frameMeanY;
    const double centeredTargetX = targetXForLandmark(index) - targetMeanX;
    const double centeredTargetY = kTargetLandmarks[index][1] - targetMeanY;
    cosineNumerator += frameX * centeredTargetX + frameY * centeredTargetY;
    sineNumerator += frameX * centeredTargetY - frameY * centeredTargetX;
    frameSquaredDistanceSum += frameX * frameX + frameY * frameY;
  }
  if (frameSquaredDistanceSum <= 1e-12) {
    throw std::invalid_argument(
        "FaceRecognizer cannot align coincident face landmarks.");
  }

  const double scaleCosine = cosineNumerator / frameSquaredDistanceSum;
  const double scaleSine = sineNumerator / frameSquaredDistanceSum;

  const double squaredScale = scaleCosine * scaleCosine + scaleSine * scaleSine;
  if (squaredScale <= 1e-12) {
    throw std::invalid_argument(
        "FaceRecognizer received a singular face alignment transform.");
  }
  return SimilarityTransform{
      scaleCosine / squaredScale,
      scaleSine / squaredScale,
      targetMeanX - scaleCosine * frameMeanX + scaleSine * frameMeanY,
      targetMeanY - scaleSine * frameMeanX - scaleCosine * frameMeanY,
  };
}

/** Maps one aligned crop coordinate back into the original frame. */
inline Point alignedToFrame(const SimilarityTransform& transform,
                            double alignedX,
                            double alignedY) {
  const double translatedX = alignedX - transform.translateX;
  const double translatedY = alignedY - transform.translateY;
  return Point{
      transform.inverseScaleCosine * translatedX +
          transform.inverseScaleSine * translatedY,
      -transform.inverseScaleSine * translatedX +
          transform.inverseScaleCosine * translatedY,
  };
}

}  // namespace margelo::nitro::facerecognizer
