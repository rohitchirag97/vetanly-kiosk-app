#pragma once

#include "DetectedFace.hpp"
#include "FaceConstants.hpp"
#include "Point.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace margelo::nitro::facerecognizer {

class FrameSampler;

/**
 * Result from one tracking pass in upright display coordinates.
 *
 * Health metrics let the engine shorten the next detector interval when
 * tracking degrades, the subject moves quickly, or a large face dominates the
 * frame where patch drift is more visible.
 */
struct TrackingResult final {
  std::vector<DetectedFace> faces;
  double minimumScore;
  double maximumMotionPixels;
  double largestFaceFrameRatio;
};

/**
 * Lightweight landmark-template tracker used between full detector passes.
 *
 * The tracker stores small normalized luminance patches around five landmarks.
 * Tracking is intentionally conservative: unreliable landmarks are discarded,
 * inconsistent translations are filtered, and the detector refresh cadence is
 * tightened by the engine when score or motion metrics degrade.
 */
class FaceTracker final {
public:
  /** Drops all active tracks. */
  void reset() noexcept;
  /** Seeds tracks from detector output in frame coordinates. */
  void resetWithFrameFaces(const FrameSampler& sampler,
                           const std::vector<DetectedFace>& faces);
  /**
   * Seeds tracks from faces that are already in upright display coordinates.
   */
  void resetWithUprightFaces(const FrameSampler& sampler,
                             const std::vector<DetectedFace>& faces);
  /**
   * Seeds tracks from a callback without materializing an intermediate vector.
   */
  template <typename FaceAt>
  void resetWithUprightFaces(const FrameSampler& sampler,
                             std::size_t faceCount,
                             const FaceAt& faceAt) {
    _tracks.clear();
    _tracks.reserve(faceCount);
    for (std::size_t index = 0; index < faceCount; ++index) {
      appendUprightFace(sampler, faceAt(index));
    }
  }
  /** Tracks active faces against the current frame and refreshes templates. */
  TrackingResult track(const FrameSampler& sampler);
  /** Returns whether a future frame can attempt tracking before detection. */
  bool hasTracks() const noexcept;

private:
  /**
   * Half-width of the square landmark patch used for normalized correlation.
   */
  static constexpr int kPatchRadius = 3;
  static constexpr int kPatchWidth = kPatchRadius * 2 + 1;
  static constexpr std::size_t kPatchPixelCount =
      static_cast<std::size_t>(kPatchWidth) *
      static_cast<std::size_t>(kPatchWidth);

  struct TrackedLandmarkTemplate final {
    Point position;
    std::array<float, kPatchPixelCount> centeredPixels;
    double centeredSum;
    double inverseNorm;
  };

  struct Track final {
    Rect bounds;
    double confidence;
    std::array<TrackedLandmarkTemplate, kFaceLandmarkCount> landmarks;
  };

  void appendFrameFace(const FrameSampler& sampler, const DetectedFace& face);
  void appendUprightFace(const FrameSampler& sampler, const DetectedFace& face);
  void appendTrack(const FrameSampler& sampler,
                   const Rect& bounds,
                   double confidence,
                   std::span<const Point> landmarks);
  void sortTracksByCenterX() noexcept;

  std::vector<Track> _tracks;
};

}  // namespace margelo::nitro::facerecognizer
