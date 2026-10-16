#include "FaceTracker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <utility>

#include "FaceEngineProfiling.hpp"
#include "FaceGeometry.hpp"
#include "FrameSampler.hpp"

namespace margelo::nitro::facerecognizer {
namespace {

constexpr double kMinPatchNorm = 1e-6;
constexpr double kMinTrackScore = 0.76;
constexpr double kConfidenceDecay = 0.97;
constexpr double kMinTrackedConfidence = 0.05;
constexpr double kMinTrackableFaceSize = 8.0;
constexpr double kMinSearchRadius = 4.0;
constexpr double kMaxSearchRadius = 28.0;
constexpr double kSearchRadiusFaceRatio = 0.14;
constexpr double kMinScaleBaseline = 8.0;
constexpr double kMinInterFrameScale = 0.96;
constexpr double kMaxInterFrameScale = 1.04;
constexpr double kScaleDeadZone = 0.008;
constexpr double kMaxScaleMedianDeviation = 0.012;
constexpr double kScaleResponse = 0.4;
constexpr std::size_t kMinTrackedLandmarks = 4;
constexpr std::size_t kMaxLandmarkPairCount =
    kFaceLandmarkCount * (kFaceLandmarkCount - 1) / 2;

/** One landmark template match accepted by normalized patch correlation. */
struct TrackedLandmarkMatch final {
  Point previousPosition;
  Point currentPosition;
  double score;
};

struct TrackedLandmarkMatches final {
  std::array<TrackedLandmarkMatch, kFaceLandmarkCount> items;
  std::size_t count;
};

struct EvaluatedLandmarkCandidate final {
  Point position;
  double score;
};

struct LandmarkSearchResult final {
  Point position;
  int offsetX;
  int offsetY;
  double score;
};

double clampCoordinate(double value, int limit) {
  return std::clamp(value, 0.0, static_cast<double>(limit - 1));
}

Point clampUprightPoint(const FrameSampler& sampler, const Point& point) {
  return Point{
      clampCoordinate(point.x, sampler.getUprightWidth()),
      clampCoordinate(point.y, sampler.getUprightHeight()),
  };
}

Rect clampUprightBounds(const FrameSampler& sampler, const Rect& bounds) {
  const double maxWidth = static_cast<double>(sampler.getUprightWidth());
  const double maxHeight = static_cast<double>(sampler.getUprightHeight());
  const double minX = std::clamp(bounds.x, 0.0, maxWidth);
  const double minY = std::clamp(bounds.y, 0.0, maxHeight);
  const double maxX = std::clamp(bounds.x + bounds.width, 0.0, maxWidth);
  const double maxY = std::clamp(bounds.y + bounds.height, 0.0, maxHeight);
  return Rect{minX, minY, maxX - minX, maxY - minY};
}

bool isTrackableFace(const Rect& bounds, std::size_t landmarkCount) {
  return landmarkCount == kFaceLandmarkCount &&
         bounds.width >= kMinTrackableFaceSize &&
         bounds.height >= kMinTrackableFaceSize;
}

double centerX(const Rect& bounds) {
  return bounds.x + bounds.width * 0.5;
}

double centerY(const Rect& bounds) {
  return bounds.y + bounds.height * 0.5;
}

int searchRadiusForBounds(const Rect& bounds) {
  const double faceSize = std::max(bounds.width, bounds.height);
  return static_cast<int>(
      std::clamp(std::round(faceSize * kSearchRadiusFaceRatio),
                 kMinSearchRadius,
                 kMaxSearchRadius));
}

template <typename PositionOfMatch>
Point averageMatchedPosition(const TrackedLandmarkMatches& matches,
                             const PositionOfMatch& positionOfMatch) {
  Point average{0.0, 0.0};
  for (std::size_t index = 0; index < matches.count; ++index) {
    const Point& point = positionOfMatch(matches.items[index]);
    average.x += point.x;
    average.y += point.y;
  }
  const double matchCount = static_cast<double>(matches.count);
  return Point{average.x / matchCount, average.y / matchCount};
}

Point averagePreviousMatchedPosition(const TrackedLandmarkMatches& matches) {
  return averageMatchedPosition(
      matches, [](const TrackedLandmarkMatch& match) -> const Point& {
        return match.previousPosition;
      });
}

Point averageCurrentMatchedPosition(const TrackedLandmarkMatches& matches) {
  return averageMatchedPosition(
      matches, [](const TrackedLandmarkMatch& match) -> const Point& {
        return match.currentPosition;
      });
}

/** Median of the given values; sorts them in place. */
double medianOf(std::span<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middleIndex = values.size() / 2;
  if (values.size() % 2 != 0) {
    return values[middleIndex];
  }
  return (values[middleIndex - 1] + values[middleIndex]) * 0.5;
}

double distanceBetweenPoints(const Point& point, const Point& otherPoint) {
  const double deltaX = point.x - otherPoint.x;
  const double deltaY = point.y - otherPoint.y;
  return std::sqrt(deltaX * deltaX + deltaY * deltaY);
}

Point medianTranslation(const TrackedLandmarkMatches& matches) {
  std::array<double, kFaceLandmarkCount> translationsX{};
  std::array<double, kFaceLandmarkCount> translationsY{};
  for (std::size_t index = 0; index < matches.count; ++index) {
    const TrackedLandmarkMatch& match = matches.items[index];
    translationsX[index] = match.currentPosition.x - match.previousPosition.x;
    translationsY[index] = match.currentPosition.y - match.previousPosition.y;
  }
  return Point{
      medianOf(std::span<double>(translationsX.data(), matches.count)),
      medianOf(std::span<double>(translationsY.data(), matches.count))};
}

double trackedScale(const TrackedLandmarkMatches& matches) {
  std::array<double, kMaxLandmarkPairCount> scaleRatios{};
  std::size_t scaleRatioCount = 0;
  for (std::size_t index = 0; index < matches.count; ++index) {
    for (std::size_t otherIndex = index + 1; otherIndex < matches.count;
         ++otherIndex) {
      const TrackedLandmarkMatch& match = matches.items[index];
      const TrackedLandmarkMatch& otherMatch = matches.items[otherIndex];
      const double previousDistance = distanceBetweenPoints(
          match.previousPosition, otherMatch.previousPosition);
      if (previousDistance < kMinScaleBaseline) {
        continue;
      }

      const double currentDistance = distanceBetweenPoints(
          match.currentPosition, otherMatch.currentPosition);
      scaleRatios[scaleRatioCount] = currentDistance / previousDistance;
      ++scaleRatioCount;
    }
  }

  if (scaleRatioCount == 0) {
    return 1.0;
  }
  const double medianScale =
      medianOf(std::span<double>(scaleRatios.data(), scaleRatioCount));
  std::array<double, kMaxLandmarkPairCount> scaleDeviations{};
  for (std::size_t index = 0; index < scaleRatioCount; ++index) {
    scaleDeviations[index] = std::abs(scaleRatios[index] - medianScale);
  }
  if (medianOf(std::span<double>(scaleDeviations.data(), scaleRatioCount)) >
      kMaxScaleMedianDeviation) {
    return 1.0;
  }

  const double clampedScale =
      std::clamp(medianScale, kMinInterFrameScale, kMaxInterFrameScale);
  const double scaleDelta = clampedScale - 1.0;
  if (std::abs(scaleDelta) <= kScaleDeadZone) {
    return 1.0;
  }
  return 1.0 + scaleDelta * kScaleResponse;
}

TrackedLandmarkMatches filterConsistentMatches(
    const TrackedLandmarkMatches& matches, const Rect& bounds) {
  // A single bad landmark can pull the face box away from the subject. Use the
  // median translation as the robust motion estimate, then discard outliers.
  const Point median = medianTranslation(matches);
  const double faceSize = std::max(bounds.width, bounds.height);
  const double maxTranslationError =
      std::max(4.0, std::min(12.0, faceSize * 0.08));
  const double maxSquaredError = maxTranslationError * maxTranslationError;

  TrackedLandmarkMatches consistentMatches{{}, 0};
  for (std::size_t index = 0; index < matches.count; ++index) {
    const TrackedLandmarkMatch& match = matches.items[index];
    const double deltaX =
        match.currentPosition.x - match.previousPosition.x - median.x;
    const double deltaY =
        match.currentPosition.y - match.previousPosition.y - median.y;
    if (deltaX * deltaX + deltaY * deltaY > maxSquaredError) {
      continue;
    }
    consistentMatches.items[consistentMatches.count] = match;
    ++consistentMatches.count;
  }
  return consistentMatches;
}

Point transformPoint(const Point& point,
                     const Point& previousCenter,
                     const Point& currentCenter,
                     double scale) {
  return Point{
      currentCenter.x + (point.x - previousCenter.x) * scale,
      currentCenter.y + (point.y - previousCenter.y) * scale,
  };
}

Rect transformBounds(const Rect& bounds,
                     const Point& previousCenter,
                     const Point& currentCenter,
                     double scale) {
  // Unlike transformPoint, the box deliberately scales about its own center
  // rather than the matched-landmark centroid: anchoring box growth on the
  // landmark centroid would let residual scale noise walk the box off the
  // face between detector refreshes.
  const double width = bounds.width * scale;
  const double height = bounds.height * scale;
  const double translatedCenterX =
      centerX(bounds) + currentCenter.x - previousCenter.x;
  const double translatedCenterY =
      centerY(bounds) + currentCenter.y - previousCenter.y;
  return Rect{translatedCenterX - width * 0.5,
              translatedCenterY - height * 0.5,
              width,
              height};
}

double faceFrameRatio(const FrameSampler& sampler, const Rect& bounds) {
  const double widthRatio =
      bounds.width / static_cast<double>(sampler.getUprightWidth());
  const double heightRatio =
      bounds.height / static_cast<double>(sampler.getUprightHeight());
  return std::max(widthRatio, heightRatio);
}

}  // namespace

void FaceTracker::reset() noexcept {
  _tracks.clear();
}

bool FaceTracker::hasTracks() const noexcept {
  return !_tracks.empty();
}

void FaceTracker::resetWithFrameFaces(const FrameSampler& sampler,
                                      const std::vector<DetectedFace>& faces) {
  _tracks.clear();
  _tracks.reserve(faces.size());
  for (const DetectedFace& face : faces) {
    appendFrameFace(sampler, face);
  }
  sortTracksByCenterX();
}

void FaceTracker::resetWithUprightFaces(
    const FrameSampler& sampler, const std::vector<DetectedFace>& faces) {
  resetWithUprightFaces(sampler,
                        faces.size(),
                        [&faces](std::size_t index) -> const DetectedFace& {
                          return faces[index];
                        });
}

void FaceTracker::appendFrameFace(const FrameSampler& sampler,
                                  const DetectedFace& face) {
  const Rect bounds = toUprightBounds(sampler, face.bounds);
  if (!isTrackableFace(bounds, face.landmarks.size())) {
    return;
  }

  std::array<Point, kFaceLandmarkCount> landmarks{};
  for (std::size_t index = 0; index < kFaceLandmarkCount; ++index) {
    landmarks[index] = sampler.frameToUpright(face.landmarks[index]);
  }
  appendTrack(
      sampler, bounds, face.confidence, std::span<const Point>(landmarks));
}

void FaceTracker::appendUprightFace(const FrameSampler& sampler,
                                    const DetectedFace& face) {
  appendTrack(sampler, face.bounds, face.confidence, face.landmarks);
}

void FaceTracker::appendTrack(const FrameSampler& sampler,
                              const Rect& bounds,
                              double confidence,
                              std::span<const Point> landmarks) {
  if (!isTrackableFace(bounds, landmarks.size())) {
    return;
  }

  Track& track = _tracks.emplace_back();
  track.bounds = clampUprightBounds(sampler, bounds);
  track.confidence = confidence;
  for (std::size_t index = 0; index < kFaceLandmarkCount; ++index) {
    const Point landmarkPosition = clampUprightPoint(sampler, landmarks[index]);
    TrackedLandmarkTemplate& landmarkTemplate = track.landmarks[index];
    landmarkTemplate.position = landmarkPosition;
    std::array<float, kPatchPixelCount> patchPixels{};
    double luminanceSum = 0.0;
    sampler.sampleUprightLuminancePatch(
        landmarkPosition.x - static_cast<double>(kPatchRadius),
        landmarkPosition.y - static_cast<double>(kPatchRadius),
        kPatchWidth,
        kPatchWidth,
        std::span<float>(patchPixels.data(), patchPixels.size()));
    for (float value : patchPixels) {
      luminanceSum += value;
    }

    const double luminanceMean =
        luminanceSum / static_cast<double>(kPatchPixelCount);
    // Store zero-mean patches and inverse norms so each future candidate score
    // is a normalized correlation instead of brightness-sensitive difference.
    double centeredSquaredNorm = 0.0;
    double centeredSum = 0.0;
    for (std::size_t pixelIndex = 0; pixelIndex < kPatchPixelCount;
         ++pixelIndex) {
      const float centered = static_cast<float>(
          static_cast<double>(patchPixels[pixelIndex]) - luminanceMean);
      landmarkTemplate.centeredPixels[pixelIndex] = centered;
      centeredSum += centered;
      centeredSquaredNorm += static_cast<double>(centered) * centered;
    }
    landmarkTemplate.centeredSum = centeredSum;
    landmarkTemplate.inverseNorm = centeredSquaredNorm <= kMinPatchNorm
                                       ? 0.0
                                       : 1.0 / std::sqrt(centeredSquaredNorm);
  }
}

void FaceTracker::sortTracksByCenterX() noexcept {
  std::sort(_tracks.begin(),
            _tracks.end(),
            [](const Track& track, const Track& otherTrack) {
              return centerX(track.bounds) < centerX(otherTrack.bounds);
            });
}

TrackingResult FaceTracker::track(const FrameSampler& sampler) {
  FaceProfileScope profile(FaceProfileStage::TrackFaces);
  std::vector<DetectedFace> faces;
  faces.reserve(_tracks.size());
  double minimumScore = std::numeric_limits<double>::infinity();
  double maximumMotionPixels = 0.0;
  double largestFaceFrameRatio = 0.0;
  constexpr int kRefinementRadius = 1;
  constexpr int kMaxSampledSearchRadius =
      static_cast<int>(kMaxSearchRadius) + kRefinementRadius;
  constexpr int kMaxSampledSearchWidth =
      kMaxSampledSearchRadius * 2 + kPatchWidth;
  constexpr std::size_t kMaxSampledSearchPixelCount =
      static_cast<std::size_t>(kMaxSampledSearchWidth) *
      static_cast<std::size_t>(kMaxSampledSearchWidth);
  std::array<float, kMaxSampledSearchPixelCount> sampledSearchPixels;
  const auto searchLandmark =
      [](const Point& origin, int searchRadius, const auto& evaluateCandidate) {
        LandmarkSearchResult best{
            origin, 0, 0, -std::numeric_limits<double>::infinity()};
        const auto consider =
            [&](const Point& requestedPosition, int offsetX, int offsetY) {
              const EvaluatedLandmarkCandidate candidate =
                  evaluateCandidate(requestedPosition, offsetX, offsetY);
              if (candidate.score > best.score) {
                best = LandmarkSearchResult{
                    candidate.position, offsetX, offsetY, candidate.score};
              }
            };

        for (int y = -searchRadius; y <= searchRadius; y += 2) {
          for (int x = -searchRadius; x <= searchRadius; x += 2) {
            consider(Point{origin.x + static_cast<double>(x),
                           origin.y + static_cast<double>(y)},
                     x,
                     y);
          }
        }

        const Point refinementOrigin = best.position;
        const int refinementOffsetX = best.offsetX;
        const int refinementOffsetY = best.offsetY;
        for (int y = -kRefinementRadius; y <= kRefinementRadius; ++y) {
          for (int x = -kRefinementRadius; x <= kRefinementRadius; ++x) {
            consider(Point{refinementOrigin.x + static_cast<double>(x),
                           refinementOrigin.y + static_cast<double>(y)},
                     refinementOffsetX + x,
                     refinementOffsetY + y);
          }
        }
        return best;
      };

  for (const Track& track : _tracks) {
    const int searchRadius = searchRadiusForBounds(track.bounds);
    TrackedLandmarkMatches landmarkMatches{{}, 0};

    for (std::size_t landmarkIndex = 0; landmarkIndex < kFaceLandmarkCount;
         ++landmarkIndex) {
      const TrackedLandmarkTemplate& landmarkTemplate =
          track.landmarks[landmarkIndex];
      if (landmarkTemplate.inverseNorm == 0.0) {
        continue;
      }

      const int sampledSearchRadius = searchRadius + kRefinementRadius;
      const int sampledSearchWidth = sampledSearchRadius * 2 + kPatchWidth;
      const double sampledSearchLeft =
          landmarkTemplate.position.x -
          static_cast<double>(sampledSearchRadius + kPatchRadius);
      const double sampledSearchTop =
          landmarkTemplate.position.y -
          static_cast<double>(sampledSearchRadius + kPatchRadius);
      const double sampledSearchRight =
          sampledSearchLeft + static_cast<double>(sampledSearchWidth - 1);
      const double sampledSearchBottom =
          sampledSearchTop + static_cast<double>(sampledSearchWidth - 1);
      const bool canReuseSampledSearch =
          sampledSearchLeft >= 0.0 && sampledSearchTop >= 0.0 &&
          sampledSearchRight <=
              static_cast<double>(sampler.getUprightWidth() - 1) &&
          sampledSearchBottom <=
              static_cast<double>(sampler.getUprightHeight() - 1);
      std::size_t sampledSearchPixelCount = 0;
      if (canReuseSampledSearch) {
        sampledSearchPixelCount = static_cast<std::size_t>(sampledSearchWidth) *
                                  static_cast<std::size_t>(sampledSearchWidth);
        sampler.sampleUprightLuminancePatch(
            sampledSearchLeft,
            sampledSearchTop,
            sampledSearchWidth,
            sampledSearchWidth,
            std::span<float>(sampledSearchPixels.data(),
                             sampledSearchPixelCount));
      }

      const LuminancePatchTemplate reference{
          std::span<const float>(landmarkTemplate.centeredPixels.data(),
                                 landmarkTemplate.centeredPixels.size()),
          landmarkTemplate.centeredSum,
          landmarkTemplate.inverseNorm,
      };
      LandmarkSearchResult searchResult{
          landmarkTemplate.position,
          0,
          0,
          -std::numeric_limits<double>::infinity()};
      if (canReuseSampledSearch) {
        const auto evaluateSampledCandidate = [&](const Point& candidate,
                                                  int candidateOffsetX,
                                                  int candidateOffsetY) {
          const int patchLeft = candidateOffsetX + sampledSearchRadius;
          const int patchTop = candidateOffsetY + sampledSearchRadius;
          const std::size_t firstPixelIndex =
              static_cast<std::size_t>(patchTop) *
                  static_cast<std::size_t>(sampledSearchWidth) +
              static_cast<std::size_t>(patchLeft);
          const double score = scoreSampledLuminancePatchUnchecked(
              std::span<const float>(sampledSearchPixels.data(),
                                     sampledSearchPixelCount)
                  .subspan(firstPixelIndex),
              sampledSearchWidth,
              kPatchWidth,
              kPatchWidth,
              reference,
              kMinPatchNorm);
          return EvaluatedLandmarkCandidate{candidate, score};
        };
        searchResult = searchLandmark(
            landmarkTemplate.position, searchRadius, evaluateSampledCandidate);
      } else {
        const auto evaluateClampedCandidate =
            [&](const Point& requestedPosition, int, int) {
              const Point candidate =
                  clampUprightPoint(sampler, requestedPosition);
              const double score = sampler.scoreUprightLuminancePatch(
                  candidate.x - static_cast<double>(kPatchRadius),
                  candidate.y - static_cast<double>(kPatchRadius),
                  kPatchWidth,
                  kPatchWidth,
                  reference,
                  kMinPatchNorm);
              return EvaluatedLandmarkCandidate{candidate, score};
            };
        searchResult = searchLandmark(
            landmarkTemplate.position, searchRadius, evaluateClampedCandidate);
      }

      if (searchResult.score < kMinTrackScore) {
        continue;
      }
      landmarkMatches.items[landmarkMatches.count] = TrackedLandmarkMatch{
          landmarkTemplate.position, searchResult.position, searchResult.score};
      ++landmarkMatches.count;
    }

    if (landmarkMatches.count < kMinTrackedLandmarks) {
      continue;
    }

    const TrackedLandmarkMatches consistentLandmarkMatches =
        filterConsistentMatches(landmarkMatches, track.bounds);
    if (consistentLandmarkMatches.count < kMinTrackedLandmarks) {
      continue;
    }

    const Point previousCenter =
        averagePreviousMatchedPosition(consistentLandmarkMatches);
    const Point currentCenter =
        averageCurrentMatchedPosition(consistentLandmarkMatches);
    const double scale = trackedScale(consistentLandmarkMatches);
    const Rect nextBounds = clampUprightBounds(
        sampler,
        transformBounds(track.bounds, previousCenter, currentCenter, scale));
    if (nextBounds.width < kMinTrackableFaceSize ||
        nextBounds.height < kMinTrackableFaceSize) {
      continue;
    }

    std::array<Point, kFaceLandmarkCount> trackedLandmarks{};
    for (std::size_t index = 0; index < kFaceLandmarkCount; ++index) {
      trackedLandmarks[index] =
          clampUprightPoint(sampler,
                            transformPoint(track.landmarks[index].position,
                                           previousCenter,
                                           currentCenter,
                                           scale));
    }

    double matchedScoreSum = 0.0;
    for (std::size_t index = 0; index < consistentLandmarkMatches.count;
         ++index) {
      matchedScoreSum += consistentLandmarkMatches.items[index].score;
    }

    const double averageScore =
        matchedScoreSum / static_cast<double>(consistentLandmarkMatches.count);
    minimumScore = std::min(minimumScore, averageScore);
    const double translationMotionPixels =
        distanceBetweenPoints(previousCenter, currentCenter);
    const double scaleMotionPixels =
        std::max(track.bounds.width, track.bounds.height) *
        std::abs(scale - 1.0) * 0.5;
    maximumMotionPixels =
        std::max(maximumMotionPixels,
                 std::max(translationMotionPixels, scaleMotionPixels));
    largestFaceFrameRatio =
        std::max(largestFaceFrameRatio, faceFrameRatio(sampler, nextBounds));

    const double nextConfidence =
        std::max(kMinTrackedConfidence,
                 std::min(track.confidence, averageScore) * kConfidenceDecay);
    faces.push_back(DetectedFace{nextBounds, nextConfidence, trackedLandmarks});
  }

  resetWithUprightFaces(sampler, faces);
  profile.setItemCount(static_cast<double>(faces.size()));
  if (faces.empty()) {
    minimumScore = 0.0;
  }
  return TrackingResult{
      std::move(faces),
      minimumScore,
      maximumMotionPixels,
      largestFaceFrameRatio,
  };
}

}  // namespace margelo::nitro::facerecognizer
