#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "FaceEngine.hpp"
#include "FaceEngineProfiling.hpp"
#include "FaceGeometry.hpp"
#include "FrameSampler.hpp"

namespace margelo::nitro::facerecognizer {
namespace {

constexpr int kLostDetectionInterval = 2;
constexpr int kWeakTrackDetectionInterval = 2;
constexpr int kFastMotionDetectionInterval = 3;
constexpr int kLargeFaceDetectionInterval = 4;
constexpr double kWeakTrackScore = 0.82;
constexpr double kFastTrackMotionPixels = 16.0;
constexpr double kLargeFaceFrameRatio = 0.45;

struct RecognitionFaceCandidate final {
  std::optional<DetectedFace> frameFace;
  DetectedFace uprightFace;
};

double centerX(const Rect& bounds) {
  return bounds.x + bounds.width * 0.5;
}

double centerY(const Rect& bounds) {
  return bounds.y + bounds.height * 0.5;
}

double centerDistanceSquared(const Rect& bounds, const Rect& otherBounds) {
  const double deltaX = centerX(bounds) - centerX(otherBounds);
  const double deltaY = centerY(bounds) - centerY(otherBounds);
  return deltaX * deltaX + deltaY * deltaY;
}

bool isLargeEnoughForRecognition(const DetectedFace& face,
                                 const FaceEngineConfig& config) {
  return face.bounds.width >= config.recognitionMinFaceSize &&
         face.bounds.height >= config.recognitionMinFaceSize;
}

double matchDistanceLimitSquared(const Rect& bounds,
                                 const Rect& previousBounds) {
  const double faceSize = std::max({bounds.width,
                                    bounds.height,
                                    previousBounds.width,
                                    previousBounds.height});
  const double limit = std::max(16.0, faceSize * 0.55);
  return limit * limit;
}

int detectorIntervalForTrackingResult(const TrackingResult& trackingResult,
                                      int maximumInterval) {
  if (trackingResult.faces.empty()) {
    return kLostDetectionInterval;
  }
  if (trackingResult.minimumScore < kWeakTrackScore) {
    return std::min(maximumInterval, kWeakTrackDetectionInterval);
  }
  if (trackingResult.maximumMotionPixels > kFastTrackMotionPixels) {
    return std::min(maximumInterval, kFastMotionDetectionInterval);
  }
  if (trackingResult.largestFaceFrameRatio > kLargeFaceFrameRatio) {
    return std::min(maximumInterval, kLargeFaceDetectionInterval);
  }
  return maximumInterval;
}

std::vector<RecognitionFaceCandidate> makeDetectorFaceCandidates(
    const FrameSampler& sampler,
    const std::vector<DetectedFace>& detectedFaces) {
  std::vector<RecognitionFaceCandidate> faceCandidates;
  faceCandidates.reserve(detectedFaces.size());
  for (const DetectedFace& detectedFace : detectedFaces) {
    DetectedFace uprightFace = detectedFace;
    toUprightFace(sampler, uprightFace);
    faceCandidates.push_back(RecognitionFaceCandidate{
        std::optional<DetectedFace>(detectedFace), uprightFace});
  }
  std::sort(faceCandidates.begin(),
            faceCandidates.end(),
            [](const RecognitionFaceCandidate& candidate,
               const RecognitionFaceCandidate& otherCandidate) {
              return centerX(candidate.uprightFace.bounds) <
                     centerX(otherCandidate.uprightFace.bounds);
            });
  return faceCandidates;
}

std::vector<RecognitionFaceCandidate> makeTrackedFaceCandidates(
    const std::vector<DetectedFace>& trackedFaces) {
  std::vector<RecognitionFaceCandidate> faceCandidates;
  faceCandidates.reserve(trackedFaces.size());
  for (const DetectedFace& trackedFace : trackedFaces) {
    faceCandidates.push_back(
        RecognitionFaceCandidate{std::nullopt, trackedFace});
  }
  return faceCandidates;
}

std::optional<std::size_t> findRecentMatchIndex(
    const Rect& bounds, const std::vector<FaceEngine::RecentMatch>& matches) {
  std::optional<std::size_t> bestIndex;
  double bestSquaredDistance = std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index < matches.size(); ++index) {
    const FaceEngine::RecentMatch& match = matches[index];
    const double squaredDistance = centerDistanceSquared(bounds, match.bounds);
    if (squaredDistance >= bestSquaredDistance ||
        squaredDistance > matchDistanceLimitSquared(bounds, match.bounds)) {
      continue;
    }
    bestSquaredDistance = squaredDistance;
    bestIndex = index;
  }
  return bestIndex;
}

std::optional<std::size_t> findPendingRecognitionIndex(
    const DetectedFace& face,
    const FaceEmbedding& faceEmbedding,
    const std::vector<FaceEngine::PendingRecognition>& pendingRecognitions,
    float minimumSimilarity) {
  std::optional<std::size_t> bestIndex;
  float bestSimilarity = std::numeric_limits<float>::lowest();
  double bestSquaredDistance = std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index < pendingRecognitions.size(); ++index) {
    const FaceEngine::PendingRecognition& pending = pendingRecognitions[index];
    const double squaredDistance =
        centerDistanceSquared(face.bounds, pending.bounds);
    if (squaredDistance >
        matchDistanceLimitSquared(face.bounds, pending.bounds)) {
      continue;
    }
    if (pending.samples.empty() ||
        pending.samples.back().size() != faceEmbedding.size()) {
      throw std::logic_error(
          "FaceRecognizer pending recognition has an invalid embedding.");
    }
    const float similarity =
        cosineSimilarity(faceEmbedding, pending.samples.back());
    if (similarity < minimumSimilarity || similarity < bestSimilarity ||
        (similarity == bestSimilarity &&
         squaredDistance >= bestSquaredDistance)) {
      continue;
    }
    bestSimilarity = similarity;
    bestSquaredDistance = squaredDistance;
    bestIndex = index;
  }
  return bestIndex;
}

NativeRecognizedFace makeRecognizedFace(const DetectedFace& face,
                                        FaceRecognitionStatus status) {
  NativeRecognizedFace recognizedFace;
  recognizedFace.bounds = face.bounds;
  recognizedFace.confidence = face.confidence;
  recognizedFace.landmarks.assign(face.landmarks.begin(), face.landmarks.end());
  recognizedFace.status = status;
  return recognizedFace;
}

NativeRecognizedFace makeUnmatchedFace(const DetectedFace& face) {
  return makeRecognizedFace(face, FaceRecognitionStatus::UNMATCHED);
}

NativeRecognizedFace makeCollectingFace(const DetectedFace& face,
                                        std::size_t samplesCollected,
                                        int samplesRequired) {
  NativeRecognizedFace recognizedFace =
      makeRecognizedFace(face, FaceRecognitionStatus::COLLECTING);
  recognizedFace.samplesCollected = static_cast<double>(samplesCollected);
  recognizedFace.samplesRequired = static_cast<double>(samplesRequired);
  return recognizedFace;
}

NativeRecognizedFace makeMatchedFace(const DetectedFace& face,
                                     std::string_view enrollmentId,
                                     double similarity,
                                     int matchAge) {
  NativeRecognizedFace recognizedFace =
      makeRecognizedFace(face, FaceRecognitionStatus::MATCHED);
  recognizedFace.enrollmentId = std::string(enrollmentId);
  recognizedFace.similarity = similarity;
  recognizedFace.matchAge = static_cast<double>(matchAge);
  return recognizedFace;
}

NativeRecognizedFace makeSpoofedFace(const DetectedFace& face,
                                     double livenessScore) {
  NativeRecognizedFace recognizedFace =
      makeRecognizedFace(face, FaceRecognitionStatus::SPOOFED);
  recognizedFace.livenessScore = livenessScore;
  return recognizedFace;
}

}  // namespace

std::vector<NativeRecognizedFace> FaceEngine::recognizeFaces(
    const std::shared_ptr<margelo::nitro::camera::HybridFrameSpec>& frame) {
  FaceProfileScope profile(FaceProfileStage::RecognizeFaces);
  std::lock_guard<std::mutex> lock(_inferenceMutex);
  ensureActiveLocked();
  const FrameSampler sampler(frame);
  const bool hasEnrolledFaces = !_enrollments.empty();
  if (!hasEnrolledFaces) {
    _recentMatches.clear();
    _pendingRecognitions.clear();
    releaseIdleModelsLocked();
  }

  for (RecentMatch& match : _recentMatches) {
    ++match.age;
  }

  const bool hasPendingRecognitions = !_pendingRecognitions.empty();
  const int recognitionInterval = _recentMatches.empty()
                                      ? _config.recognitionInterval
                                      : _config.matchedRecognitionInterval;
  const bool recognitionCadenceElapsed =
      hasEnrolledFaces && (hasPendingRecognitions ||
                           _framesSinceRecognition >= recognitionInterval);
  const bool hasActiveTracks = _tracker.hasTracks();
  const int detectionInterval =
      hasActiveTracks ? _config.detectionInterval : kLostDetectionInterval;
  bool shouldRunDetector = hasPendingRecognitions || !_config.trackingEnabled ||
                           _framesSinceDetection >= detectionInterval;
  bool didRunDetector = false;

  std::vector<RecognitionFaceCandidate> faceCandidates;
  if (!shouldRunDetector) {
    TrackingResult trackingResult = _tracker.track(sampler);
    if (trackingResult.faces.empty()) {
      if (!hasActiveTracks) {
        _pendingRecognitions.clear();
        ++_framesSinceDetection;
        ++_framesSinceRecognition;
        profile.setItemCount(0.0);
        return {};
      }
      shouldRunDetector = true;
    } else {
      const int adaptiveDetectionInterval = detectorIntervalForTrackingResult(
          trackingResult, _config.detectionInterval);
      if (_framesSinceDetection >= adaptiveDetectionInterval) {
        shouldRunDetector = true;
      } else {
        faceCandidates = makeTrackedFaceCandidates(trackingResult.faces);
        ++_framesSinceDetection;
      }
    }
  }

  if (shouldRunDetector) {
    faceCandidates = makeDetectorFaceCandidates(
        sampler, _detector.detect(sampler, _config, _config.maxFaces));
    if (_config.trackingEnabled) {
      _tracker.resetWithUprightFaces(
          sampler,
          faceCandidates.size(),
          [&faceCandidates](std::size_t index) -> const DetectedFace& {
            return faceCandidates[index].uprightFace;
          });
    }
    _framesSinceDetection = 0;
    didRunDetector = true;
  }

  const bool willRecognize =
      didRunDetector && recognitionCadenceElapsed && !faceCandidates.empty();
  if (willRecognize) {
    _recognizer.ensureLoaded(_config, _memoryInfo, _enrollments.index());
    if (_config.livenessEnabled) {
      _liveness.ensureLoaded(_config, _memoryInfo);
    }
  }

  std::vector<NativeRecognizedFace> recognizedFaces;
  std::vector<RecentMatch> nextRecentMatches;
  std::vector<PendingRecognition> unclaimedPendingRecognitions =
      std::move(_pendingRecognitions);
  std::vector<PendingRecognition> nextPendingRecognitions;
  recognizedFaces.reserve(faceCandidates.size());
  nextRecentMatches.reserve(faceCandidates.size());
  nextPendingRecognitions.reserve(faceCandidates.size());

  for (const RecognitionFaceCandidate& candidate : faceCandidates) {
    const DetectedFace& uprightFace = candidate.uprightFace;

    if (!willRecognize) {
      const std::optional<std::size_t> recentMatchIndex =
          findRecentMatchIndex(uprightFace.bounds, _recentMatches);
      if (recentMatchIndex.has_value()) {
        const RecentMatch& recentMatch = _recentMatches[*recentMatchIndex];
        nextRecentMatches.push_back(RecentMatch{uprightFace.bounds,
                                                recentMatch.enrollmentId,
                                                recentMatch.similarity,
                                                recentMatch.age});
        recognizedFaces.push_back(makeMatchedFace(uprightFace,
                                                  recentMatch.enrollmentId,
                                                  recentMatch.similarity,
                                                  recentMatch.age));
        continue;
      }
      recognizedFaces.push_back(makeUnmatchedFace(uprightFace));
      continue;
    }

    std::optional<EnrollmentMatch> match;

    if (hasUsableRecognitionGeometry(uprightFace) &&
        isLargeEnoughForRecognition(uprightFace, _config)) {
      if (!candidate.frameFace.has_value()) {
        throw std::logic_error(
            "FaceRecognizer detector face is required when recognition runs.");
      }
      const FaceEmbedding& faceEmbedding =
          _recognizer.embed(sampler, *candidate.frameFace);
      {
        FaceProfileScope matchingProfile(FaceProfileStage::MatchFace);
        if (_config.recognitionSamplesPerDecision == 1) {
          match = _enrollments.findBestMatch(faceEmbedding,
                                             _config.recognitionThreshold,
                                             _config.recognitionMinimumMargin,
                                             _config.candidateRetrieval);
        } else {
          std::optional<PendingRecognition> pendingRecognition;
          const std::optional<std::size_t> pendingIndex =
              findPendingRecognitionIndex(uprightFace,
                                          faceEmbedding,
                                          unclaimedPendingRecognitions,
                                          _config.recognitionThreshold);
          if (pendingIndex.has_value()) {
            pendingRecognition =
                std::move(unclaimedPendingRecognitions[*pendingIndex]);
            unclaimedPendingRecognitions.erase(
                unclaimedPendingRecognitions.begin() +
                static_cast<std::ptrdiff_t>(*pendingIndex));
            pendingRecognition->bounds = uprightFace.bounds;
          }
          if (!pendingRecognition.has_value()) {
            pendingRecognition = PendingRecognition{uprightFace.bounds, {}};
            pendingRecognition->samples.reserve(static_cast<std::size_t>(
                _config.recognitionSamplesPerDecision));
          }
          pendingRecognition->samples.push_back(faceEmbedding);
          if (pendingRecognition->samples.size() <
              static_cast<std::size_t>(_config.recognitionSamplesPerDecision)) {
            const std::size_t samplesCollected =
                pendingRecognition->samples.size();
            nextPendingRecognitions.push_back(std::move(*pendingRecognition));
            recognizedFaces.push_back(
                makeCollectingFace(uprightFace,
                                   samplesCollected,
                                   _config.recognitionSamplesPerDecision));
            matchingProfile.setItemCount(1.0);
            continue;
          }
          // The matcher uses their centroid only for coarse retrieval; keeping
          // the individual embeddings lets its final decision require temporal
          // agreement instead of trusting one averaged probe.
          match = _enrollments.findBestMatch(pendingRecognition->samples,
                                             _config.recognitionThreshold,
                                             _config.recognitionMinimumMargin,
                                             _config.candidateRetrieval);
        }
        matchingProfile.setItemCount(1.0);
      }
    }

    if (match.has_value() && _config.livenessEnabled) {
      const double livenessScore = _liveness.score(sampler, uprightFace.bounds);
      if (livenessScore < _config.livenessThreshold) {
        recognizedFaces.push_back(makeSpoofedFace(uprightFace, livenessScore));
        continue;
      }
    }

    if (match.has_value()) {
      nextRecentMatches.push_back(RecentMatch{uprightFace.bounds,
                                              std::string(match->enrollmentId),
                                              match->similarity,
                                              0});
      recognizedFaces.push_back(makeMatchedFace(
          uprightFace, match->enrollmentId, match->similarity, 0));
      continue;
    }

    recognizedFaces.push_back(makeUnmatchedFace(uprightFace));
  }

  _recentMatches = std::move(nextRecentMatches);
  _pendingRecognitions = std::move(nextPendingRecognitions);
  _framesSinceRecognition = willRecognize ? 0 : _framesSinceRecognition + 1;

  profile.setItemCount(static_cast<double>(recognizedFaces.size()));
  return recognizedFaces;
}

}  // namespace margelo::nitro::facerecognizer
