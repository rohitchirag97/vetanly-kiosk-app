#pragma once

#include "DetectedFace.hpp"
#include "DetectorSession.hpp"
#include "EnrollmentStore.hpp"
#include "EnrollFaceResult.hpp"
#include "FaceEngineConfig.hpp"
#include "FaceTracker.hpp"
#include "LivenessEnsemble.hpp"
#include "NativeRecognizedFace.hpp"
#include "RecognizerSession.hpp"

#include "onnxruntime_cxx_api.h"
#include <NitroModules/ArrayBuffer.hpp>
#include <VisionCamera/HybridFrameSpec.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace margelo::nitro::facerecognizer {

class FrameSampler;

/**
 * Owns native detector, recognizer, liveness, enrollment, and tracking state.
 *
 * Public methods are serialized by `_inferenceMutex` because ONNX sessions,
 * reusable tensors, recent-match state, and enrollment indexes are mutable. A
 * single instance is intended for one active camera pipeline.
 */
class FaceEngine final {
public:
  /**
   * Reusable match carried across frames when recognition is intentionally
   * skipped.
   */
  struct RecentMatch final {
    Rect bounds;
    std::string enrollmentId;
    double similarity;
    int age;
  };

  /** Fresh embeddings collected for one spatially associated face. */
  struct PendingRecognition final {
    Rect bounds;
    EnrollmentSamples samples;
  };

  explicit FaceEngine(const FaceRecognizerOptions& options);
  ~FaceEngine();

  FaceEngine(const FaceEngine&) = delete;
  FaceEngine& operator=(const FaceEngine&) = delete;
  FaceEngine(FaceEngine&&) = delete;
  FaceEngine& operator=(FaceEngine&&) = delete;

  /**
   * Attempts to enroll exactly one live, high-quality face from a frame.
   *
   * The detector may request more than the configured `maxFaces` so it can
   * reliably reject multi-person frames before writing a template.
   */
  EnrollFaceResult enrollFace(
      const std::string& enrollmentId,
      const std::shared_ptr<margelo::nitro::camera::HybridFrameSpec>& frame);

  /**
   * Detects/tracks faces and optionally embeds them for identity matching.
   *
   * This method is synchronous so VisionCamera frame processors can call it
   * without scheduling JS work. The caller remains responsible for disposing
   * the frame after the call returns.
   */
  std::vector<NativeRecognizedFace> recognizeFaces(
      const std::shared_ptr<margelo::nitro::camera::HybridFrameSpec>& frame);

  void addEnrollment(
      const std::string& enrollmentId,
      const std::shared_ptr<margelo::nitro::ArrayBuffer>& enrollment);
  std::shared_ptr<margelo::nitro::ArrayBuffer> getEnrollment(
      const std::string& enrollmentId);
  std::vector<std::string> enrollmentIds();
  void removeEnrollment(const std::string& enrollmentId);

  void clearEnrollments();
  void dispose() noexcept;
  std::size_t externalMemorySize() const noexcept;

private:
  void releaseIdleModelsLocked() noexcept;
  void releaseResourcesLocked() noexcept;
  void trimNativeHeapLocked() const noexcept;
  void ensureActiveLocked() const;

  FaceEngineConfig _config;
  Ort::MemoryInfo _memoryInfo;
  DetectorSession _detector;
  RecognizerSession _recognizer;
  LivenessEnsemble _liveness;
  EnrollmentStore _enrollments;
  FaceTracker _tracker;
  int _framesSinceDetection;
  int _framesSinceRecognition;
  std::vector<RecentMatch> _recentMatches;
  std::vector<PendingRecognition> _pendingRecognitions;
  mutable std::mutex _inferenceMutex;
  /**
   * Last memory estimate observed under the lock. `externalMemorySize()` is
   * called from the JS GC and must never block behind a frame in flight or a
   * model load, so it refreshes this only when the lock is free.
   */
  mutable std::atomic<std::size_t> _lastKnownExternalMemorySize{0};
  bool _isDisposed;
};

}  // namespace margelo::nitro::facerecognizer
