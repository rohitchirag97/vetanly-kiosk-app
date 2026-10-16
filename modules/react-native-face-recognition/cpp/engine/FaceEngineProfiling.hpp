#pragma once

#include <cstdint>

#if !defined(NDEBUG) || defined(FACE_RECOGNIZER_PROFILE_RELEASE)
#define FACE_RECOGNIZER_ENABLE_PROFILING 1
#endif

namespace margelo::nitro::facerecognizer {

enum class FaceProfileStage : std::uint8_t {
  LoadDetector,
  TrackFaces,
  DetectFaces,
  LoadLiveness,
  ScoreLiveness,
  RecognizeFaces,
  AlignFace,
  EmbedFace,
  MatchFace,
  LoadRecognizer,
  Count,
};

}  // namespace margelo::nitro::facerecognizer

#if FACE_RECOGNIZER_ENABLE_PROFILING
#include <chrono>

namespace margelo::nitro::facerecognizer {

class FaceProfileScope final {
public:
  explicit FaceProfileScope(FaceProfileStage stage) noexcept;
  ~FaceProfileScope() noexcept;

  FaceProfileScope(const FaceProfileScope&) = delete;
  FaceProfileScope& operator=(const FaceProfileScope&) = delete;
  FaceProfileScope(FaceProfileScope&&) = delete;
  FaceProfileScope& operator=(FaceProfileScope&&) = delete;

  void setItemCount(double itemCount) noexcept;

private:
  FaceProfileStage _stage;
  std::chrono::steady_clock::time_point _startedAt;
  double _itemCount = 0.0;
};

}  // namespace margelo::nitro::facerecognizer

#else

namespace margelo::nitro::facerecognizer {

class FaceProfileScope final {
public:
  explicit FaceProfileScope(FaceProfileStage stage) noexcept {
    (void)stage;
  }
  ~FaceProfileScope() noexcept = default;

  FaceProfileScope(const FaceProfileScope&) = delete;
  FaceProfileScope& operator=(const FaceProfileScope&) = delete;
  FaceProfileScope(FaceProfileScope&&) = delete;
  FaceProfileScope& operator=(FaceProfileScope&&) = delete;

  void setItemCount(double itemCount) noexcept {
    (void)itemCount;
  }
};

}  // namespace margelo::nitro::facerecognizer

#endif
