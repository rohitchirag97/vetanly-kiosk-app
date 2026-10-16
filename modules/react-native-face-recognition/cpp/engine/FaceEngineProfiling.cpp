#include "FaceEngineProfiling.hpp"

#if FACE_RECOGNIZER_ENABLE_PROFILING

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <mutex>

#if defined(__ANDROID__)
#include <android/log.h>
#endif

namespace margelo::nitro::facerecognizer {
namespace {

struct ProfileAccumulator final {
  double elapsedMs = 0.0;
  double maxElapsedMs = 0.0;
  double itemCount = 0.0;
  int count = 0;
  int over16MsCount = 0;
  int over33MsCount = 0;
};

inline constexpr int kProfileLogInterval = 60;
inline constexpr double kSixtyFpsBudgetMs = 1000.0 / 60.0;
inline constexpr double kThirtyFpsBudgetMs = 1000.0 / 30.0;

constexpr std::size_t stageIndex(FaceProfileStage stage) noexcept {
  return static_cast<std::size_t>(stage);
}

constexpr const char* stageName(FaceProfileStage stage) noexcept {
  switch (stage) {
    case FaceProfileStage::LoadDetector:
      return "load_detector";
    case FaceProfileStage::TrackFaces:
      return "track_faces";
    case FaceProfileStage::DetectFaces:
      return "detect_faces";
    case FaceProfileStage::LoadLiveness:
      return "load_liveness";
    case FaceProfileStage::ScoreLiveness:
      return "score_liveness";
    case FaceProfileStage::RecognizeFaces:
      return "recognize_faces";
    case FaceProfileStage::AlignFace:
      return "align_face";
    case FaceProfileStage::EmbedFace:
      return "embed_face";
    case FaceProfileStage::MatchFace:
      return "match_face";
    case FaceProfileStage::LoadRecognizer:
      return "load_recognizer";
    case FaceProfileStage::Count:
      break;
  }
  return "unknown";
}

constexpr bool shouldLogImmediately(FaceProfileStage stage) noexcept {
  return stage == FaceProfileStage::LoadDetector ||
         stage == FaceProfileStage::LoadLiveness ||
         stage == FaceProfileStage::LoadRecognizer;
}

void logProfileSummary(FaceProfileStage stage,
                       const ProfileAccumulator& sample,
                       int emittedCount) noexcept {
  const double averageElapsedMs =
      sample.elapsedMs / static_cast<double>(emittedCount);
  const double averageItemCount =
      sample.itemCount / static_cast<double>(emittedCount);
  const char* name = stageName(stage);

#if defined(__ANDROID__)
  __android_log_print(
      ANDROID_LOG_INFO,
      "FaceRecognizer",
      "event=face_profile stage=%s calls=%d avg_ms=%.3f max_ms=%.3f "
      "over_16ms=%d over_33ms=%d avg_items=%.2f",
      name,
      emittedCount,
      averageElapsedMs,
      sample.maxElapsedMs,
      sample.over16MsCount,
      sample.over33MsCount,
      averageItemCount);
#else
  std::fprintf(stderr,
               "FaceRecognizer event=face_profile stage=%s calls=%d "
               "avg_ms=%.3f max_ms=%.3f over_16ms=%d over_33ms=%d "
               "avg_items=%.2f\n",
               name,
               emittedCount,
               averageElapsedMs,
               sample.maxElapsedMs,
               sample.over16MsCount,
               sample.over33MsCount,
               averageItemCount);
#endif
}

}  // namespace

FaceProfileScope::FaceProfileScope(FaceProfileStage stage) noexcept
    : _stage(stage), _startedAt(std::chrono::steady_clock::now()) {}

FaceProfileScope::~FaceProfileScope() noexcept {
  const std::chrono::steady_clock::time_point finishedAt =
      std::chrono::steady_clock::now();
  const std::chrono::duration<double, std::milli> elapsed =
      finishedAt - _startedAt;

  static std::mutex profileMutex;
  static std::array<ProfileAccumulator, stageIndex(FaceProfileStage::Count)>
      samples{};

  std::lock_guard<std::mutex> lock(profileMutex);
  ProfileAccumulator& stageSample = samples[stageIndex(_stage)];
  const double elapsedMs = elapsed.count();
  stageSample.elapsedMs += elapsedMs;
  stageSample.maxElapsedMs = std::max(stageSample.maxElapsedMs, elapsedMs);
  stageSample.itemCount += _itemCount;
  stageSample.count += 1;
  if (elapsedMs > kSixtyFpsBudgetMs) {
    stageSample.over16MsCount += 1;
  }
  if (elapsedMs > kThirtyFpsBudgetMs) {
    stageSample.over33MsCount += 1;
  }

  if (shouldLogImmediately(_stage)) {
    logProfileSummary(_stage, stageSample, stageSample.count);
    stageSample = ProfileAccumulator{};
    return;
  }
  if (stageSample.count < kProfileLogInterval) {
    return;
  }
  logProfileSummary(_stage, stageSample, stageSample.count);
  stageSample = ProfileAccumulator{};
}

void FaceProfileScope::setItemCount(double itemCount) noexcept {
  _itemCount = itemCount;
}

}  // namespace margelo::nitro::facerecognizer

#endif
