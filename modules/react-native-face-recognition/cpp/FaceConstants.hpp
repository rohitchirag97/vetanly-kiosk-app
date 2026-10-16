#pragma once

#include <array>
#include <cstddef>

namespace margelo::nitro::facerecognizer {

/**
 * Detector input defaults and guardrails.
 *
 * YuNet accepts square inputs whose side length is divisible by the model
 * stride grid. Keeping the bounds here makes JS option validation and native
 * tensor allocation agree on the same fast path.
 */
inline constexpr int kDefaultDetectorInputSize = 320;
inline constexpr int kDetectorInputMultiple = 32;
inline constexpr int kMinDetectorInputSize = 64;
inline constexpr int kMaxDetectorInputSize = 1280;

/** YuNet output tensor layout: stride level x output prefix. */
inline constexpr std::array<int, 3> kYuNetStrides{8, 16, 32};
inline constexpr std::array<const char*, 4> kYuNetPrefixes{
    "cls", "obj", "bbox", "kps"};
inline constexpr std::array<int, 4> kYuNetChannels{1, 1, 4, 10};
inline constexpr int kYuNetOutputCount =
    static_cast<int>(kYuNetStrides.size() * kYuNetPrefixes.size());

/**
 * Five-point landmark order: left eye, right eye, nose, left mouth,
 * right mouth.
 */
inline constexpr std::size_t kFaceLandmarkCount = 5;

/** Non-maximum suppression overlap threshold for detector candidates. */
inline constexpr float kNmsIouThreshold = 0.3f;

/** Enrollment diversity limits for one identity. */
inline constexpr std::size_t kMaxSamplesPerEnrollment = 8;
inline constexpr float kEnrollDuplicateSimilarity = 0.92f;

/** ArcFace-compatible aligned crop size and target landmark template. */
inline constexpr int kRecognizerInputSize = 112;
inline constexpr double kAlignedFaceSize =
    static_cast<double>(kRecognizerInputSize);
inline constexpr std::array<std::array<double, 2>, 5> kTargetLandmarks{{
    {38.2946, 51.6963},
    {73.5318, 51.5014},
    {56.0252, 71.7366},
    {41.5493, 92.3655},
    {70.7299, 92.2041},
}};
inline constexpr double kMinRecognitionEyeDistanceRatio = 0.16;
inline constexpr double kMinRecognitionMouthDistanceRatio = 0.12;
inline constexpr double kMinRecognitionVerticalSpreadRatio = 0.18;
inline constexpr double kMinRecognitionMouthEyeRatio = 0.55;
inline constexpr double kMaxRecognitionMouthEyeRatio = 1.35;
inline constexpr double kMaxRecognitionNoseOffsetRatio = 0.45;
inline constexpr double kMaxRecognitionMouthCenterOffsetRatio = 0.35;

/**
 * Passive anti-spoofing model contract.
 *
 * MiniFASNet / Silent-Face models consume an 80x80 BGR crop in raw [0,255]
 * NCHW layout. The crop is expanded from the upright face box by `cropScale`.
 * Each model emits softmax classes where class 1 is "live"; the ensemble score
 * is the average live probability.
 */
inline constexpr int kLivenessInputSize = 80;
inline constexpr int kLivenessLiveClassIndex = 1;

/** Bundled liveness model metadata used to build the ensemble. */
struct LivenessModelSpec final {
  const char* baseName;
  float cropScale;
};

inline constexpr std::array<LivenessModelSpec, 2> kLivenessModels{{
    {"minifasnet_v2", 2.7f},
    {"minifasnet_v1se", 4.0f},
}};

}  // namespace margelo::nitro::facerecognizer
