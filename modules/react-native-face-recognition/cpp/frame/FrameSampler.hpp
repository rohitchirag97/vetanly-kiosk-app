#pragma once

#include <VisionCamera/HybridFrameSpec.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "DetectedFace.hpp"
#include "LetterboxTransform.hpp"

namespace margelo::nitro::facerecognizer {

struct DetectorInputCache;
struct DetectorInputLayout;

/** Normalized luminance patch used by the tracker for correlation scoring. */
struct LuminancePatchTemplate final {
  std::span<const float> centeredPixels;
  double centeredSum;
  double inverseNorm;
};

/** Scores one row-strided luminance patch against a normalized template. */
double scoreSampledLuminancePatch(std::span<const float> samples,
                                  int sampleRowStride,
                                  int patchWidth,
                                  int patchHeight,
                                  const LuminancePatchTemplate& reference,
                                  double minPatchNorm);

/** Scores a validated row-strided patch without repeating bounds checks. */
double scoreSampledLuminancePatchUnchecked(
    std::span<const float> samples,
    int sampleRowStride,
    int patchWidth,
    int patchHeight,
    const LuminancePatchTemplate& reference,
    double minPatchNorm) noexcept;

/**
 * Read-only frame adapter for detector, recognizer, liveness, and tracker
 * input.
 *
 * The sampler centralizes YUV plane access, orientation, mirroring, color
 * conversion, and crop resampling. It never owns the frame; callers must keep
 * the `HybridFrameSpec` alive for the sampler lifetime.
 */
class FrameSampler final {
public:
  explicit FrameSampler(
      const std::shared_ptr<margelo::nitro::camera::HybridFrameSpec>& frame);

  /** Raw frame width before orientation is applied. */
  int getFrameWidth() const noexcept;
  /** Raw frame height before orientation is applied. */
  int getFrameHeight() const noexcept;
  /** Display width after orientation is applied. */
  int getUprightWidth() const noexcept;
  /** Display height after orientation is applied. */
  int getUprightHeight() const noexcept;

  /**
   * Builds a square BGR letterboxed detector tensor, reusing cached
   * YUV-to-tensor pixel mappings when the frame layout is unchanged.
   */
  void createDetectorInput(int inputSize,
                           LetterboxTransform& transform,
                           std::vector<float>& output,
                           DetectorInputCache& cache) const;

  /** Builds an aligned 112x112 recognizer tensor for one frame-space face. */
  void createRecognizerInput(const DetectedFace& face,
                             std::vector<float>& output) const;

  /**
   * Builds an 80x80 BGR [0,255] NCHW crop for anti-spoofing.
   *
   * `uprightBounds` is the face box in upright display space. It is expanded
   * by `cropScale` about its center, clamped to the frame, then resampled.
   */
  void createLivenessInput(const Rect& uprightBounds,
                           float cropScale,
                           std::vector<float>& output) const;

  /** Reverses detector letterbox coordinates into frame coordinates. */
  Point detectorToFrame(const Point& point,
                        const LetterboxTransform& transform) const;

  /** Converts a frame point into upright display coordinates. */
  Point frameToUpright(const Point& point) const;
  /** Samples one upright luminance value with bounds handling. */
  float sampleUprightLuminance(double uprightX, double uprightY) const;
  /** Samples an upright luminance patch for tracker template creation. */
  void sampleUprightLuminancePatch(double uprightLeft,
                                   double uprightTop,
                                   int patchWidth,
                                   int patchHeight,
                                   std::span<float> output) const;
  /**
   * Scores an upright luminance patch against a normalized tracker template.
   */
  double scoreUprightLuminancePatch(double uprightLeft,
                                    double uprightTop,
                                    int patchWidth,
                                    int patchHeight,
                                    const LuminancePatchTemplate& reference,
                                    double minPatchNorm) const;

private:
  struct YuvPlane final {
    std::shared_ptr<margelo::nitro::ArrayBuffer> buffer;
    const uint8_t* data = nullptr;
    int bytesPerRow = 0;
    int pixelStride = 0;
  };

  struct BgrColor final {
    float blue;
    float green;
    float red;
  };

  struct LuminancePatchSamplingPlan final {
    Point frameTopLeft;
    bool canReadFrameWithoutBoundsChecks;
  };

  void initializeUprightBasis();
  BgrColor yuvToBgr(float luminance, float blueColor, float redColor) const;
  DetectorInputLayout detectorInputLayout(int inputSize) const;
  void rebuildDetectorInputCache(int inputSize,
                                 DetectorInputCache& cache) const;
  BgrColor sampleFrame(double frameX, double frameY) const;
  float sampleFrameLuminance(double frameX, double frameY) const;
  float sampleFrameLuminanceUnchecked(double frameX, double frameY) const;
  Point cachedUprightToFrame(double uprightX, double uprightY) const noexcept;
  Point uprightToFrame(double uprightX, double uprightY) const;
  LuminancePatchSamplingPlan createLuminancePatchSamplingPlan(
      double uprightLeft,
      double uprightTop,
      int patchWidth,
      int patchHeight) const noexcept;
  template <typename SampleConsumer>
  void sampleEachUprightLuminancePatchPixel(
      double uprightLeft,
      double uprightTop,
      int patchWidth,
      int patchHeight,
      SampleConsumer&& consumeSample) const;
  float samplePlane(const YuvPlane& plane,
                    double sampleX,
                    double sampleY,
                    int sampleWidth,
                    int sampleHeight,
                    int componentOffset) const;
  float samplePlaneUnchecked(const YuvPlane& plane,
                             double sampleX,
                             double sampleY,
                             int componentOffset) const;
  uint8_t readPlane(const YuvPlane& plane,
                    int pixelX,
                    int pixelY,
                    int componentOffset) const;

  int _frameWidth;
  int _frameHeight;
  int _uprightWidth;
  int _uprightHeight;
  int _colorPlaneWidth;
  int _colorPlaneHeight;
  bool _isMirrored;
  bool _isFullRange;
  bool _hasInterleavedColor;
  margelo::nitro::camera::CameraOrientation _orientation;
  Point _uprightFrameOrigin;
  Point _uprightFrameXStep;
  Point _uprightFrameYStep;
  std::array<YuvPlane, 3> _planes;
};

}  // namespace margelo::nitro::facerecognizer
