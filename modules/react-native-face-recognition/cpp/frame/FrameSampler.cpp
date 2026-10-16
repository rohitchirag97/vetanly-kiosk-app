#include "FrameSampler.hpp"

#include "DetectorInputCache.hpp"
#include "FaceAlignment.hpp"

#if !defined(FACE_RECOGNIZER_BENCH_STANDALONE)
#include <VisionCamera/HybridFramePlaneSpec.hpp>
#include <VisionCamera/PixelFormat.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace margelo::nitro::facerecognizer {
namespace {

#if !defined(FACE_RECOGNIZER_BENCH_STANDALONE)
int checkedPositiveIntegerDimension(double value, const char* name) {
  if (!std::isfinite(value) || value <= 0.0 || std::trunc(value) != value) {
    throw std::invalid_argument("FaceRecognizer received an invalid frame " +
                                std::string(name) + ": " +
                                std::to_string(value) + ".");
  }
  return static_cast<int>(value);
}
#endif

float clampBgrComponent(float value) {
  return std::clamp(value, 0.0f, 255.0f);
}

std::size_t checkedPatchPixelCount(int patchWidth,
                                   int patchHeight,
                                   std::size_t actualPixelCount,
                                   const char* targetName) {
  if (patchWidth <= 0 || patchHeight <= 0) {
    throw std::invalid_argument(
        "FaceRecognizer luminance patch dimensions must be positive.");
  }
  const std::size_t expectedPixelCount = static_cast<std::size_t>(patchWidth) *
                                         static_cast<std::size_t>(patchHeight);
  if (actualPixelCount != expectedPixelCount) {
    throw std::invalid_argument(
        "FaceRecognizer luminance patch " + std::string(targetName) +
        " size does not match the requested dimensions.");
  }
  return expectedPixelCount;
}

class LuminancePatchScoreAccumulator final {
public:
  explicit LuminancePatchScoreAccumulator(
      const LuminancePatchTemplate& reference) noexcept
      : _reference(reference) {}

  void add(std::size_t pixelIndex, float value) noexcept {
    const double sample = value;
    _candidateSum += sample;
    _candidateSquaredSum += sample * sample;
    _rawDot +=
        static_cast<double>(_reference.centeredPixels[pixelIndex]) * sample;
  }

  double finish(std::size_t pixelCount, double minPatchNorm) const noexcept {
    const double count = static_cast<double>(pixelCount);
    const double candidateMean = _candidateSum / count;
    const double candidateSquaredNorm =
        _candidateSquaredSum - count * candidateMean * candidateMean;
    if (candidateSquaredNorm <= minPatchNorm) {
      return -std::numeric_limits<double>::infinity();
    }
    const double centeredDot = _rawDot - candidateMean * _reference.centeredSum;
    return centeredDot * _reference.inverseNorm /
           std::sqrt(candidateSquaredNorm);
  }

private:
  const LuminancePatchTemplate& _reference;
  double _candidateSum = 0.0;
  double _candidateSquaredSum = 0.0;
  double _rawDot = 0.0;
};

}  // namespace

double scoreSampledLuminancePatch(std::span<const float> samples,
                                  int sampleRowStride,
                                  int patchWidth,
                                  int patchHeight,
                                  const LuminancePatchTemplate& reference,
                                  double minPatchNorm) {
  checkedPatchPixelCount(
      patchWidth, patchHeight, reference.centeredPixels.size(), "reference");
  if (sampleRowStride < patchWidth) {
    throw std::invalid_argument(
        "FaceRecognizer luminance sample row stride is smaller than the "
        "requested patch width.");
  }
  const std::size_t lastRowOffset = static_cast<std::size_t>(patchHeight - 1) *
                                    static_cast<std::size_t>(sampleRowStride);
  const std::size_t requiredSampleCount =
      lastRowOffset + static_cast<std::size_t>(patchWidth);
  if (samples.size() < requiredSampleCount) {
    throw std::invalid_argument(
        "FaceRecognizer luminance samples do not contain the requested "
        "row-strided patch.");
  }

  return scoreSampledLuminancePatchUnchecked(samples,
                                             sampleRowStride,
                                             patchWidth,
                                             patchHeight,
                                             reference,
                                             minPatchNorm);
}

double scoreSampledLuminancePatchUnchecked(
    std::span<const float> samples,
    int sampleRowStride,
    int patchWidth,
    int patchHeight,
    const LuminancePatchTemplate& reference,
    double minPatchNorm) noexcept {
  const std::size_t patchPixelCount = static_cast<std::size_t>(patchWidth) *
                                      static_cast<std::size_t>(patchHeight);
  LuminancePatchScoreAccumulator score(reference);
  std::size_t pixelIndex = 0;
  for (int y = 0; y < patchHeight; ++y) {
    const std::size_t rowOffset =
        static_cast<std::size_t>(y) * static_cast<std::size_t>(sampleRowStride);
    for (int x = 0; x < patchWidth; ++x) {
      score.add(pixelIndex, samples[rowOffset + static_cast<std::size_t>(x)]);
      ++pixelIndex;
    }
  }
  return score.finish(patchPixelCount, minPatchNorm);
}

#if !defined(FACE_RECOGNIZER_BENCH_STANDALONE)
FrameSampler::FrameSampler(
    const std::shared_ptr<margelo::nitro::camera::HybridFrameSpec>& frame)
    : _frameWidth(0),
      _frameHeight(0),
      _uprightWidth(0),
      _uprightHeight(0),
      _colorPlaneWidth(0),
      _colorPlaneHeight(0),
      _isMirrored(false),
      _isFullRange(false),
      _hasInterleavedColor(false),
      _orientation(margelo::nitro::camera::CameraOrientation::UP),
      _uprightFrameOrigin{0.0, 0.0},
      _uprightFrameXStep{1.0, 0.0},
      _uprightFrameYStep{0.0, 1.0} {
  if (frame == nullptr) {
    throw std::invalid_argument("FaceRecognizer received a null frame.");
  }
  if (!frame->getIsValid()) {
    throw std::invalid_argument(
        "FaceRecognizer cannot process a disposed VisionCamera frame.");
  }
  if (!frame->getIsPlanar() || !frame->getHasPixelBuffer()) {
    throw std::invalid_argument(
        "FaceRecognizer requires a CPU-readable planar YUV frame. Configure "
        "useFrameOutput({ pixelFormat: 'yuv' }).");
  }

  const margelo::nitro::camera::PixelFormat pixelFormat =
      frame->getPixelFormat();
  if (pixelFormat != margelo::nitro::camera::PixelFormat::YUV_420_8_BIT_VIDEO &&
      pixelFormat != margelo::nitro::camera::PixelFormat::YUV_420_8_BIT_FULL) {
    throw std::invalid_argument(
        "FaceRecognizer supports only 8-bit YUV 4:2:0 frames; received pixel "
        "format value " +
        std::to_string(static_cast<int>(pixelFormat)) + ".");
  }

  _frameWidth = checkedPositiveIntegerDimension(frame->getWidth(), "width");
  _frameHeight = checkedPositiveIntegerDimension(frame->getHeight(), "height");
  _orientation = frame->getOrientation();
  _isMirrored = frame->getIsMirrored();
  _isFullRange =
      pixelFormat == margelo::nitro::camera::PixelFormat::YUV_420_8_BIT_FULL;

  if (_orientation == margelo::nitro::camera::CameraOrientation::RIGHT ||
      _orientation == margelo::nitro::camera::CameraOrientation::LEFT) {
    _uprightWidth = _frameHeight;
    _uprightHeight = _frameWidth;
  } else {
    _uprightWidth = _frameWidth;
    _uprightHeight = _frameHeight;
  }
  _colorPlaneWidth = (_frameWidth + 1) / 2;
  _colorPlaneHeight = (_frameHeight + 1) / 2;
  initializeUprightBasis();

  const std::vector<
      std::shared_ptr<margelo::nitro::camera::HybridFramePlaneSpec>>
      framePlanes = frame->getPlanes();
  if (framePlanes.size() != 2 && framePlanes.size() != 3) {
    throw std::invalid_argument(
        "FaceRecognizer requires two-plane NV12 or three-plane YUV420; "
        "received " +
        std::to_string(framePlanes.size()) + " planes.");
  }
  _hasInterleavedColor = framePlanes.size() == 2;
  for (size_t planeIndex = 0; planeIndex < framePlanes.size(); ++planeIndex) {
    const std::shared_ptr<margelo::nitro::camera::HybridFramePlaneSpec>&
        framePlane = framePlanes[planeIndex];
    if (framePlane == nullptr || !framePlane->getIsValid()) {
      throw std::invalid_argument(
          "FaceRecognizer received an invalid frame plane at index " +
          std::to_string(planeIndex) + ".");
    }
    const int bytesPerRow = checkedPositiveIntegerDimension(
        framePlane->getBytesPerRow(), "plane bytes per row");
    const int planeWidth =
        checkedPositiveIntegerDimension(framePlane->getWidth(), "plane width");
    int pixelStride = 1;
    if (planeIndex > 0 && framePlanes.size() == 2) {
      pixelStride = 2;
    } else if (planeIndex > 0) {
      if (bytesPerRow % planeWidth != 0) {
        throw std::invalid_argument(
            "FaceRecognizer could not derive the Android color-plane pixel "
            "stride for frame plane " +
            std::to_string(planeIndex) + ".");
      }
      pixelStride = bytesPerRow / planeWidth;
    }
    if (pixelStride < 1 || pixelStride > 2) {
      throw std::invalid_argument(
          "FaceRecognizer supports only one-byte or two-byte YUV plane "
          "strides; received " +
          std::to_string(pixelStride) + ".");
    }
    std::shared_ptr<margelo::nitro::ArrayBuffer> buffer =
        framePlane->getPixelBuffer();
    if (buffer == nullptr || buffer->data() == nullptr || buffer->size() == 0) {
      throw std::invalid_argument(
          "FaceRecognizer received an empty frame plane at index " +
          std::to_string(planeIndex) + ".");
    }
    // Bound the buffer once so per-pixel sampling can index it without a
    // branch. samplePlane clamps reads to [0, sampleWidth - 1] x
    // [0, sampleHeight - 1], and NV12 color planes additionally read
    // componentOffset 1, so this is the largest byte offset readPlane can ever
    // produce for this plane.
    const int sampleWidth =
        planeIndex == 0 ? _frameWidth : (_frameWidth + 1) / 2;
    const int sampleHeight =
        planeIndex == 0 ? _frameHeight : (_frameHeight + 1) / 2;
    const int maxComponentOffset =
        (planeIndex > 0 && framePlanes.size() == 2) ? 1 : 0;
    const size_t maxByteOffset = static_cast<size_t>(sampleHeight - 1) *
                                     static_cast<size_t>(bytesPerRow) +
                                 static_cast<size_t>(sampleWidth - 1) *
                                     static_cast<size_t>(pixelStride) +
                                 static_cast<size_t>(maxComponentOffset);
    if (maxByteOffset >= buffer->size()) {
      throw std::invalid_argument(
          "FaceRecognizer frame plane metadata points outside its pixel "
          "buffer at index " +
          std::to_string(planeIndex) + ".");
    }
    if (maxByteOffset > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument(
          "FaceRecognizer frame plane exceeds the compact offset range at "
          "index " +
          std::to_string(planeIndex) + ".");
    }
    const uint8_t* bufferData = buffer->data();
    _planes[planeIndex] = YuvPlane{
        std::move(buffer),
        bufferData,
        bytesPerRow,
        pixelStride,
    };
  }
}
#endif

void FrameSampler::initializeUprightBasis() {
  // Cache the affine map from upright display space into raw frame space so
  // per-pixel sampling can avoid the orientation switch.
  _uprightFrameOrigin = uprightToFrame(0.0, 0.0);
  const Point frameXStep = uprightToFrame(1.0, 0.0);
  const Point frameYStep = uprightToFrame(0.0, 1.0);
  _uprightFrameXStep = Point{
      frameXStep.x - _uprightFrameOrigin.x,
      frameXStep.y - _uprightFrameOrigin.y,
  };
  _uprightFrameYStep = Point{
      frameYStep.x - _uprightFrameOrigin.x,
      frameYStep.y - _uprightFrameOrigin.y,
  };
}

int FrameSampler::getFrameWidth() const noexcept {
  return _frameWidth;
}

int FrameSampler::getFrameHeight() const noexcept {
  return _frameHeight;
}

int FrameSampler::getUprightWidth() const noexcept {
  return _uprightWidth;
}

int FrameSampler::getUprightHeight() const noexcept {
  return _uprightHeight;
}

void FrameSampler::createDetectorInput(int inputSize,
                                       LetterboxTransform& transform,
                                       std::vector<float>& output,
                                       DetectorInputCache& cache) const {
  if (inputSize <= 0) {
    throw std::invalid_argument(
        "FaceRecognizer detector input size must be positive.");
  }

  const DetectorInputLayout layout = detectorInputLayout(inputSize);
  const bool shouldRebuildCache =
      !cache.hasReusablePixelMappings || cache.layout != layout;
  if (shouldRebuildCache) {
    rebuildDetectorInputCache(inputSize, cache);
  }
  transform = cache.transform;

  const size_t planeStride = static_cast<size_t>(inputSize) * inputSize;
  const size_t inputElementCount = planeStride * 3;
  const bool resized = output.size() != inputElementCount;
  if (resized) {
    output.resize(inputElementCount);
  }
  if (shouldRebuildCache || resized) {
    std::fill(output.begin(), output.end(), 0.0f);
  }

  float* blue = output.data();
  float* green = output.data() + planeStride;
  float* red = output.data() + planeStride * 2;
  const uint8_t* luminanceData = _planes[0].data;
  const uint8_t* blueColorData = _planes[1].data;
  const uint8_t* redColorData =
      _hasInterleavedColor ? _planes[1].data : _planes[2].data;

  for (const DetectorInputPixelMapping& pixelMapping : cache.pixelMappings) {
    const BgrColor color = yuvToBgr(luminanceData[pixelMapping.luminanceOffset],
                                    blueColorData[pixelMapping.blueColorOffset],
                                    redColorData[pixelMapping.redColorOffset]);
    blue[pixelMapping.destinationIndex] = color.blue;
    green[pixelMapping.destinationIndex] = color.green;
    red[pixelMapping.destinationIndex] = color.red;
  }
}

void FrameSampler::createRecognizerInput(const DetectedFace& face,
                                         std::vector<float>& output) const {
  const SimilarityTransform alignmentTransform =
      createFaceAlignmentTransform(face.landmarks, _isMirrored);
  // Planar NCHW: channel 0 (red), 1 (green), 2 (blue). Every pixel is written
  // below, so this resize only sizes the reused buffer (no padding to zero).
  const size_t planeStride =
      static_cast<size_t>(kRecognizerInputSize) * kRecognizerInputSize;
  output.resize(planeStride * 3);

  float* red = output.data();
  float* green = output.data() + planeStride;
  float* blue = output.data() + planeStride * 2;
  for (int y = 0; y < kRecognizerInputSize; ++y) {
    for (int x = 0; x < kRecognizerInputSize; ++x) {
      const Point framePoint = alignedToFrame(
          alignmentTransform, static_cast<double>(x), static_cast<double>(y));
      const BgrColor color = sampleFrame(framePoint.x, framePoint.y);
      const int pixelIndex = y * kRecognizerInputSize + x;
      red[pixelIndex] = color.red;
      green[pixelIndex] = color.green;
      blue[pixelIndex] = color.blue;
    }
  }
}

void FrameSampler::createLivenessInput(const Rect& uprightBounds,
                                       float cropScale,
                                       std::vector<float>& output) const {
  // Expand the face box by cropScale about its center, then shift it back
  // inside the frame (matching the MiniFASNet/Silent-Face CropImage geometry),
  // so the model sees the face plus the surrounding context it was trained on.
  const double uprightWidth = static_cast<double>(_uprightWidth);
  const double uprightHeight = static_cast<double>(_uprightHeight);
  const double boxWidth = std::max(1.0, uprightBounds.width);
  const double boxHeight = std::max(1.0, uprightBounds.height);

  double scale = std::min(static_cast<double>(cropScale),
                          std::min((uprightWidth - 1.0) / boxWidth,
                                   (uprightHeight - 1.0) / boxHeight));
  if (!std::isfinite(scale) || scale <= 0.0) {
    scale = cropScale;
  }

  const double newWidth = boxWidth * scale;
  const double newHeight = boxHeight * scale;
  const double centerX = uprightBounds.x + boxWidth * 0.5;
  const double centerY = uprightBounds.y + boxHeight * 0.5;

  double cropLeft = centerX - newWidth * 0.5;
  double cropTop = centerY - newHeight * 0.5;
  double cropRight = centerX + newWidth * 0.5;
  double cropBottom = centerY + newHeight * 0.5;
  if (cropLeft < 0.0) {
    cropRight -= cropLeft;
    cropLeft = 0.0;
  }
  if (cropTop < 0.0) {
    cropBottom -= cropTop;
    cropTop = 0.0;
  }
  if (cropRight > uprightWidth - 1.0) {
    cropLeft -= cropRight - (uprightWidth - 1.0);
    cropRight = uprightWidth - 1.0;
  }
  if (cropBottom > uprightHeight - 1.0) {
    cropTop -= cropBottom - (uprightHeight - 1.0);
    cropBottom = uprightHeight - 1.0;
  }
  cropLeft = std::max(0.0, cropLeft);
  cropTop = std::max(0.0, cropTop);
  const double cropWidth = std::max(1.0, cropRight - cropLeft);
  const double cropHeight = std::max(1.0, cropBottom - cropTop);

  // Planar NCHW BGR in raw [0,255]. MiniFASNet/Silent-Face feeds the cv2 crop
  // straight through float() with no /255 and no mean/std (see the reference
  // onnx_inference.py _preprocess). Every pixel is written, so resize only
  // sizes the reused buffer.
  const size_t planeStride =
      static_cast<size_t>(kLivenessInputSize) * kLivenessInputSize;
  output.resize(planeStride * 3);
  float* blue = output.data();
  float* green = output.data() + planeStride;
  float* red = output.data() + planeStride * 2;
  for (int y = 0; y < kLivenessInputSize; ++y) {
    const double uprightY =
        cropTop + (y + 0.5) / kLivenessInputSize * cropHeight;
    for (int x = 0; x < kLivenessInputSize; ++x) {
      const double uprightX =
          cropLeft + (x + 0.5) / kLivenessInputSize * cropWidth;
      const Point framePoint = uprightToFrame(uprightX, uprightY);
      const BgrColor color = sampleFrame(framePoint.x, framePoint.y);
      const int pixelIndex = y * kLivenessInputSize + x;
      blue[pixelIndex] = color.blue;
      green[pixelIndex] = color.green;
      red[pixelIndex] = color.red;
    }
  }
}

Point FrameSampler::detectorToFrame(const Point& point,
                                    const LetterboxTransform& transform) const {
  return uprightToFrame((point.x - transform.offsetX) / transform.scale,
                        (point.y - transform.offsetY) / transform.scale);
}

Point FrameSampler::frameToUpright(const Point& point) const {
  // Undo the sensor orientation first, then apply the front-camera mirror, so
  // the result lands in the upright _uprightWidth x _uprightHeight display
  // space.
  double x = point.x;
  double y = point.y;
  switch (_orientation) {
    case margelo::nitro::camera::CameraOrientation::UP:
      break;
    case margelo::nitro::camera::CameraOrientation::RIGHT:
      x = point.y;
      y = static_cast<double>(_frameWidth - 1) - point.x;
      break;
    case margelo::nitro::camera::CameraOrientation::DOWN:
      x = static_cast<double>(_frameWidth - 1) - point.x;
      y = static_cast<double>(_frameHeight - 1) - point.y;
      break;
    case margelo::nitro::camera::CameraOrientation::LEFT:
      x = static_cast<double>(_frameHeight - 1) - point.y;
      y = point.x;
      break;
  }
  if (_isMirrored) {
    x = static_cast<double>(_uprightWidth - 1) - x;
  }
  return Point{x, y};
}

float FrameSampler::sampleUprightLuminance(double uprightX,
                                           double uprightY) const {
  const Point framePoint = cachedUprightToFrame(uprightX, uprightY);
  return sampleFrameLuminance(framePoint.x, framePoint.y);
}

FrameSampler::LuminancePatchSamplingPlan
FrameSampler::createLuminancePatchSamplingPlan(double uprightLeft,
                                               double uprightTop,
                                               int patchWidth,
                                               int patchHeight) const noexcept {
  // Affine extrema occur at corners; -2 keeps bilinear neighbours in bounds.
  const Point frameTopLeft = cachedUprightToFrame(uprightLeft, uprightTop);
  const Point frameTopRight = cachedUprightToFrame(
      uprightLeft + static_cast<double>(patchWidth - 1), uprightTop);
  const Point frameBottomLeft = cachedUprightToFrame(
      uprightLeft, uprightTop + static_cast<double>(patchHeight - 1));
  const Point frameBottomRight =
      cachedUprightToFrame(uprightLeft + static_cast<double>(patchWidth - 1),
                           uprightTop + static_cast<double>(patchHeight - 1));
  const double minFrameX =
      std::min(std::min(frameTopLeft.x, frameTopRight.x),
               std::min(frameBottomLeft.x, frameBottomRight.x));
  const double maxFrameX =
      std::max(std::max(frameTopLeft.x, frameTopRight.x),
               std::max(frameBottomLeft.x, frameBottomRight.x));
  const double minFrameY =
      std::min(std::min(frameTopLeft.y, frameTopRight.y),
               std::min(frameBottomLeft.y, frameBottomRight.y));
  const double maxFrameY =
      std::max(std::max(frameTopLeft.y, frameTopRight.y),
               std::max(frameBottomLeft.y, frameBottomRight.y));
  return LuminancePatchSamplingPlan{
      frameTopLeft,
      minFrameX >= 0.0 && minFrameY >= 0.0 &&
          maxFrameX <= static_cast<double>(_frameWidth - 2) &&
          maxFrameY <= static_cast<double>(_frameHeight - 2),
  };
}

template <typename SampleConsumer>
void FrameSampler::sampleEachUprightLuminancePatchPixel(
    double uprightLeft,
    double uprightTop,
    int patchWidth,
    int patchHeight,
    SampleConsumer&& consumeSample) const {
  const LuminancePatchSamplingPlan plan = createLuminancePatchSamplingPlan(
      uprightLeft, uprightTop, patchWidth, patchHeight);

  std::size_t pixelIndex = 0;
  for (int y = 0; y < patchHeight; ++y) {
    const double rowFrameX =
        plan.frameTopLeft.x + _uprightFrameYStep.x * static_cast<double>(y);
    const double rowFrameY =
        plan.frameTopLeft.y + _uprightFrameYStep.y * static_cast<double>(y);
    for (int x = 0; x < patchWidth; ++x) {
      const double frameX =
          rowFrameX + _uprightFrameXStep.x * static_cast<double>(x);
      const double frameY =
          rowFrameY + _uprightFrameXStep.y * static_cast<double>(x);
      const float value = plan.canReadFrameWithoutBoundsChecks
                              ? sampleFrameLuminanceUnchecked(frameX, frameY)
                              : sampleFrameLuminance(frameX, frameY);
      consumeSample(pixelIndex, value);
      ++pixelIndex;
    }
  }
}

void FrameSampler::sampleUprightLuminancePatch(double uprightLeft,
                                               double uprightTop,
                                               int patchWidth,
                                               int patchHeight,
                                               std::span<float> output) const {
  checkedPatchPixelCount(patchWidth, patchHeight, output.size(), "output");
  sampleEachUprightLuminancePatchPixel(
      uprightLeft,
      uprightTop,
      patchWidth,
      patchHeight,
      [&output](std::size_t pixelIndex, float value) {
        output[pixelIndex] = value;
      });
}

double FrameSampler::scoreUprightLuminancePatch(
    double uprightLeft,
    double uprightTop,
    int patchWidth,
    int patchHeight,
    const LuminancePatchTemplate& reference,
    double minPatchNorm) const {
  const std::size_t patchPixelCount = checkedPatchPixelCount(
      patchWidth, patchHeight, reference.centeredPixels.size(), "reference");
  LuminancePatchScoreAccumulator score(reference);
  sampleEachUprightLuminancePatchPixel(
      uprightLeft,
      uprightTop,
      patchWidth,
      patchHeight,
      [&score](std::size_t pixelIndex, float value) {
        score.add(pixelIndex, value);
      });
  return score.finish(patchPixelCount, minPatchNorm);
}

FrameSampler::BgrColor FrameSampler::yuvToBgr(float luminance,
                                              float blueColor,
                                              float redColor) const {
  const float blueColorDelta = blueColor - 128.0f;
  const float redColorDelta = redColor - 128.0f;
  if (_isFullRange) {
    return BgrColor{
        clampBgrComponent(luminance + 1.772f * blueColorDelta),
        clampBgrComponent(luminance - 0.344136f * blueColorDelta -
                          0.714136f * redColorDelta),
        clampBgrComponent(luminance + 1.402f * redColorDelta),
    };
  }

  const float scaledLuminance = 1.164383f * (luminance - 16.0f);
  return BgrColor{
      clampBgrComponent(scaledLuminance + 2.017232f * blueColorDelta),
      clampBgrComponent(scaledLuminance - 0.391762f * blueColorDelta -
                        0.812968f * redColorDelta),
      clampBgrComponent(scaledLuminance + 1.596027f * redColorDelta),
  };
}

DetectorInputLayout FrameSampler::detectorInputLayout(int inputSize) const {
  const YuvPlane& luminance = _planes[0];
  const YuvPlane& blueColor = _planes[1];
  const YuvPlane& redColor = _hasInterleavedColor ? _planes[1] : _planes[2];
  return DetectorInputLayout{
      inputSize,
      _frameWidth,
      _frameHeight,
      _uprightWidth,
      _uprightHeight,
      luminance.bytesPerRow,
      blueColor.bytesPerRow,
      blueColor.pixelStride,
      redColor.bytesPerRow,
      redColor.pixelStride,
      _isMirrored,
      _isFullRange,
      _hasInterleavedColor,
      _orientation,
  };
}

void FrameSampler::rebuildDetectorInputCache(int inputSize,
                                             DetectorInputCache& cache) const {
  cache.layout = detectorInputLayout(inputSize);
  cache.transform.scale =
      std::min(static_cast<double>(inputSize) / _uprightWidth,
               static_cast<double>(inputSize) / _uprightHeight);
  const int contentWidth = std::min(
      inputSize,
      static_cast<int>(std::round(_uprightWidth * cache.transform.scale)));
  const int contentHeight = std::min(
      inputSize,
      static_cast<int>(std::round(_uprightHeight * cache.transform.scale)));
  cache.transform.offsetX = (inputSize - contentWidth) * 0.5;
  cache.transform.offsetY = (inputSize - contentHeight) * 0.5;
  cache.pixelMappings.clear();
  cache.pixelMappings.reserve(static_cast<std::size_t>(contentWidth) *
                              static_cast<std::size_t>(contentHeight));

  for (int y = 0; y < contentHeight; ++y) {
    const double uprightY = (y + 0.5) / cache.transform.scale - 0.5;
    const int destinationY = y + static_cast<int>(cache.transform.offsetY);
    for (int x = 0; x < contentWidth; ++x) {
      const double uprightX = (x + 0.5) / cache.transform.scale - 0.5;
      const Point framePoint = cachedUprightToFrame(uprightX, uprightY);
      const double frameX =
          std::clamp(framePoint.x, 0.0, static_cast<double>(_frameWidth - 1));
      const double frameY =
          std::clamp(framePoint.y, 0.0, static_cast<double>(_frameHeight - 1));
      const int framePixelX = static_cast<int>(std::round(frameX));
      const int framePixelY = static_cast<int>(std::round(frameY));
      const int colorPlaneX = std::min(framePixelX / 2, _colorPlaneWidth - 1);
      const int colorPlaneY = std::min(framePixelY / 2, _colorPlaneHeight - 1);
      const int destinationX = x + static_cast<int>(cache.transform.offsetX);
      const int destinationIndex = destinationY * inputSize + destinationX;
      const std::size_t luminanceOffset =
          static_cast<std::size_t>(framePixelY) *
              static_cast<std::size_t>(_planes[0].bytesPerRow) +
          static_cast<std::size_t>(framePixelX) *
              static_cast<std::size_t>(_planes[0].pixelStride);
      const std::size_t blueColorOffset =
          static_cast<std::size_t>(colorPlaneY) *
              static_cast<std::size_t>(_planes[1].bytesPerRow) +
          static_cast<std::size_t>(colorPlaneX) *
              static_cast<std::size_t>(_planes[1].pixelStride);
      const std::size_t redColorOffset =
          _hasInterleavedColor
              ? blueColorOffset + 1
              : static_cast<std::size_t>(colorPlaneY) *
                        static_cast<std::size_t>(_planes[2].bytesPerRow) +
                    static_cast<std::size_t>(colorPlaneX) *
                        static_cast<std::size_t>(_planes[2].pixelStride);
      cache.pixelMappings.push_back(DetectorInputPixelMapping{
          static_cast<std::uint32_t>(destinationIndex),
          static_cast<std::uint32_t>(luminanceOffset),
          static_cast<std::uint32_t>(blueColorOffset),
          static_cast<std::uint32_t>(redColorOffset),
      });
    }
  }
  cache.hasReusablePixelMappings = true;
}

FrameSampler::BgrColor FrameSampler::sampleFrame(double frameX,
                                                 double frameY) const {
  frameX = std::clamp(frameX, 0.0, static_cast<double>(_frameWidth - 1));
  frameY = std::clamp(frameY, 0.0, static_cast<double>(_frameHeight - 1));
  const float luminance =
      samplePlane(_planes[0], frameX, frameY, _frameWidth, _frameHeight, 0);

  const double colorPlaneX = frameX * 0.5;
  const double colorPlaneY = frameY * 0.5;
  const float blueColorValue = samplePlane(_planes[1],
                                           colorPlaneX,
                                           colorPlaneY,
                                           _colorPlaneWidth,
                                           _colorPlaneHeight,
                                           0);
  const float redColorValue = _hasInterleavedColor
                                  ? samplePlane(_planes[1],
                                                colorPlaneX,
                                                colorPlaneY,
                                                _colorPlaneWidth,
                                                _colorPlaneHeight,
                                                1)
                                  : samplePlane(_planes[2],
                                                colorPlaneX,
                                                colorPlaneY,
                                                _colorPlaneWidth,
                                                _colorPlaneHeight,
                                                0);
  return yuvToBgr(luminance, blueColorValue, redColorValue);
}

float FrameSampler::sampleFrameLuminance(double frameX, double frameY) const {
  frameX = std::clamp(frameX, 0.0, static_cast<double>(_frameWidth - 1));
  frameY = std::clamp(frameY, 0.0, static_cast<double>(_frameHeight - 1));
  return samplePlane(_planes[0], frameX, frameY, _frameWidth, _frameHeight, 0);
}

float FrameSampler::sampleFrameLuminanceUnchecked(double frameX,
                                                  double frameY) const {
  return samplePlaneUnchecked(_planes[0], frameX, frameY, 0);
}

Point FrameSampler::cachedUprightToFrame(double uprightX,
                                         double uprightY) const noexcept {
  return Point{
      _uprightFrameOrigin.x + _uprightFrameXStep.x * uprightX +
          _uprightFrameYStep.x * uprightY,
      _uprightFrameOrigin.y + _uprightFrameXStep.y * uprightX +
          _uprightFrameYStep.y * uprightY,
  };
}

Point FrameSampler::uprightToFrame(double uprightX, double uprightY) const {
  if (_isMirrored) {
    uprightX = static_cast<double>(_uprightWidth - 1) - uprightX;
  }
  switch (_orientation) {
    case margelo::nitro::camera::CameraOrientation::UP:
      return Point{uprightX, uprightY};
    case margelo::nitro::camera::CameraOrientation::RIGHT:
      return Point{static_cast<double>(_frameWidth - 1) - uprightY, uprightX};
    case margelo::nitro::camera::CameraOrientation::DOWN:
      return Point{
          static_cast<double>(_frameWidth - 1) - uprightX,
          static_cast<double>(_frameHeight - 1) - uprightY,
      };
    case margelo::nitro::camera::CameraOrientation::LEFT:
      return Point{uprightY, static_cast<double>(_frameHeight - 1) - uprightX};
  }
  throw std::invalid_argument(
      "FaceRecognizer received an unsupported frame orientation.");
}

float FrameSampler::samplePlane(const YuvPlane& plane,
                                double sampleX,
                                double sampleY,
                                int sampleWidth,
                                int sampleHeight,
                                int componentOffset) const {
  sampleX = std::clamp(sampleX, 0.0, static_cast<double>(sampleWidth - 1));
  sampleY = std::clamp(sampleY, 0.0, static_cast<double>(sampleHeight - 1));
  const int leftPixelX = static_cast<int>(std::floor(sampleX));
  const int topPixelY = static_cast<int>(std::floor(sampleY));
  const int rightPixelX = std::min(leftPixelX + 1, sampleWidth - 1);
  const int bottomPixelY = std::min(topPixelY + 1, sampleHeight - 1);
  const float horizontalWeight = static_cast<float>(sampleX - leftPixelX);
  const float verticalWeight = static_cast<float>(sampleY - topPixelY);
  const float top = readPlane(plane, leftPixelX, topPixelY, componentOffset) *
                        (1.0f - horizontalWeight) +
                    readPlane(plane, rightPixelX, topPixelY, componentOffset) *
                        horizontalWeight;
  const float bottom =
      readPlane(plane, leftPixelX, bottomPixelY, componentOffset) *
          (1.0f - horizontalWeight) +
      readPlane(plane, rightPixelX, bottomPixelY, componentOffset) *
          horizontalWeight;
  return top * (1.0f - verticalWeight) + bottom * verticalWeight;
}

float FrameSampler::samplePlaneUnchecked(const YuvPlane& plane,
                                         double sampleX,
                                         double sampleY,
                                         int componentOffset) const {
  const int leftPixelX = static_cast<int>(std::floor(sampleX));
  const int topPixelY = static_cast<int>(std::floor(sampleY));
  const int rightPixelX = leftPixelX + 1;
  const int bottomPixelY = topPixelY + 1;
  const float horizontalWeight = static_cast<float>(sampleX - leftPixelX);
  const float verticalWeight = static_cast<float>(sampleY - topPixelY);
  const float top = readPlane(plane, leftPixelX, topPixelY, componentOffset) *
                        (1.0f - horizontalWeight) +
                    readPlane(plane, rightPixelX, topPixelY, componentOffset) *
                        horizontalWeight;
  const float bottom =
      readPlane(plane, leftPixelX, bottomPixelY, componentOffset) *
          (1.0f - horizontalWeight) +
      readPlane(plane, rightPixelX, bottomPixelY, componentOffset) *
          horizontalWeight;
  return top * (1.0f - verticalWeight) + bottom * verticalWeight;
}

uint8_t FrameSampler::readPlane(const YuvPlane& plane,
                                int pixelX,
                                int pixelY,
                                int componentOffset) const {
  const size_t offset = static_cast<size_t>(pixelY) * plane.bytesPerRow +
                        static_cast<size_t>(pixelX) * plane.pixelStride +
                        componentOffset;
  return plane.data[offset];
}

}  // namespace margelo::nitro::facerecognizer
