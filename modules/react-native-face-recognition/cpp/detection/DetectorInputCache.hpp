#pragma once

#include <VisionCamera/HybridFrameSpec.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "LetterboxTransform.hpp"

namespace margelo::nitro::facerecognizer {

/** Mapping from one detector tensor pixel to YUV plane offsets. */
struct DetectorInputPixelMapping final {
  std::uint32_t destinationIndex = 0;
  std::uint32_t luminanceOffset = 0;
  std::uint32_t blueColorOffset = 0;
  std::uint32_t redColorOffset = 0;
};

static_assert(sizeof(DetectorInputPixelMapping) == sizeof(std::uint32_t) * 4);

/** Snapshot used to invalidate cached detector input offsets. */
struct DetectorInputLayout final {
  int inputSize = 0;
  int frameWidth = 0;
  int frameHeight = 0;
  int uprightWidth = 0;
  int uprightHeight = 0;
  int luminanceBytesPerRow = 0;
  int blueColorBytesPerRow = 0;
  int blueColorPixelStride = 0;
  int redColorBytesPerRow = 0;
  int redColorPixelStride = 0;
  bool isMirrored = false;
  bool isFullRange = false;
  bool hasInterleavedColor = false;
  margelo::nitro::camera::CameraOrientation orientation =
      margelo::nitro::camera::CameraOrientation::UP;

  bool operator==(const DetectorInputLayout&) const noexcept = default;
};

struct DetectorInputCache final {
  DetectorInputLayout layout;
  LetterboxTransform transform;
  std::vector<DetectorInputPixelMapping> pixelMappings;
  bool hasReusablePixelMappings = false;

  void releaseMemory() noexcept {
    std::vector<DetectorInputPixelMapping>().swap(pixelMappings);
    layout = DetectorInputLayout{};
    transform = LetterboxTransform{};
    hasReusablePixelMappings = false;
  }
};

}  // namespace margelo::nitro::facerecognizer
