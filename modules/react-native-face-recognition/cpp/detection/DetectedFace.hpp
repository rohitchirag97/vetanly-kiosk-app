#pragma once

#include <array>

#include "FaceConstants.hpp"
#include "Point.hpp"
#include "Rect.hpp"

namespace margelo::nitro::facerecognizer {

struct DetectedFace final {
  Rect bounds;
  double confidence;
  std::array<Point, kFaceLandmarkCount> landmarks;
};

}  // namespace margelo::nitro::facerecognizer
