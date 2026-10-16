#pragma once

namespace margelo::nitro::facerecognizer {

/**
 * Scale and padding used when fitting a frame into the square detector input.
 */
struct LetterboxTransform final {
  double scale = 0.0;
  double offsetX = 0.0;
  double offsetY = 0.0;
};

}  // namespace margelo::nitro::facerecognizer
