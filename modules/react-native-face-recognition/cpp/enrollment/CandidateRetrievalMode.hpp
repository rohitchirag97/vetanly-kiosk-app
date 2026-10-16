#pragma once

namespace margelo::nitro::facerecognizer {

/** Coarse identity retrieval policy used before template verification. */
enum class CandidateRetrievalMode {
  CENTROID,
  TEMPLATE_MAX,
};

}  // namespace margelo::nitro::facerecognizer
