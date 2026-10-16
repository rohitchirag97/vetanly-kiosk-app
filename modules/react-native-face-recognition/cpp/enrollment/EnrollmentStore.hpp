#pragma once

#include <NitroModules/ArrayBuffer.hpp>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "EnrollFaceStatus.hpp"
#include "EnrollmentMatcher.hpp"

namespace margelo::nitro::facerecognizer {

/**
 * Mutable in-memory enrollment index with precomputed identity centroids.
 *
 * The store owns normalized embedding samples. Callers serialize/deserialize
 * through `EnrollmentCodec` when crossing the JS persistence boundary.
 */
class EnrollmentStore final {
public:
  void importSerialized(
      const std::string& enrollmentId,
      const std::shared_ptr<margelo::nitro::ArrayBuffer>& serializedEnrollment,
      const std::optional<std::size_t>& expectedEmbeddingLength);
  std::shared_ptr<margelo::nitro::ArrayBuffer> exportSerialized(
      const std::string& enrollmentId) const;
  std::vector<std::string> ids() const;
  void remove(const std::string& enrollmentId);
  void clear() noexcept;

  EnrollFaceStatus addSample(const std::string& enrollmentId,
                             FaceEmbedding faceEmbedding);
  std::optional<EnrollmentMatch> findBestMatch(
      const FaceEmbedding& faceEmbedding,
      float threshold,
      float minimumMargin,
      CandidateRetrievalMode candidateRetrieval) const;
  std::optional<EnrollmentMatch> findBestMatch(
      std::span<const FaceEmbedding> faceEmbeddings,
      float threshold,
      float minimumMargin,
      CandidateRetrievalMode candidateRetrieval) const;

  bool empty() const noexcept;
  const EnrollmentIndex& index() const noexcept;
  std::size_t externalMemorySize() const noexcept;

private:
  EnrollmentIndex _enrollments;
};

void validateEnrollmentId(const std::string& enrollmentId);

}  // namespace margelo::nitro::facerecognizer
