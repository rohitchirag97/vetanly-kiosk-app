#pragma once

#include "CandidateRetrievalMode.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace margelo::nitro::facerecognizer {

/** Unit-normalized face embedding vector produced by the recognizer model. */
using FaceEmbedding = std::vector<float>;
/** Diverse sample set for one enrollment id. */
using EnrollmentSamples = std::vector<FaceEmbedding>;
/** Enrollment samples plus their precomputed normalized centroid. */
struct EnrollmentRecord final {
  EnrollmentSamples samples;
  FaceEmbedding centroid;
};
/** In-memory enrollment index keyed by application-provided identity id. */
using EnrollmentIndex = std::unordered_map<std::string, EnrollmentRecord>;

/** Best enrollment candidate accepted by the configured decision policy. */
struct EnrollmentMatch final {
  std::string_view enrollmentId;
  float similarity;
};

/** Prepared mutation for adding one diverse sample without cloning the set. */
struct EnrollmentSampleUpdate final {
  FaceEmbedding centroid;
  std::optional<std::size_t> evictionIndex;
};

/**
 * Normalizes raw model output into `embedding` without reallocating when
 * possible.
 */
void normalizeEmbeddingInto(const float* values,
                            std::size_t length,
                            FaceEmbedding& embedding);
/** Validates and normalizes an embedding loaded from external storage. */
FaceEmbedding normalizeStoredEmbedding(FaceEmbedding embedding);
/** Creates one unit-normalized centroid from compatible normalized vectors. */
FaceEmbedding createNormalizedCentroid(
    std::span<const FaceEmbedding> embeddings, std::string_view errorContext);
/** Cosine similarity for already normalized embeddings. */
float cosineSimilarity(const FaceEmbedding& embedding,
                       const FaceEmbedding& otherEmbedding);
/**
 * Retrieves candidate identities according to `candidateRetrieval`, then
 * verifies each candidate against individual templates before threshold/margin.
 */
std::optional<EnrollmentMatch> findBestEnrollmentMatch(
    const FaceEmbedding& faceEmbedding,
    const EnrollmentIndex& enrollments,
    float threshold,
    float minimumMargin,
    CandidateRetrievalMode candidateRetrieval);
/**
 * Uses `candidateRetrieval` for coarse retrieval, then requires a strict
 * majority of per-probe template matches to agree on one enrollment.
 */
std::optional<EnrollmentMatch> findBestEnrollmentMatch(
    std::span<const FaceEmbedding> faceEmbeddings,
    const EnrollmentIndex& enrollments,
    float threshold,
    float minimumMargin,
    CandidateRetrievalMode candidateRetrieval);
/**
 * Prepares a diverse sample update, or returns nullopt for a near-duplicate.
 */
std::optional<EnrollmentSampleUpdate> prepareEnrollmentSampleUpdate(
    const EnrollmentSamples& samples, const FaceEmbedding& faceEmbedding);

}  // namespace margelo::nitro::facerecognizer
