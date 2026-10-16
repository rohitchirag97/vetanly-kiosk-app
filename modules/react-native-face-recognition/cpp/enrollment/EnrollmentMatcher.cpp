#include "EnrollmentMatcher.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "FaceConstants.hpp"

namespace margelo::nitro::facerecognizer {
namespace {

// Coarse retrieval is deliberately followed by exact template verification.
// Keep this in sync with the candidate-recall benchmark.
constexpr std::size_t kTemplateVerificationCandidateCount = 15;

float checkedInverseNorm(std::span<const float> values) {
  constexpr double kMinSquaredNorm = 1e-12;
  double squaredNorm = 0.0;
  for (const float value : values) {
    squaredNorm += static_cast<double>(value) * value;
  }
  if (squaredNorm <= kMinSquaredNorm) {
    throw std::runtime_error(
        "FaceRecognizer produced a zero-length face embedding.");
  }
  return static_cast<float>(1.0 / std::sqrt(squaredNorm));
}

std::size_t mostRedundantSampleIndex(const EnrollmentSamples& samples) {
  std::size_t evictIndex = 0;
  float highestRedundancy = -std::numeric_limits<float>::infinity();
  for (std::size_t index = 0; index < samples.size(); ++index) {
    float redundancy = 0.0f;
    for (std::size_t otherIndex = 0; otherIndex < samples.size();
         ++otherIndex) {
      if (otherIndex == index) {
        continue;
      }
      redundancy += cosineSimilarity(samples[index], samples[otherIndex]);
    }
    if (redundancy > highestRedundancy) {
      highestRedundancy = redundancy;
      evictIndex = index;
    }
  }
  return evictIndex;
}

void addEmbeddingInto(FaceEmbedding& sum,
                      const FaceEmbedding& embedding,
                      std::string_view errorContext) {
  if (embedding.size() != sum.size()) {
    throw std::runtime_error(
        "FaceRecognizer " + std::string(errorContext) +
        " contains incompatible embedding sizes; expected " +
        std::to_string(sum.size()) + " values, received " +
        std::to_string(embedding.size()) + ".");
  }
  for (std::size_t index = 0; index < embedding.size(); ++index) {
    sum[index] += embedding[index];
  }
}

void normalizeEmbeddingSum(FaceEmbedding& sum) {
  const float inverseNorm = checkedInverseNorm(sum);
  for (float& value : sum) {
    value *= inverseNorm;
  }
}

float topTwoEnrollmentSimilarity(const FaceEmbedding& faceEmbedding,
                                 const EnrollmentSamples& samples,
                                 std::string_view enrollmentId) {
  if (samples.empty()) {
    throw std::runtime_error("FaceRecognizer enrollment '" +
                             std::string(enrollmentId) +
                             "' contains no templates.");
  }

  float bestSimilarity = std::numeric_limits<float>::lowest();
  float secondBestSimilarity = std::numeric_limits<float>::lowest();
  for (const FaceEmbedding& sample : samples) {
    if (sample.size() != faceEmbedding.size()) {
      throw std::runtime_error(
          "FaceRecognizer enrollment '" + std::string(enrollmentId) +
          "' contains an incompatible template size; expected " +
          std::to_string(faceEmbedding.size()) + " values, received " +
          std::to_string(sample.size()) + ".");
    }
    const float similarity = cosineSimilarity(faceEmbedding, sample);
    if (similarity > bestSimilarity) {
      secondBestSimilarity = bestSimilarity;
      bestSimilarity = similarity;
    } else if (similarity > secondBestSimilarity) {
      secondBestSimilarity = similarity;
    }
  }
  return samples.size() == 1 ? bestSimilarity
                             : (bestSimilarity + secondBestSimilarity) * 0.5f;
}

struct EnrollmentCandidate final {
  const EnrollmentIndex::value_type* enrollment;
  float retrievalSimilarity;
};

void addCandidate(
    std::vector<EnrollmentCandidate>& candidates,
    const EnrollmentIndex::value_type& enrollment,
    float retrievalSimilarity) {
  const auto insertionPoint = std::lower_bound(
      candidates.begin(), candidates.end(), retrievalSimilarity,
      [](const EnrollmentCandidate& candidate, float similarity) {
        return candidate.retrievalSimilarity > similarity;
      });
  candidates.insert(insertionPoint, {&enrollment, retrievalSimilarity});
  if (candidates.size() > kTemplateVerificationCandidateCount) {
    candidates.pop_back();
  }
}

std::vector<EnrollmentCandidate> findCentroidCandidates(
    const FaceEmbedding& faceEmbedding, const EnrollmentIndex& enrollments) {
  std::vector<EnrollmentCandidate> candidates;
  candidates.reserve(
      std::min(enrollments.size(), kTemplateVerificationCandidateCount));
  for (const EnrollmentIndex::value_type& enrollment : enrollments) {
    const FaceEmbedding& centroid = enrollment.second.centroid;
    if (centroid.size() != faceEmbedding.size()) {
      throw std::runtime_error(
          "FaceRecognizer enrollment '" + enrollment.first +
          "' contains an incompatible centroid size; expected " +
          std::to_string(faceEmbedding.size()) + " values, received " +
          std::to_string(centroid.size()) + ".");
    }
    addCandidate(candidates,
                 enrollment,
                 cosineSimilarity(faceEmbedding, centroid));
  }
  return candidates;
}

std::vector<EnrollmentCandidate> findTemplateMaxCandidates(
    std::span<const FaceEmbedding> faceEmbeddings,
    const EnrollmentIndex& enrollments) {
  std::vector<EnrollmentCandidate> candidates;
  candidates.reserve(
      std::min(enrollments.size(), kTemplateVerificationCandidateCount));
  for (const EnrollmentIndex::value_type& enrollment : enrollments) {
    if (enrollment.second.samples.empty()) {
      throw std::runtime_error("FaceRecognizer enrollment '" +
                               enrollment.first +
                               "' contains no templates.");
    }
    float bestSimilarity = std::numeric_limits<float>::lowest();
    for (const FaceEmbedding& faceEmbedding : faceEmbeddings) {
      for (const FaceEmbedding& sample : enrollment.second.samples) {
        if (sample.size() != faceEmbedding.size()) {
          throw std::runtime_error(
              "FaceRecognizer enrollment '" + enrollment.first +
              "' contains an incompatible template size; expected " +
              std::to_string(faceEmbedding.size()) + " values, received " +
              std::to_string(sample.size()) + ".");
        }
        bestSimilarity = std::max(
            bestSimilarity, cosineSimilarity(faceEmbedding, sample));
      }
    }
    addCandidate(candidates, enrollment, bestSimilarity);
  }
  return candidates;
}

std::vector<EnrollmentCandidate> findCandidates(
    std::span<const FaceEmbedding> faceEmbeddings,
    const EnrollmentIndex& enrollments,
    CandidateRetrievalMode candidateRetrieval) {
  switch (candidateRetrieval) {
    case CandidateRetrievalMode::CENTROID: {
      const FaceEmbedding centroid =
          createNormalizedCentroid(faceEmbeddings, "recognition sample set");
      return findCentroidCandidates(centroid, enrollments);
    }
    case CandidateRetrievalMode::TEMPLATE_MAX:
      return findTemplateMaxCandidates(faceEmbeddings, enrollments);
  }
  throw std::logic_error("FaceRecognizer received an unknown candidate "
                         "retrieval mode.");
}

float medianSimilarity(std::vector<float> similarities) {
  const std::size_t upperMiddleIndex = similarities.size() / 2;
  auto upperMiddle = similarities.begin() +
                     static_cast<std::ptrdiff_t>(upperMiddleIndex);
  std::nth_element(similarities.begin(), upperMiddle, similarities.end());
  if (similarities.size() % 2 != 0) {
    return *upperMiddle;
  }
  const float lowerMiddle = *std::max_element(similarities.begin(), upperMiddle);
  return (lowerMiddle + *upperMiddle) * 0.5f;
}

}  // namespace

void normalizeEmbeddingInto(const float* values,
                            std::size_t length,
                            FaceEmbedding& embedding) {
  const float inverseNorm = checkedInverseNorm({values, length});
  embedding.resize(length);
  for (std::size_t index = 0; index < length; ++index) {
    embedding[index] = values[index] * inverseNorm;
  }
}

FaceEmbedding normalizeStoredEmbedding(FaceEmbedding embedding) {
  double squaredNorm = 0.0;
  for (const float value : embedding) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(
          "FaceRecognizer addEnrollment received a template with non-finite "
          "values.");
    }
    squaredNorm += static_cast<double>(value) * static_cast<double>(value);
  }
  if (squaredNorm <= 1e-12) {
    throw std::invalid_argument(
        "FaceRecognizer addEnrollment received a zero-length template.");
  }
  const float inverseNorm = static_cast<float>(1.0 / std::sqrt(squaredNorm));
  for (float& value : embedding) {
    value *= inverseNorm;
  }
  return embedding;
}

FaceEmbedding createNormalizedCentroid(
    std::span<const FaceEmbedding> embeddings, std::string_view errorContext) {
  if (embeddings.empty()) {
    throw std::invalid_argument("FaceRecognizer " + std::string(errorContext) +
                                " contains no embeddings.");
  }
  FaceEmbedding sum(embeddings.front().size(), 0.0f);
  for (const FaceEmbedding& embedding : embeddings) {
    addEmbeddingInto(sum, embedding, errorContext);
  }
  normalizeEmbeddingSum(sum);
  return sum;
}

float cosineSimilarity(const FaceEmbedding& embedding,
                       const FaceEmbedding& otherEmbedding) {
  return std::inner_product(
      embedding.begin(), embedding.end(), otherEmbedding.begin(), 0.0f);
}

std::optional<EnrollmentMatch> findBestEnrollmentMatch(
    const FaceEmbedding& faceEmbedding,
    const EnrollmentIndex& enrollments,
    float threshold,
    float minimumMargin,
    CandidateRetrievalMode candidateRetrieval) {
  float bestSimilarity = std::numeric_limits<float>::lowest();
  float runnerUpSimilarity = std::numeric_limits<float>::lowest();
  const std::string* bestEnrollmentId = nullptr;
  const std::vector<EnrollmentCandidate> candidates =
      findCandidates(std::span<const FaceEmbedding>(&faceEmbedding, 1),
                     enrollments,
                     candidateRetrieval);
  for (const EnrollmentCandidate& candidate : candidates) {
    const EnrollmentIndex::value_type& enrollment = *candidate.enrollment;
    const float similarity = topTwoEnrollmentSimilarity(
        faceEmbedding, enrollment.second.samples, enrollment.first);
    if (similarity > bestSimilarity) {
      runnerUpSimilarity = bestSimilarity;
      bestSimilarity = similarity;
      bestEnrollmentId = &enrollment.first;
    } else if (similarity > runnerUpSimilarity) {
      runnerUpSimilarity = similarity;
    }
  }
  if (bestEnrollmentId == nullptr || bestSimilarity < threshold ||
      (candidates.size() > 1 &&
       bestSimilarity - runnerUpSimilarity < minimumMargin)) {
    return std::nullopt;
  }
  return EnrollmentMatch{std::string_view(*bestEnrollmentId), bestSimilarity};
}

std::optional<EnrollmentMatch> findBestEnrollmentMatch(
    std::span<const FaceEmbedding> faceEmbeddings,
    const EnrollmentIndex& enrollments,
    float threshold,
    float minimumMargin,
    CandidateRetrievalMode candidateRetrieval) {
  if (faceEmbeddings.empty()) {
    throw std::invalid_argument(
        "FaceRecognizer recognition sample set contains no embeddings.");
  }
  const std::vector<EnrollmentCandidate> candidates =
      findCandidates(faceEmbeddings, enrollments, candidateRetrieval);
  if (candidates.empty()) {
    return std::nullopt;
  }

  std::vector<std::vector<float>> candidateSimilarities(candidates.size());
  std::vector<std::size_t> voteCounts(candidates.size(), 0);
  for (std::vector<float>& similarities : candidateSimilarities) {
    similarities.reserve(faceEmbeddings.size());
  }
  for (const FaceEmbedding& faceEmbedding : faceEmbeddings) {
    float bestSimilarity = std::numeric_limits<float>::lowest();
    float runnerUpSimilarity = std::numeric_limits<float>::lowest();
    std::size_t winningCandidateIndex = candidates.size();
    for (std::size_t candidateIndex = 0; candidateIndex < candidates.size();
         ++candidateIndex) {
      const EnrollmentIndex::value_type& enrollment =
          *candidates[candidateIndex].enrollment;
      const float similarity = topTwoEnrollmentSimilarity(
          faceEmbedding, enrollment.second.samples, enrollment.first);
      candidateSimilarities[candidateIndex].push_back(similarity);
      if (similarity > bestSimilarity) {
        runnerUpSimilarity = bestSimilarity;
        bestSimilarity = similarity;
        winningCandidateIndex = candidateIndex;
      } else if (similarity > runnerUpSimilarity) {
        runnerUpSimilarity = similarity;
      }
    }
    if (bestSimilarity >= threshold &&
        (candidates.size() == 1 ||
         bestSimilarity - runnerUpSimilarity >= minimumMargin)) {
      ++voteCounts[winningCandidateIndex];
    }
  }

  const std::size_t requiredVotes = faceEmbeddings.size() / 2 + 1;
  std::size_t winningCandidateIndex = candidates.size();
  std::size_t highestVoteCount = 0;
  bool isVoteTie = false;
  for (std::size_t candidateIndex = 0; candidateIndex < candidates.size();
       ++candidateIndex) {
    if (voteCounts[candidateIndex] > highestVoteCount) {
      highestVoteCount = voteCounts[candidateIndex];
      winningCandidateIndex = candidateIndex;
      isVoteTie = false;
    } else if (voteCounts[candidateIndex] == highestVoteCount &&
               highestVoteCount > 0) {
      isVoteTie = true;
    }
  }
  if (winningCandidateIndex == candidates.size() || isVoteTie ||
      highestVoteCount < requiredVotes) {
    return std::nullopt;
  }

  const EnrollmentIndex::value_type& winningEnrollment =
      *candidates[winningCandidateIndex].enrollment;
  return EnrollmentMatch{
      std::string_view(winningEnrollment.first),
      medianSimilarity(std::move(candidateSimilarities[winningCandidateIndex]))};
}

std::optional<EnrollmentSampleUpdate> prepareEnrollmentSampleUpdate(
    const EnrollmentSamples& samples, const FaceEmbedding& faceEmbedding) {
  if (samples.size() > kMaxSamplesPerEnrollment) {
    throw std::logic_error(
        "FaceRecognizer enrollment contains more samples than supported.");
  }
  for (const FaceEmbedding& sample : samples) {
    if (sample.size() != faceEmbedding.size()) {
      throw std::invalid_argument(
          "FaceRecognizer cannot combine enrollment embeddings with "
          "different sizes.");
    }
    if (cosineSimilarity(faceEmbedding, sample) >= kEnrollDuplicateSimilarity) {
      return std::nullopt;
    }
  }

  const std::optional<std::size_t> evictionIndex =
      samples.size() == kMaxSamplesPerEnrollment
          ? std::optional<std::size_t>(mostRedundantSampleIndex(samples))
          : std::nullopt;
  FaceEmbedding centroid(faceEmbedding.size(), 0.0f);
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (!evictionIndex.has_value() || index != *evictionIndex) {
      addEmbeddingInto(centroid, samples[index], "enrollment sample set");
    }
  }
  addEmbeddingInto(centroid, faceEmbedding, "enrollment sample set");
  normalizeEmbeddingSum(centroid);
  return EnrollmentSampleUpdate{std::move(centroid), evictionIndex};
}

}  // namespace margelo::nitro::facerecognizer
