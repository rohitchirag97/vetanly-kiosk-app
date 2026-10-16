#include "EnrollmentStore.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "BufferMemory.hpp"
#include "EnrollmentCodec.hpp"
#include "FaceConstants.hpp"

namespace margelo::nitro::facerecognizer {

void validateEnrollmentId(const std::string& enrollmentId) {
  if (enrollmentId.empty()) {
    throw std::invalid_argument(
        "FaceRecognizer enrollment requires a non-empty id.");
  }
}

void EnrollmentStore::importSerialized(
    const std::string& enrollmentId,
    const std::shared_ptr<margelo::nitro::ArrayBuffer>& serializedEnrollment,
    const std::optional<std::size_t>& expectedEmbeddingLength) {
  validateEnrollmentId(enrollmentId);

  EnrollmentSamples samples = deserializeEnrollment(serializedEnrollment);
  samples.reserve(kMaxSamplesPerEnrollment);
  if (expectedEmbeddingLength.has_value()) {
    for (const FaceEmbedding& sample : samples) {
      if (sample.size() != *expectedEmbeddingLength) {
        throw std::runtime_error(
            "FaceRecognizer addEnrollment received a template with " +
            std::to_string(sample.size()) +
            " values but the recognizer model produces " +
            std::to_string(*expectedEmbeddingLength) +
            "-dimensional embeddings.");
      }
    }
  }

  FaceEmbedding centroid =
      createNormalizedCentroid(samples, "enrollment '" + enrollmentId + "'");
  _enrollments.insert_or_assign(
      enrollmentId, EnrollmentRecord{std::move(samples), std::move(centroid)});
}

std::shared_ptr<margelo::nitro::ArrayBuffer> EnrollmentStore::exportSerialized(
    const std::string& enrollmentId) const {
  const EnrollmentIndex::const_iterator entry = _enrollments.find(enrollmentId);
  if (entry == _enrollments.end()) {
    throw std::invalid_argument("FaceRecognizer has no enrollment for id '" +
                                enrollmentId + "'.");
  }
  return serializeEnrollment(entry->second.samples);
}

std::vector<std::string> EnrollmentStore::ids() const {
  std::vector<std::string> enrollmentIds;
  enrollmentIds.reserve(_enrollments.size());
  for (const EnrollmentIndex::value_type& entry : _enrollments) {
    enrollmentIds.push_back(entry.first);
  }
  return enrollmentIds;
}

void EnrollmentStore::remove(const std::string& enrollmentId) {
  _enrollments.erase(enrollmentId);
  if (_enrollments.empty()) {
    clear();
  }
}

void EnrollmentStore::clear() noexcept {
  EnrollmentIndex().swap(_enrollments);
}

EnrollFaceStatus EnrollmentStore::addSample(const std::string& enrollmentId,
                                            FaceEmbedding faceEmbedding) {
  validateEnrollmentId(enrollmentId);
  const EnrollmentIndex::iterator existing = _enrollments.find(enrollmentId);
  if (existing == _enrollments.end()) {
    std::optional<EnrollmentSampleUpdate> update =
        prepareEnrollmentSampleUpdate({}, faceEmbedding);
    if (!update.has_value()) {
      throw std::logic_error(
          "FaceRecognizer rejected the first enrollment sample as redundant.");
    }
    EnrollmentSamples samples;
    samples.reserve(kMaxSamplesPerEnrollment);
    samples.push_back(std::move(faceEmbedding));
    _enrollments.try_emplace(
        enrollmentId,
        EnrollmentRecord{std::move(samples), std::move(update->centroid)});
    return EnrollFaceStatus::ENROLLED;
  }

  EnrollmentRecord& record = existing->second;
  std::optional<EnrollmentSampleUpdate> update =
      prepareEnrollmentSampleUpdate(record.samples, faceEmbedding);
  if (!update.has_value()) {
    return EnrollFaceStatus::REDUNDANT;
  }
  if (update->evictionIndex.has_value()) {
    record.samples.erase(record.samples.begin() +
                         static_cast<std::ptrdiff_t>(*update->evictionIndex));
  }
  record.samples.push_back(std::move(faceEmbedding));
  record.centroid = std::move(update->centroid);
  return EnrollFaceStatus::ENROLLED;
}

std::optional<EnrollmentMatch> EnrollmentStore::findBestMatch(
    const FaceEmbedding& faceEmbedding,
    float threshold,
    float minimumMargin,
    CandidateRetrievalMode candidateRetrieval) const {
  return findBestEnrollmentMatch(
      faceEmbedding,
      _enrollments,
      threshold,
      minimumMargin,
      candidateRetrieval);
}

std::optional<EnrollmentMatch> EnrollmentStore::findBestMatch(
    std::span<const FaceEmbedding> faceEmbeddings,
    float threshold,
    float minimumMargin,
    CandidateRetrievalMode candidateRetrieval) const {
  return findBestEnrollmentMatch(
      faceEmbeddings,
      _enrollments,
      threshold,
      minimumMargin,
      candidateRetrieval);
}

bool EnrollmentStore::empty() const noexcept {
  return _enrollments.empty();
}

const EnrollmentIndex& EnrollmentStore::index() const noexcept {
  return _enrollments;
}

std::size_t EnrollmentStore::externalMemorySize() const noexcept {
  std::size_t byteSize = 0;
  for (const EnrollmentIndex::value_type& entry : _enrollments) {
    byteSize += entry.first.capacity();
    byteSize += vectorCapacityByteSize(entry.second.samples);
    for (const FaceEmbedding& sample : entry.second.samples) {
      byteSize += vectorCapacityByteSize(sample);
    }
    byteSize += vectorCapacityByteSize(entry.second.centroid);
  }
  return byteSize;
}

}  // namespace margelo::nitro::facerecognizer
