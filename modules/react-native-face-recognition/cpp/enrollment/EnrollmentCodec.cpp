#include "EnrollmentCodec.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "EnrollmentMatcher.hpp"
#include "FaceConstants.hpp"

namespace margelo::nitro::facerecognizer {
namespace {

// Header: magic, version, sample count, and embedding length as little-endian
// uint32 values.
constexpr uint32_t kEnrollmentMagic = 0x45464352;
constexpr uint32_t kEnrollmentVersion = 1;
constexpr std::size_t kEnrollmentHeaderSize = 16;
constexpr std::size_t kFloat32ByteSize = sizeof(float);

uint32_t readUint32LittleEndian(const uint8_t* data) noexcept {
  return static_cast<uint32_t>(data[0]) |
         (static_cast<uint32_t>(data[1]) << 8) |
         (static_cast<uint32_t>(data[2]) << 16) |
         (static_cast<uint32_t>(data[3]) << 24);
}

void writeUint32LittleEndian(std::vector<uint8_t>& data,
                             std::size_t offset,
                             uint32_t value) noexcept {
  data[offset] = static_cast<uint8_t>(value & 0xff);
  data[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xff);
  data[offset + 2] = static_cast<uint8_t>((value >> 16) & 0xff);
  data[offset + 3] = static_cast<uint8_t>((value >> 24) & 0xff);
}

}  // namespace

std::shared_ptr<margelo::nitro::ArrayBuffer> serializeEnrollment(
    const EnrollmentSamples& samples) {
  if (samples.empty()) {
    throw std::runtime_error(
        "FaceRecognizer cannot serialize an enrollment with no templates.");
  }
  const std::size_t embeddingLength = samples.front().size();
  if (embeddingLength == 0) {
    throw std::runtime_error(
        "FaceRecognizer cannot serialize zero-length enrollment templates.");
  }
  if (samples.size() > kMaxSamplesPerEnrollment) {
    throw std::runtime_error("FaceRecognizer cannot serialize more than " +
                             std::to_string(kMaxSamplesPerEnrollment) +
                             " enrollment templates for one identity.");
  }
  if (samples.size() > std::numeric_limits<uint32_t>::max() ||
      embeddingLength > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        "FaceRecognizer enrollment is too large to serialize.");
  }
  if (embeddingLength >
      (std::numeric_limits<std::size_t>::max() / samples.size()) /
          kFloat32ByteSize) {
    throw std::runtime_error(
        "FaceRecognizer enrollment byte size overflowed during serialization.");
  }

  const std::size_t valueCount = samples.size() * embeddingLength;
  const std::size_t byteCount =
      kEnrollmentHeaderSize + valueCount * kFloat32ByteSize;
  std::vector<uint8_t> bytes(byteCount);
  writeUint32LittleEndian(bytes, 0, kEnrollmentMagic);
  writeUint32LittleEndian(bytes, 4, kEnrollmentVersion);
  writeUint32LittleEndian(bytes, 8, static_cast<uint32_t>(samples.size()));
  writeUint32LittleEndian(bytes, 12, static_cast<uint32_t>(embeddingLength));

  std::size_t offset = kEnrollmentHeaderSize;
  for (const FaceEmbedding& sample : samples) {
    if (sample.size() != embeddingLength) {
      throw std::runtime_error(
          "FaceRecognizer cannot serialize enrollment templates with mixed "
          "dimensions.");
    }
    std::memcpy(
        bytes.data() + offset, sample.data(), sample.size() * kFloat32ByteSize);
    offset += sample.size() * kFloat32ByteSize;
  }
  return margelo::nitro::ArrayBuffer::move(std::move(bytes));
}

EnrollmentSamples deserializeEnrollment(
    const std::shared_ptr<margelo::nitro::ArrayBuffer>& serializedEnrollment) {
  if (serializedEnrollment == nullptr ||
      serializedEnrollment->data() == nullptr) {
    throw std::invalid_argument(
        "FaceRecognizer addEnrollment received an empty enrollment buffer.");
  }
  const std::size_t byteCount = serializedEnrollment->size();
  if (byteCount < kEnrollmentHeaderSize) {
    throw std::invalid_argument(
        "FaceRecognizer addEnrollment received an enrollment buffer smaller "
        "than the header.");
  }
  const uint8_t* data = serializedEnrollment->data();
  const uint32_t magic = readUint32LittleEndian(data);
  const uint32_t version = readUint32LittleEndian(data + 4);
  const uint32_t sampleCount = readUint32LittleEndian(data + 8);
  const uint32_t embeddingLength = readUint32LittleEndian(data + 12);

  if (magic != kEnrollmentMagic) {
    throw std::invalid_argument(
        "FaceRecognizer addEnrollment received an unsupported enrollment "
        "buffer format.");
  }
  if (version != kEnrollmentVersion) {
    throw std::invalid_argument(
        "FaceRecognizer addEnrollment received unsupported enrollment "
        "version " +
        std::to_string(version) + ".");
  }
  if (sampleCount == 0 || sampleCount > kMaxSamplesPerEnrollment) {
    throw std::invalid_argument("FaceRecognizer addEnrollment received " +
                                std::to_string(sampleCount) +
                                " templates; expected between 1 and " +
                                std::to_string(kMaxSamplesPerEnrollment) + ".");
  }
  if (embeddingLength == 0) {
    throw std::invalid_argument(
        "FaceRecognizer addEnrollment received zero-dimensional templates.");
  }
  if (static_cast<std::size_t>(embeddingLength) >
      (std::numeric_limits<std::size_t>::max() /
       static_cast<std::size_t>(sampleCount)) /
          kFloat32ByteSize) {
    throw std::invalid_argument(
        "FaceRecognizer addEnrollment received an enrollment buffer whose "
        "declared size overflows this platform.");
  }

  const std::size_t valueCount = static_cast<std::size_t>(sampleCount) *
                                 static_cast<std::size_t>(embeddingLength);
  const std::size_t expectedByteCount =
      kEnrollmentHeaderSize + valueCount * kFloat32ByteSize;
  if (byteCount != expectedByteCount) {
    throw std::invalid_argument(
        "FaceRecognizer addEnrollment received " + std::to_string(byteCount) +
        " bytes but expected " + std::to_string(expectedByteCount) +
        " bytes for " + std::to_string(sampleCount) + " templates with " +
        std::to_string(embeddingLength) + " values each.");
  }

  EnrollmentSamples samples;
  samples.reserve(sampleCount);
  std::size_t offset = kEnrollmentHeaderSize;
  for (uint32_t sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
    FaceEmbedding sample(embeddingLength);
    std::memcpy(sample.data(),
                data + offset,
                static_cast<std::size_t>(embeddingLength) * kFloat32ByteSize);
    samples.push_back(normalizeStoredEmbedding(std::move(sample)));
    offset += static_cast<std::size_t>(embeddingLength) * kFloat32ByteSize;
  }
  return samples;
}

}  // namespace margelo::nitro::facerecognizer
