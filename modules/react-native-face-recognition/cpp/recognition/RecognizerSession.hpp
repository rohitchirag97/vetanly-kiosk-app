#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "DetectedFace.hpp"
#include "EnrollmentMatcher.hpp"
#include "FaceEngineConfig.hpp"
#include "onnxruntime_cxx_api.h"

namespace margelo::nitro::facerecognizer {

class FrameSampler;

/**
 * Lazy-loaded face embedding model session.
 *
 * The session owns reusable aligned-input and embedding-output buffers. It is
 * loaded only when enrollments exist or enrollment capture needs an embedding.
 */
class RecognizerSession final {
public:
  RecognizerSession() noexcept;

  RecognizerSession(const RecognizerSession&) = delete;
  RecognizerSession& operator=(const RecognizerSession&) = delete;
  RecognizerSession(RecognizerSession&&) = delete;
  RecognizerSession& operator=(RecognizerSession&&) = delete;

  /**
   * Loads the recognizer model and validates embedding size against existing
   * enrollments.
   */
  void ensureLoaded(const FaceEngineConfig& config,
                    const Ort::MemoryInfo& memoryInfo,
                    const EnrollmentIndex& enrollments);
  /**
   * Creates an aligned crop for `face`, runs inference, and returns the
   * normalized embedding.
   */
  const FaceEmbedding& embed(const FrameSampler& sampler,
                             const DetectedFace& face);
  void release() noexcept;
  bool isLoaded() const noexcept;
  std::size_t embeddingLength() const noexcept;
  std::size_t externalMemorySize() const noexcept;

private:
  std::unique_ptr<Ort::Session> _session;
  std::size_t _modelByteSize;
  std::string _inputName;
  std::string _outputName;
  std::vector<float> _input;
  Ort::Value _inputValue{nullptr};
  std::vector<float> _output;
  std::vector<Ort::Value> _outputValues;
  FaceEmbedding _embedding;
};

}  // namespace margelo::nitro::facerecognizer
