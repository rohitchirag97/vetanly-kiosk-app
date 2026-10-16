#include "RecognizerSession.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "BufferMemory.hpp"
#include "FaceConstants.hpp"
#include "FaceEngineProfiling.hpp"
#include "FaceEngineSession.hpp"
#include "FrameSampler.hpp"
#include "ModelFile.hpp"
#include "ModelLoader.hpp"

namespace margelo::nitro::facerecognizer {

RecognizerSession::RecognizerSession() noexcept : _modelByteSize(0) {}

void RecognizerSession::ensureLoaded(const FaceEngineConfig& config,
                                     const Ort::MemoryInfo& memoryInfo,
                                     const EnrollmentIndex& enrollments) {
  if (_session != nullptr) {
    return;
  }

  FaceProfileScope profile(FaceProfileStage::LoadRecognizer);
  const std::string recognizerPath = bundledRecognizerModelPath();
  const std::size_t modelByteSize =
      checkedModelFileByteSize(recognizerPath, "recognizer");
  std::unique_ptr<Ort::Session> session =
      createSession(recognizerPath,
                    config.inferenceThreads,
                    config.executionProvider,
                    /*useSharedArena=*/false,
                    std::span<const SessionDimensionOverride>{});

  const std::vector<int64_t> inputShape =
      session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if (inputShape.size() == 4) {
    const int64_t height = inputShape[2];
    const int64_t width = inputShape[3];
    if ((height > 0 && height != kRecognizerInputSize) ||
        (width > 0 && width != kRecognizerInputSize)) {
      throw std::runtime_error(
          "FaceRecognizer recognizer model expects a " + std::to_string(width) +
          "x" + std::to_string(height) + " input, but faces are aligned to " +
          std::to_string(kRecognizerInputSize) + "x" +
          std::to_string(kRecognizerInputSize) + ". Provide a model with a " +
          std::to_string(kRecognizerInputSize) + "x" +
          std::to_string(kRecognizerInputSize) + " input.");
    }
  }

  const std::vector<int64_t> outputShape =
      session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if (outputShape.empty() || outputShape.back() <= 0) {
    throw std::runtime_error(
        "FaceRecognizer recognizer model must declare a fixed embedding length "
        "in its output shape.");
  }
  const int64_t embeddingLength = outputShape.back();

  for (const EnrollmentIndex::value_type& enrollment : enrollments) {
    for (const FaceEmbedding& sample : enrollment.second.samples) {
      if (static_cast<int64_t>(sample.size()) != embeddingLength) {
        throw std::runtime_error(
            "FaceRecognizer enrollment '" + enrollment.first +
            "' has a template with " + std::to_string(sample.size()) +
            " values but the recognizer model produces " +
            std::to_string(embeddingLength) +
            "-dimensional embeddings. Re-enroll with this model or clear the "
            "stored templates.");
      }
    }
  }

  Ort::AllocatorWithDefaultOptions allocator;
  _inputName = session->GetInputNameAllocated(0, allocator).get();
  _outputName = session->GetOutputNameAllocated(0, allocator).get();

  const std::array<int64_t, 4> recognizerInputShape{
      1, 3, kRecognizerInputSize, kRecognizerInputSize};
  const std::size_t inputPlaneSize =
      static_cast<std::size_t>(kRecognizerInputSize) * kRecognizerInputSize;
  _input.assign(inputPlaneSize * 3, 0.0f);
  _inputValue = Ort::Value::CreateTensor<float>(memoryInfo,
                                                _input.data(),
                                                _input.size(),
                                                recognizerInputShape.data(),
                                                recognizerInputShape.size());

  const std::array<int64_t, 2> recognizerOutputShape{1, embeddingLength};
  _output.assign(static_cast<size_t>(embeddingLength), 0.0f);
  _embedding.assign(static_cast<size_t>(embeddingLength), 0.0f);
  _outputValues.clear();
  _outputValues.emplace_back(
      Ort::Value::CreateTensor<float>(memoryInfo,
                                      _output.data(),
                                      _output.size(),
                                      recognizerOutputShape.data(),
                                      recognizerOutputShape.size()));

  _modelByteSize = modelByteSize;
  _session = std::move(session);
}

const FaceEmbedding& RecognizerSession::embed(const FrameSampler& sampler,
                                              const DetectedFace& face) {
  FaceProfileScope profile(FaceProfileStage::EmbedFace);
  if (_session == nullptr) {
    throw std::logic_error(
        "FaceRecognizer recognizer network must be loaded before embedding.");
  }

  {
    FaceProfileScope alignmentProfile(FaceProfileStage::AlignFace);
    sampler.createRecognizerInput(face, _input);
    alignmentProfile.setItemCount(1.0);
  }
#if defined(FACE_RECOGNIZER_BENCH_CENTERED_RECOGNIZER_INPUT)
  for (float& value : _input) {
    value = (value - 127.5f) / 127.5f;
  }
#endif

  const char* inputNames[1] = {_inputName.c_str()};
  const char* outputNames[1] = {_outputName.c_str()};
  _session->Run(Ort::RunOptions{nullptr},
                inputNames,
                &_inputValue,
                1,
                outputNames,
                _outputValues.data(),
                1);

  normalizeEmbeddingInto(_output.data(), _output.size(), _embedding);
  profile.setItemCount(1.0);
  return _embedding;
}

void RecognizerSession::release() noexcept {
  std::vector<Ort::Value>().swap(_outputValues);
  _inputValue = Ort::Value{nullptr};
  _session.reset();
  _inputName.clear();
  _outputName.clear();
  clearFloatVector(_input);
  clearFloatVector(_output);
  clearFloatVector(_embedding);
  _modelByteSize = 0;
}

bool RecognizerSession::isLoaded() const noexcept {
  return _session != nullptr;
}

std::size_t RecognizerSession::embeddingLength() const noexcept {
  return _output.size();
}

std::size_t RecognizerSession::externalMemorySize() const noexcept {
  return modelResidencyEstimate(_modelByteSize) +
         vectorCapacityByteSize(_input) + vectorCapacityByteSize(_output) +
         vectorCapacityByteSize(_outputValues) +
         vectorCapacityByteSize(_embedding);
}

}  // namespace margelo::nitro::facerecognizer
