#include "LivenessEnsemble.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "BufferMemory.hpp"
#include "FaceEngineProfiling.hpp"
#include "FaceEngineSession.hpp"
#include "FrameSampler.hpp"
#include "ModelFile.hpp"
#include "ModelLoader.hpp"

namespace margelo::nitro::facerecognizer {

LivenessEnsemble::LivenessEnsemble() noexcept = default;

void LivenessEnsemble::ensureLoaded(const FaceEngineConfig& config,
                                    const Ort::MemoryInfo& memoryInfo) {
  if (!_models.empty()) {
    return;
  }

  FaceProfileScope profile(FaceProfileStage::LoadLiveness);
  const std::array<int64_t, 4> inputShape{
      1, 3, kLivenessInputSize, kLivenessInputSize};
  const std::size_t inputPlaneSize =
      static_cast<std::size_t>(kLivenessInputSize) * kLivenessInputSize;
  std::vector<float> input(inputPlaneSize * 3, 0.0f);
  Ort::Value inputValue = Ort::Value::CreateTensor<float>(memoryInfo,
                                                          input.data(),
                                                          input.size(),
                                                          inputShape.data(),
                                                          inputShape.size());
  std::vector<LivenessModel> models;
  models.reserve(kLivenessModels.size());
  for (const LivenessModelSpec& spec : kLivenessModels) {
    const std::string modelPath = bundledLivenessModelPath(spec.baseName);
    const std::size_t modelByteSize =
        checkedModelFileByteSize(modelPath, spec.baseName);
    std::unique_ptr<Ort::Session> session =
        createSession(modelPath,
                      config.inferenceThreads,
                      config.executionProvider,
                      /*useSharedArena=*/false,
                      std::span<const SessionDimensionOverride>{});

    const std::vector<int64_t> inputShape =
        session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    // Validate bundled model shapes at load time so bad assets fail before a
    // frame-processing path can return an ambiguous spoof result.
    if (inputShape.size() == 4) {
      const int64_t height = inputShape[2];
      const int64_t width = inputShape[3];
      if ((height > 0 && height != kLivenessInputSize) ||
          (width > 0 && width != kLivenessInputSize)) {
        throw std::runtime_error(
            "FaceRecognizer liveness model '" + std::string(spec.baseName) +
            "' expects a " + std::to_string(width) + "x" +
            std::to_string(height) + " input, but anti-spoofing crops are " +
            std::to_string(kLivenessInputSize) + "x" +
            std::to_string(kLivenessInputSize) + ".");
      }
    }

    const std::vector<int64_t> outputShape =
        session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (outputShape.empty() || outputShape.back() <= kLivenessLiveClassIndex) {
      throw std::runtime_error(
          "FaceRecognizer liveness model '" + std::string(spec.baseName) +
          "' must output at least " +
          std::to_string(kLivenessLiveClassIndex + 1) +
          " softmax classes (class " + std::to_string(kLivenessLiveClassIndex) +
          " is 'live').");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    LivenessModel loadedModel;
    loadedModel.inputName = session->GetInputNameAllocated(0, allocator).get();
    loadedModel.outputName =
        session->GetOutputNameAllocated(0, allocator).get();
    loadedModel.modelByteSize = modelByteSize;
    loadedModel.classCount = static_cast<int>(outputShape.back());
    loadedModel.cropScale = spec.cropScale;
    loadedModel.session = std::move(session);

    const std::array<int64_t, 2> outputTensorShape{
        1, static_cast<int64_t>(loadedModel.classCount)};
    loadedModel.output.assign(static_cast<size_t>(loadedModel.classCount),
                              0.0f);
    // Output tensors point into reusable vectors owned by the model wrapper.
    // Keep vector capacity stable while the ORT value is alive.
    loadedModel.outputValues.emplace_back(
        Ort::Value::CreateTensor<float>(memoryInfo,
                                        loadedModel.output.data(),
                                        loadedModel.output.size(),
                                        outputTensorShape.data(),
                                        outputTensorShape.size()));

    models.push_back(std::move(loadedModel));
  }

  _models = std::move(models);
  _input = std::move(input);
  _inputValue = std::move(inputValue);
  profile.setItemCount(static_cast<double>(_models.size()));
}

double LivenessEnsemble::score(const FrameSampler& sampler,
                               const Rect& uprightBounds) {
  FaceProfileScope profile(FaceProfileStage::ScoreLiveness);
  if (_models.empty()) {
    throw std::logic_error(
        "FaceRecognizer liveness models must be loaded before scoring.");
  }

  double summedLiveProbability = 0.0;
  for (LivenessModel& loadedModel : _models) {
    sampler.createLivenessInput(uprightBounds, loadedModel.cropScale, _input);

    const char* inputNames[1] = {loadedModel.inputName.c_str()};
    const char* outputNames[1] = {loadedModel.outputName.c_str()};
    loadedModel.session->Run(Ort::RunOptions{nullptr},
                             inputNames,
                             &_inputValue,
                             1,
                             outputNames,
                             loadedModel.outputValues.data(),
                             loadedModel.outputValues.size());

    const float* logits = loadedModel.output.data();
    const int classCount = loadedModel.classCount;
    double maxLogit = logits[0];
    for (int index = 1; index < classCount; ++index) {
      maxLogit = std::max(maxLogit, static_cast<double>(logits[index]));
    }
    double summedExponentials = 0.0;
    for (int index = 0; index < classCount; ++index) {
      summedExponentials += std::exp(logits[index] - maxLogit);
    }
    // Compute softmax locally because some bundled models expose logits rather
    // than normalized probabilities.
    const double liveProbability =
        std::exp(logits[kLivenessLiveClassIndex] - maxLogit) /
        summedExponentials;
    summedLiveProbability += liveProbability;
  }

  const double result =
      summedLiveProbability / static_cast<double>(_models.size());
  profile.setItemCount(1.0);
  return result;
}

void LivenessEnsemble::release() noexcept {
  for (LivenessModel& model : _models) {
    std::vector<Ort::Value>().swap(model.outputValues);
    model.session.reset();
    model.inputName.clear();
    model.outputName.clear();
    model.modelByteSize = 0;
    clearFloatVector(model.output);
  }
  std::vector<LivenessModel>().swap(_models);
  _inputValue = Ort::Value{nullptr};
  clearFloatVector(_input);
}

bool LivenessEnsemble::isLoaded() const noexcept {
  return !_models.empty();
}

std::size_t LivenessEnsemble::externalMemorySize() const noexcept {
  std::size_t byteSize =
      vectorCapacityByteSize(_input) + vectorCapacityByteSize(_models);
  for (const LivenessModel& model : _models) {
    byteSize += modelResidencyEstimate(model.modelByteSize);
    byteSize += vectorCapacityByteSize(model.output);
    byteSize += vectorCapacityByteSize(model.outputValues);
  }
  return byteSize;
}

}  // namespace margelo::nitro::facerecognizer
