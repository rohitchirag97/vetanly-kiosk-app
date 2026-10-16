#include "DetectorSession.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "BufferMemory.hpp"
#include "FaceEngineProfiling.hpp"
#include "FaceEngineSession.hpp"
#include "FaceGeometry.hpp"
#include "FrameSampler.hpp"
#include "ModelFile.hpp"
#include "ModelLoader.hpp"
#include "YuNetDecoder.hpp"

namespace margelo::nitro::facerecognizer {

DetectorSession::DetectorSession(const FaceEngineConfig& config,
                                 const Ort::MemoryInfo& memoryInfo)
    : _modelByteSize(0), _inputSize(config.detectorInputSize) {
  FaceProfileScope profile(FaceProfileStage::LoadDetector);
  const std::string detectorPath = bundledDetectorModelPath();
  _modelByteSize = checkedModelFileByteSize(detectorPath, "detector");
  const std::array<SessionDimensionOverride, 2> dimensionOverrides{
      SessionDimensionOverride{"height", _inputSize},
      SessionDimensionOverride{"width", _inputSize},
  };
  _session = createSession(detectorPath,
                           config.detectorThreads,
                           config.executionProvider,
                           /*useSharedArena=*/true,
                           dimensionOverrides);
  configure(memoryInfo);
}

void DetectorSession::configure(const Ort::MemoryInfo& memoryInfo) {
  Ort::AllocatorWithDefaultOptions allocator;
  _inputName = _session->GetInputNameAllocated(0, allocator).get();
  validateInputSize();
  bindInput(memoryInfo);
  bindOutputs(memoryInfo);
}

void DetectorSession::validateInputSize() const {
  const std::vector<int64_t> inputShape =
      _session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if (inputShape.size() != 4) {
    throw std::runtime_error(
        "FaceRecognizer detector model input must be a four-dimensional "
        "NCHW tensor.");
  }
  const int64_t modelHeight = inputShape[2];
  const int64_t modelWidth = inputShape[3];
  if ((modelHeight > 0 && modelHeight != _inputSize) ||
      (modelWidth > 0 && modelWidth != _inputSize)) {
    throw std::runtime_error(
        "FaceRecognizer configured detection.inputSize=" +
        std::to_string(_inputSize) +
        ", but the YuNet model has a fixed input shape of " +
        std::to_string(modelWidth) + "x" + std::to_string(modelHeight) +
        ". Provide a dynamic-shape model or configure the matching size.");
  }
}

void DetectorSession::bindInput(const Ort::MemoryInfo& memoryInfo) {
  const std::size_t planeStride = static_cast<std::size_t>(_inputSize) *
                                  static_cast<std::size_t>(_inputSize);
  _input.assign(planeStride * 3, 0.0f);
  const std::array<int64_t, 4> inputShape{1, 3, _inputSize, _inputSize};
  _inputValue = Ort::Value::CreateTensor<float>(memoryInfo,
                                                _input.data(),
                                                _input.size(),
                                                inputShape.data(),
                                                inputShape.size());
}

void DetectorSession::bindOutputs(const Ort::MemoryInfo& memoryInfo) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> availableOutputNames;
  availableOutputNames.reserve(_session->GetOutputCount());
  for (size_t index = 0; index < _session->GetOutputCount(); ++index) {
    availableOutputNames.emplace_back(
        _session->GetOutputNameAllocated(index, allocator).get());
  }

  _outputValues.clear();
  _outputValues.reserve(kYuNetOutputCount);
  size_t outputSlot = 0;
  for (size_t outputGroup = 0; outputGroup < kYuNetPrefixes.size();
       ++outputGroup) {
    for (int stride : kYuNetStrides) {
      std::string outputName = std::string(kYuNetPrefixes[outputGroup]) + "_" +
                               std::to_string(stride);
      if (std::find(availableOutputNames.begin(),
                    availableOutputNames.end(),
                    outputName) == availableOutputNames.end()) {
        throw std::runtime_error(
            "FaceRecognizer YuNet model is missing the expected output '" +
            outputName + "'. Provide an OpenCV Zoo YuNet ONNX model.");
      }
      _outputNames[outputSlot] = std::move(outputName);

      const int grid = _inputSize / stride;
      const int64_t locations = static_cast<int64_t>(grid) * grid;
      const int64_t channels = kYuNetChannels[outputGroup];
      std::vector<float>& buffer = _outputBuffers[outputSlot];
      buffer.assign(static_cast<size_t>(locations * channels), 0.0f);
      const std::array<int64_t, 3> shape{1, locations, channels};
      _outputValues.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo,
                                                                 buffer.data(),
                                                                 buffer.size(),
                                                                 shape.data(),
                                                                 shape.size()));
      ++outputSlot;
    }
  }
}

std::vector<DetectedFace> DetectorSession::detect(
    const FrameSampler& sampler, const FaceEngineConfig& config, int maxFaces) {
  FaceProfileScope profile(FaceProfileStage::DetectFaces);
  if (_session == nullptr) {
    throw std::logic_error(
        "FaceRecognizer detector network must be loaded before detection.");
  }

  LetterboxTransform letterbox{};
  sampler.createDetectorInput(_inputSize, letterbox, _input, _inputCache);

  const char* inputNames[1] = {_inputName.c_str()};
  std::array<const char*, kYuNetOutputCount> outputNames{};
  for (size_t index = 0; index < outputNames.size(); ++index) {
    outputNames[index] = _outputNames[index].c_str();
  }
  _session->Run(Ort::RunOptions{nullptr},
                inputNames,
                &_inputValue,
                1,
                outputNames.data(),
                _outputValues.data(),
                _outputValues.size());

  std::vector<DetectedFace> detected;
  std::array<const float*, kYuNetOutputCount> outputs{};
  for (size_t index = 0; index < outputs.size(); ++index) {
    outputs[index] = _outputBuffers[index].data();
  }
  const YuNetDecodeParams params{_inputSize,
                                 config.detectionThreshold,
                                 config.detectionMinFaceSize,
                                 letterbox.scale};
  std::vector<DetectionCandidate> selected = suppressOverlaps(
      decodeYuNetOutputs(outputs, params), kNmsIouThreshold, maxFaces);

  detected.reserve(selected.size());
  for (const DetectionCandidate& candidate : selected) {
    std::array<Point, kFaceLandmarkCount> landmarks{};
    for (std::size_t index = 0; index < landmarks.size(); ++index) {
      landmarks[index] =
          sampler.detectorToFrame(candidate.landmarks[index], letterbox);
    }
    detected.push_back(DetectedFace{
        mapCandidateBounds(candidate, sampler, letterbox),
        candidate.confidence,
        landmarks,
    });
  }
  profile.setItemCount(static_cast<double>(detected.size()));
  return detected;
}

void DetectorSession::release() noexcept {
  std::vector<Ort::Value>().swap(_outputValues);
  _inputValue = Ort::Value{nullptr};
  _session.reset();
  _inputName.clear();
  _outputNames = {};
  clearFloatVector(_input);
  _inputCache.releaseMemory();
  for (std::vector<float>& buffer : _outputBuffers) {
    clearFloatVector(buffer);
  }
  _modelByteSize = 0;
}

std::size_t DetectorSession::externalMemorySize() const noexcept {
  return modelResidencyEstimate(_modelByteSize) +
         vectorCapacityByteSize(_input) +
         vectorCapacityByteSize(_inputCache.pixelMappings) +
         vectorArrayCapacityByteSize(_outputBuffers) +
         vectorCapacityByteSize(_outputValues);
}

}  // namespace margelo::nitro::facerecognizer
