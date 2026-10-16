#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "DetectedFace.hpp"
#include "DetectorInputCache.hpp"
#include "FaceConstants.hpp"
#include "FaceEngineConfig.hpp"
#include "onnxruntime_cxx_api.h"

namespace margelo::nitro::facerecognizer {

class FrameSampler;

/**
 * YuNet detector ONNX session with reusable input/output buffers.
 *
 * The detector remains loaded for the engine lifetime because it is needed for
 * both recognition and enrollment. Input pixel mappings are cached when the
 * frame layout is stable.
 */
class DetectorSession final {
public:
  DetectorSession(const FaceEngineConfig& config,
                  const Ort::MemoryInfo& memoryInfo);

  DetectorSession(const DetectorSession&) = delete;
  DetectorSession& operator=(const DetectorSession&) = delete;
  DetectorSession(DetectorSession&&) = delete;
  DetectorSession& operator=(DetectorSession&&) = delete;

  std::vector<DetectedFace> detect(const FrameSampler& sampler,
                                   const FaceEngineConfig& config,
                                   int maxFaces);
  void release() noexcept;
  std::size_t externalMemorySize() const noexcept;

private:
  void configure(const Ort::MemoryInfo& memoryInfo);
  void validateInputSize() const;
  void bindInput(const Ort::MemoryInfo& memoryInfo);
  void bindOutputs(const Ort::MemoryInfo& memoryInfo);

  std::unique_ptr<Ort::Session> _session;
  std::size_t _modelByteSize;
  int _inputSize;
  std::string _inputName;
  std::array<std::string, kYuNetOutputCount> _outputNames;
  std::vector<float> _input;
  Ort::Value _inputValue{nullptr};
  DetectorInputCache _inputCache;
  std::array<std::vector<float>, kYuNetOutputCount> _outputBuffers;
  std::vector<Ort::Value> _outputValues;
};

}  // namespace margelo::nitro::facerecognizer
