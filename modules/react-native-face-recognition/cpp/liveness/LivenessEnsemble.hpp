#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "FaceConstants.hpp"
#include "FaceEngineConfig.hpp"
#include "Rect.hpp"
#include "onnxruntime_cxx_api.h"

namespace margelo::nitro::facerecognizer {

class FrameSampler;

/**
 * Lazy-loaded passive liveness ensemble.
 *
 * Models are loaded only when liveness is enabled and a score is required.
 * The instance reuses input/output buffers across calls and must be accessed
 * under `FaceEngine`'s inference mutex.
 */
class LivenessEnsemble final {
public:
  LivenessEnsemble() noexcept;

  LivenessEnsemble(const LivenessEnsemble&) = delete;
  LivenessEnsemble& operator=(const LivenessEnsemble&) = delete;
  LivenessEnsemble(LivenessEnsemble&&) = delete;
  LivenessEnsemble& operator=(LivenessEnsemble&&) = delete;

  void ensureLoaded(const FaceEngineConfig& config,
                    const Ort::MemoryInfo& memoryInfo);
  /**
   * Returns averaged live probability for an upright face box.
   *
   * `ensureLoaded` must have completed successfully before this is called.
   */
  double score(const FrameSampler& sampler, const Rect& uprightBounds);
  void release() noexcept;
  bool isLoaded() const noexcept;
  std::size_t externalMemorySize() const noexcept;

private:
  struct LivenessModel final {
    std::unique_ptr<Ort::Session> session;
    std::string inputName;
    std::string outputName;
    std::size_t modelByteSize;
    int classCount;
    float cropScale;
    std::vector<float> output;
    std::vector<Ort::Value> outputValues;
  };

  std::vector<LivenessModel> _models;
  std::vector<float> _input;
  Ort::Value _inputValue{nullptr};
};

}  // namespace margelo::nitro::facerecognizer
