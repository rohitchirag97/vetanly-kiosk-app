#include "FaceEngineSession.hpp"

#include "onnxruntime_session_options_config_keys.h"

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

#if defined(__APPLE__)
#include "coreml_provider_factory.h"
#elif defined(__ANDROID__)
#include <android/log.h>
#include <nnapi_provider_factory.h>
#endif

namespace margelo::nitro::facerecognizer {
namespace {

Ort::Env& globalOrtEnv() {
  static Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "FaceRecognizer");
  return env;
}

void ensureSharedCpuArena() {
  static const bool registered = [] {
    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::ArenaCfg arenaCfg(/*max_mem*/ 0,
                           /*arena_extend_strategy*/ 1,
                           /*initial_chunk_size_bytes*/ -1,
                           /*max_dead_bytes_per_chunk*/ -1);
    globalOrtEnv().CreateAndRegisterAllocator(memoryInfo, arenaCfg);
    return true;
  }();
  (void)registered;
}

void appendXnnpackProvider(Ort::SessionOptions& options, int inferenceThreads) {
  options.AppendExecutionProvider(
      "XNNPACK", {{"intra_op_num_threads", std::to_string(inferenceThreads)}});
  options.SetIntraOpNumThreads(1);
}

#if defined(__ANDROID__)
void logAndroidProviderStatus(const char* provider,
                              const char* status,
                              const std::string& detail) noexcept {
  __android_log_print(ANDROID_LOG_INFO,
                      "FaceRecognizer",
                      "event=provider_status provider=%s status=%s detail=%s",
                      provider,
                      status,
                      detail.c_str());
}

bool appendNnapiProvider(Ort::SessionOptions& options) {
  try {
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(
        options, NNAPI_FLAG_CPU_DISABLED));
    logAndroidProviderStatus("nnapi", "enabled", "cpu_disabled");
    return true;
  } catch (const Ort::Exception& error) {
    logAndroidProviderStatus("nnapi", "unavailable", error.what());
    return false;
  }
}

bool appendQnnGpuProvider(Ort::SessionOptions& options) {
  try {
    options.AppendExecutionProvider(
        "QNN",
        {{"backend_type", "gpu"}, {"offload_graph_io_quantization", "0"}});
    logAndroidProviderStatus("qnn_gpu", "enabled", "backend_type=gpu");
    return true;
  } catch (const Ort::Exception& error) {
    logAndroidProviderStatus("qnn_gpu", "unavailable", error.what());
    return false;
  }
}
#endif

Ort::SessionOptions buildSessionOptions(int inferenceThreads,
                                        FaceExecutionProvider provider,
                                        bool useSharedArena) {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(inferenceThreads);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  if (useSharedArena) {
    // Persistent sessions (the detector) draw weights/activations from a
    // process-wide arena so repeated runs reuse the same memory.
    ensureSharedCpuArena();
    options.AddConfigEntry(kOrtSessionOptionsConfigUseEnvAllocators, "1");
  } else {
    // Recognition/liveness sessions are on-demand and much larger than the
    // detector. Avoid private arenas so releasing the session returns native
    // heap instead of keeping it resident for future calls.
    options.DisableCpuMemArena();
  }
  // Static frame shapes make ORT memory patterns unnecessary here; the detector
  // reuses its shared arena and on-demand sessions avoid arenas entirely.
  options.DisableMemPattern();

  if (provider == FaceExecutionProvider::CPU) {
    appendXnnpackProvider(options, inferenceThreads);
#if defined(__ANDROID__)
    logAndroidProviderStatus("xnnpack", "enabled", "requested_cpu");
#endif
    return options;
  }

#if defined(__ANDROID__)
  if (provider == FaceExecutionProvider::AUTO) {
    appendNnapiProvider(options);
    appendXnnpackProvider(options, inferenceThreads);
    logAndroidProviderStatus("xnnpack", "enabled", "auto_fallback");
    return options;
  }

  if (provider == FaceExecutionProvider::GPU) {
    if (!appendQnnGpuProvider(options)) {
      appendNnapiProvider(options);
    }
    appendXnnpackProvider(options, inferenceThreads);
    logAndroidProviderStatus("xnnpack", "enabled", "gpu_fallback");
    return options;
  }

  options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(
      options, NNAPI_FLAG_CPU_DISABLED));
  logAndroidProviderStatus("nnapi", "enabled", "strict_cpu_disabled");
#elif defined(__APPLE__)
  if (provider == FaceExecutionProvider::AUTO) {
    appendXnnpackProvider(options, inferenceThreads);
    return options;
  }

  options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
  uint32_t coremlFlags = COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES;
  if (provider == FaceExecutionProvider::GPU) {
    coremlFlags |= COREML_FLAG_USE_CPU_AND_GPU;
  }
  Ort::ThrowOnError(
      OrtSessionOptionsAppendExecutionProvider_CoreML(options, coremlFlags));
#else
  if (provider == FaceExecutionProvider::AUTO) {
    appendXnnpackProvider(options, inferenceThreads);
    return options;
  }

  throw std::invalid_argument(
      "FaceRecognizer accelerated execution providers are unsupported on this "
      "platform.");
#endif
  return options;
}

void applyDimensionOverrides(
    Ort::SessionOptions& options,
    std::span<const SessionDimensionOverride> dimensionOverrides) {
  for (const SessionDimensionOverride& dimensionOverride : dimensionOverrides) {
    Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(
        options, dimensionOverride.name.c_str(), dimensionOverride.value));
  }
}

std::unique_ptr<Ort::Session> createSessionWithProvider(
    const std::string& modelPath,
    int inferenceThreads,
    FaceExecutionProvider provider,
    bool useSharedArena,
    std::span<const SessionDimensionOverride> dimensionOverrides) {
  Ort::SessionOptions options =
      buildSessionOptions(inferenceThreads, provider, useSharedArena);
  applyDimensionOverrides(options, dimensionOverrides);
  return std::make_unique<Ort::Session>(
      globalOrtEnv(), modelPath.c_str(), options);
}

}  // namespace

std::unique_ptr<Ort::Session> createSession(
    const std::string& modelPath,
    int inferenceThreads,
    FaceExecutionProvider provider,
    bool useSharedArena,
    std::span<const SessionDimensionOverride> dimensionOverrides) {
  // Every step of the provider-fallback ladder builds the same session and
  // varies only the provider; keep the shared arguments in one place.
  const auto makeSession = [&](FaceExecutionProvider sessionProvider) {
    return createSessionWithProvider(modelPath,
                                     inferenceThreads,
                                     sessionProvider,
                                     useSharedArena,
                                     dimensionOverrides);
  };

#if defined(__ANDROID__)
  if (provider == FaceExecutionProvider::AUTO) {
    try {
      return makeSession(provider);
    } catch (const Ort::Exception& autoError) {
      try {
        return makeSession(FaceExecutionProvider::CPU);
      } catch (const Ort::Exception& fallbackError) {
        throw std::runtime_error(
            "Failed to create ONNX Runtime session for model '" + modelPath +
            "' with automatic provider fallback. Auto provider error: " +
            std::string(autoError.what()) +
            ". CPU fallback error: " + std::string(fallbackError.what()));
      }
    }
  }

  if (provider == FaceExecutionProvider::GPU) {
    try {
      return makeSession(provider);
    } catch (const Ort::Exception& gpuError) {
      try {
        return createSession(modelPath,
                             inferenceThreads,
                             FaceExecutionProvider::AUTO,
                             useSharedArena,
                             dimensionOverrides);
      } catch (const Ort::Exception& fallbackError) {
        throw std::runtime_error(
            "Failed to create ONNX Runtime session for model '" + modelPath +
            "' with GPU provider and automatic fallback. GPU error: " +
            std::string(gpuError.what()) +
            ". Fallback error: " + std::string(fallbackError.what()));
      }
    }
  }
#endif

  return makeSession(provider);
}

}  // namespace margelo::nitro::facerecognizer
