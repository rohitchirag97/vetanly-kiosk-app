#include "FaceEngine.hpp"

#include "BufferMemory.hpp"

#include <stdexcept>
#if defined(FACE_RECOGNIZER_ENABLE_BENCH) && FACE_RECOGNIZER_ENABLE_BENCH
#include "GalleryBench.hpp"
#endif
#if defined(__ANDROID__)
#include <dlfcn.h>
#include <malloc.h>
#endif

namespace margelo::nitro::facerecognizer {

FaceEngine::FaceEngine(const FaceRecognizerOptions& options)
    : _config(validateFaceRecognizerOptions(options)),
      _memoryInfo(
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      _detector(_config, _memoryInfo),
      _framesSinceDetection(_config.detectionInterval),
      _framesSinceRecognition(_config.recognitionInterval),
      _isDisposed(false) {
#if defined(FACE_RECOGNIZER_ENABLE_BENCH) && FACE_RECOGNIZER_ENABLE_BENCH
  maybeStartGalleryBenchmark();
#endif
}

FaceEngine::~FaceEngine() {
  dispose();
}

void FaceEngine::ensureActiveLocked() const {
  if (_isDisposed) {
    throw std::runtime_error(
        "FaceRecognizer native engine has been disposed and cannot be reused.");
  }
}

void FaceEngine::dispose() noexcept {
  std::lock_guard<std::mutex> lock(_inferenceMutex);
  releaseResourcesLocked();
}

void FaceEngine::releaseResourcesLocked() noexcept {
  if (_isDisposed) {
    return;
  }
  _liveness.release();
  _recognizer.release();
  _detector.release();
  _tracker.reset();
  _recentMatches.clear();
  _pendingRecognitions.clear();
  _enrollments.clear();
  _isDisposed = true;
}

void FaceEngine::trimNativeHeapLocked() const noexcept {
  // minSdk predates the NDK's mallopt() declaration (API 26), so the symbol
  // must be resolved at runtime; it is cached after the first lookup. Bionic
  // ignores unknown mallopt options, so M_PURGE_ALL is safe to pass on OS
  // versions that predate it.
#if defined(__ANDROID__) && defined(M_PURGE_ALL)
  using MalloptFunction = int (*)(int, int);
  static const MalloptFunction malloptFunction =
      reinterpret_cast<MalloptFunction>(dlsym(RTLD_DEFAULT, "mallopt"));
  if (malloptFunction != nullptr) {
    (void)malloptFunction(M_PURGE_ALL, 0);
  }
#endif
}

void FaceEngine::releaseIdleModelsLocked() noexcept {
  if (!_enrollments.empty()) {
    return;
  }
  // Only trim on a loaded -> released transition; purging the native heap
  // every no-enrollment frame would fight the allocator and fault pages back
  // in on the next frame.
  if (!_recognizer.isLoaded() && !_liveness.isLoaded()) {
    return;
  }
  _recognizer.release();
  _liveness.release();
  trimNativeHeapLocked();
}

std::size_t FaceEngine::externalMemorySize() const noexcept {
  const std::unique_lock<std::mutex> lock(_inferenceMutex, std::try_to_lock);
  if (lock.owns_lock()) {
    std::size_t byteSize = 0;
    byteSize += _detector.externalMemorySize();
    byteSize += _recognizer.externalMemorySize();
    byteSize += _liveness.externalMemorySize();
    byteSize += _enrollments.externalMemorySize();
    byteSize += vectorCapacityByteSize(_pendingRecognitions);
    for (const PendingRecognition& pending : _pendingRecognitions) {
      byteSize += vectorCapacityByteSize(pending.samples);
      for (const FaceEmbedding& sample : pending.samples) {
        byteSize += vectorCapacityByteSize(sample);
      }
    }
    _lastKnownExternalMemorySize.store(byteSize, std::memory_order_relaxed);
  }
  return _lastKnownExternalMemorySize.load(std::memory_order_relaxed);
}

}  // namespace margelo::nitro::facerecognizer
