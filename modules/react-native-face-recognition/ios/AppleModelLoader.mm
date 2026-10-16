#import <Foundation/Foundation.h>

#include "ModelLoader.hpp"

#include <stdexcept>
#include <string>

namespace margelo::nitro::facerecognizer {
namespace {

std::string bundledModelPath(NSString* name) {
  NSString* path = [[NSBundle mainBundle] pathForResource:name ofType:@"onnx"];
  if (path == nil) {
    throw std::runtime_error("FaceRecognizer could not find bundled model '" +
                             std::string(name.UTF8String) +
                             ".onnx' in the app bundle.");
  }
  return std::string(path.UTF8String);
}

}  // namespace

std::string bundledDetectorModelPath() {
  return bundledModelPath(@"yunet");
}

std::string bundledRecognizerModelPath() {
#if defined(FACE_RECOGNIZER_BENCH_RECOGNIZER_MODEL_PATH)
  return FACE_RECOGNIZER_BENCH_RECOGNIZER_MODEL_PATH;
#else
  return bundledModelPath(@"sface");
#endif
}

std::string bundledLivenessModelPath(const std::string& baseName) {
  return bundledModelPath([NSString stringWithUTF8String:baseName.c_str()]);
}

}  // namespace margelo::nitro::facerecognizer
