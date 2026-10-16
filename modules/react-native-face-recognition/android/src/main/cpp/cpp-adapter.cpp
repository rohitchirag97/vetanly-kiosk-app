#include <jni.h>
#include <fbjni/fbjni.h>
#include "FaceRecognizerOnLoad.hpp"

namespace margelo::nitro::facerecognizer {
// Defined in AndroidModelLoader.cpp. Primes fbjni's class cache with the app
// class loader while it is reachable on the JNI_OnLoad thread.
void prepareAndroidModelLoader();
}  // namespace margelo::nitro::facerecognizer

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return facebook::jni::initialize(vm, []() {
    margelo::nitro::facerecognizer::registerAllNatives();
    margelo::nitro::facerecognizer::prepareAndroidModelLoader();
  });
}
