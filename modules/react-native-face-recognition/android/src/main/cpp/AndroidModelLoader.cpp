#include "ModelLoader.hpp"

#include <fbjni/fbjni.h>
#include <jni.h>
#include <stdexcept>
#include <string>

namespace margelo::nitro::facerecognizer {
namespace {

facebook::jni::global_ref<jclass> gProviderClass;
jmethodID gMaterializeModel = nullptr;

std::string materializeBundledModel(const std::string& assetName) {
  if (!gProviderClass) {
    throw std::runtime_error(
        "FaceRecognizer JNI was not initialized; ModelAssetProvider is "
        "unavailable.");
  }

  facebook::jni::ThreadScope threadScope;
  JNIEnv* environment = facebook::jni::Environment::current();

  jstring javaAssetName = environment->NewStringUTF(assetName.c_str());
  jobject pathObject = environment->CallStaticObjectMethod(
      gProviderClass.get(), gMaterializeModel, javaAssetName);
  environment->DeleteLocalRef(javaAssetName);

  if (environment->ExceptionCheck()) {
    environment->ExceptionDescribe();
    environment->ExceptionClear();
    throw std::runtime_error(
        "FaceRecognizer failed to materialize bundled model '" + assetName +
        "'.");
  }
  if (pathObject == nullptr) {
    throw std::runtime_error(
        "FaceRecognizer received no cached path for bundled model '" +
        assetName + "'.");
  }

  jstring pathString = static_cast<jstring>(pathObject);
  const char* pathChars = environment->GetStringUTFChars(pathString, nullptr);
  std::string path(pathChars != nullptr ? pathChars : "");
  if (pathChars != nullptr) {
    environment->ReleaseStringUTFChars(pathString, pathChars);
  }
  environment->DeleteLocalRef(pathObject);
  return path;
}

}  // namespace

void prepareAndroidModelLoader() {
  // Runs during JNI_OnLoad, on the thread that called System.loadLibrary, where
  // FindClass resolves against the application class loader. We must capture
  // the class and method handles here and keep the class as a global reference,
  // because later lookups happen on Nitro's JVM-detached background threads,
  // which see only the system class loader and would throw
  // ClassNotFoundException for application classes.
  JNIEnv* environment = facebook::jni::Environment::current();
  auto providerClass =
      facebook::jni::findClassStatic("com/facerecognizer/ModelAssetProvider");
  gMaterializeModel =
      environment->GetStaticMethodID(providerClass.get(),
                                     "materializeBundledModel",
                                     "(Ljava/lang/String;)Ljava/lang/String;");
  if (gMaterializeModel == nullptr) {
    throw std::runtime_error("FaceRecognizer could not resolve "
                             "ModelAssetProvider.materializeBundledModel().");
  }
  gProviderClass = facebook::jni::make_global(providerClass);
}

std::string bundledDetectorModelPath() {
  return materializeBundledModel("yunet.onnx");
}

std::string bundledRecognizerModelPath() {
  return materializeBundledModel("sface.onnx");
}

std::string bundledLivenessModelPath(const std::string& baseName) {
  return materializeBundledModel(baseName + ".onnx");
}

}  // namespace margelo::nitro::facerecognizer
