require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "FaceRecognizer"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]

  s.platforms    = { :ios => min_ios_version_supported }
  s.source       = { :git => "https://github.com/patrickkabwe/react-native-vision-camera-face-recognizer.git", :tag => "#{s.version}" }

  s.source_files = [
    # Implementation (Swift)
    "ios/**/*.{swift}",
    # Autolinking/Registration (Objective-C++)
    "ios/**/*.{m,mm}",
    # Implementation (C++ objects)
    "cpp/**/*.{hpp,cpp}",
  ]
  s.resources = "models/*.onnx"

  load 'nitrogen/generated/ios/FaceRecognizer+autolinking.rb'
  add_nitrogen_files(s)
  s.pod_target_xcconfig = (s.attributes_hash['pod_target_xcconfig'] || {}).merge({
    "HEADER_SEARCH_PATHS" => [
      "$(PODS_TARGET_SRCROOT)/cpp",
      "$(PODS_TARGET_SRCROOT)/cpp/bridge",
      "$(PODS_TARGET_SRCROOT)/cpp/detection",
      "$(PODS_TARGET_SRCROOT)/cpp/engine",
      "$(PODS_TARGET_SRCROOT)/cpp/enrollment",
      "$(PODS_TARGET_SRCROOT)/cpp/frame",
      "$(PODS_TARGET_SRCROOT)/cpp/liveness",
      "$(PODS_TARGET_SRCROOT)/cpp/recognition",
      "$(PODS_TARGET_SRCROOT)/cpp/runtime",
      "$(PODS_TARGET_SRCROOT)/cpp/tracking",
    ].join(" "),
  })

  s.dependency 'React-jsi'
  s.dependency 'React-callinvoker'
  s.dependency 'VisionCamera'
  s.dependency 'onnxruntime-c', '~> 1.20.0'
  install_modules_dependencies(s)
end
