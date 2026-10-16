#!/usr/bin/env bash
set -euo pipefail

source_files=()
# FaceEngineSession.cpp includes the iOS CoreML ONNX Runtime provider header,
# which is not present in this checkout. It is still covered by clang-format.
while IFS= read -r -d '' file_path; do
  source_files+=("$file_path")
done < <(
  find cpp \
    -type f \
    \( \
      -name '*.cc' \
      -o -name '*.cpp' \
      -o -name '*.cxx' \
    \) \
    ! -path 'cpp/runtime/FaceEngineSession.cpp' \
    -print0
)

if [[ "${#source_files[@]}" -eq 0 ]]; then
  echo "No C++ source files found to lint." >&2
  exit 1
fi

clang-tidy \
  --quiet \
  --config-file=.clang-tidy \
  "${source_files[@]}" \
  -- \
  -std=c++20 \
  -Icpp \
  -Icpp/bridge \
  -Icpp/detection \
  -Icpp/engine \
  -Icpp/enrollment \
  -Icpp/frame \
  -Icpp/liveness \
  -Icpp/recognition \
  -Icpp/runtime \
  -Icpp/tracking \
  -isystemthird_party/onnxruntime/android/headers \
  -Initrogen/generated/shared/c++ \
  -isystem../../node_modules/react-native-nitro-modules/android/build/headers/nitromodules \
  -isystem../../node_modules/react-native-vision-camera/android/build/headers/visioncamera \
  -isystem../../node_modules/react-native-nitro-image/android/build/headers/nitroimage \
  -isystem../../node_modules/react-native/ReactCommon \
  -isystem../../node_modules/react-native/ReactCommon/jsi
