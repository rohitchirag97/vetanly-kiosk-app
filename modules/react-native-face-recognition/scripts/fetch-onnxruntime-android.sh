#!/usr/bin/env bash
# Vendors the Android ONNX Runtime native libraries from the official
# onnxruntime-android AAR into third_party/onnxruntime/android/<abi>/.
#
# Headers are committed; only the arm64-v8a libonnxruntime.so (gitignored,
# ~17 MB) is fetched here. Run once after cloning before an Android build.
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.20.0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/onnxruntime/android"
AAR_URL="https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${ORT_VERSION}/onnxruntime-android-${ORT_VERSION}.aar"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading onnxruntime-android ${ORT_VERSION}..."
curl -fSL "$AAR_URL" -o "$TMP/ort.aar"
unzip -q "$TMP/ort.aar" -d "$TMP/ort"

mkdir -p "$DEST/headers"
cp "$TMP"/ort/headers/*.h "$DEST/headers/"

# arm64-v8a only: every Android device this ships to is 64-bit ARM, and the
# other ABIs cost ~50 MB of vendored binaries for builds nobody runs.
for abi in arm64-v8a; do
  mkdir -p "$DEST/$abi"
  cp "$TMP/ort/jni/$abi/libonnxruntime.so" "$DEST/$abi/libonnxruntime.so"
  echo "  $abi/libonnxruntime.so"
done

echo "Done. ONNX Runtime libs in $DEST"
