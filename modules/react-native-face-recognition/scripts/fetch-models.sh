#!/usr/bin/env bash
# Fetches the large bundled models that are not committed.
#
#   SFace (face recognition)        -> models/sface.onnx          (~37 MB)
#   MiniFASNet V2  (anti-spoofing)  -> models/minifasnet_v2.onnx  (~1.7 MB, crop scale 2.7)
#   MiniFASNet V1SE (anti-spoofing) -> models/minifasnet_v1se.onnx(~1.7 MB, crop scale 4.0)
#
# The YuNet detector (models/yunet.onnx) is committed: it is a small (~230 KB)
# dynamic-input variant of the OpenCV Zoo model. Run this once after cloning if
# the large models are missing.
#
# The anti-spoofing models are the MiniFASNet / Silent-Face ensemble (Apache-2.0,
# https://github.com/yakhyo/face-anti-spoofing). They are optional: face
# detection/recognition work without them, and they are only loaded when
# `requireLiveness` is set or `checkLiveness()` is called. Override the source
# URLs via the env vars below if you mirror the weights.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/models"
mkdir -p "$DEST"

SFACE_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
MINIFASNET_V2_URL="${MINIFASNET_V2_URL:-https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx}"
MINIFASNET_V1SE_URL="${MINIFASNET_V1SE_URL:-https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV1SE.onnx}"

echo "Downloading SFace -> $DEST/sface.onnx"
curl -fSL "$SFACE_URL" -o "$DEST/sface.onnx"

echo "Downloading MiniFASNet V2 (anti-spoofing) -> $DEST/minifasnet_v2.onnx"
curl -fSL "$MINIFASNET_V2_URL" -o "$DEST/minifasnet_v2.onnx"

echo "Downloading MiniFASNet V1SE (anti-spoofing) -> $DEST/minifasnet_v1se.onnx"
curl -fSL "$MINIFASNET_V1SE_URL" -o "$DEST/minifasnet_v1se.onnx"

echo "Done. Models in $DEST (YuNet detector is committed)."
