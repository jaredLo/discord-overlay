#!/usr/bin/env bash
# Downloads Silero VAD v4 ONNX model into Android assets directory.
# Pin to v4.0 — must match ONNX Runtime 1.20.0 opset compatibility.
# Do NOT upgrade model version without verifying input/output tensor shapes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ASSETS_DIR="$SCRIPT_DIR/../android/app/src/main/assets"
MODEL_URL="https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx"
MODEL_FILE="$ASSETS_DIR/silero_vad.onnx"

mkdir -p "$ASSETS_DIR"

if [ -f "$MODEL_FILE" ]; then
    echo "Model already exists at $MODEL_FILE"
    echo "Delete it first if you want to re-download."
    exit 0
fi

echo "Downloading Silero VAD v4.0 ONNX model..."
curl -fSL "$MODEL_URL" -o "$MODEL_FILE"

SIZE=$(wc -c < "$MODEL_FILE" | tr -d ' ')
echo "Downloaded: $MODEL_FILE ($SIZE bytes)"
echo ""
echo "Expected: ~1.5-2MB. If significantly different, the URL may have changed."
echo "Verify at: https://github.com/snakers4/silero-vad/tree/v4.0/files"
