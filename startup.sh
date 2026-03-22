#!/bin/bash
set -euo pipefail

COMFYUI_DIR="${COMFYUI_DIR:-/comfyui}"
COMFYUI_PORT="${COMFYUI_PORT:-8188}"

if [[ ! -f "$COMFYUI_DIR/main.py" ]]; then
  echo "ComfyUI main.py not found in $COMFYUI_DIR" >&2
  exit 1
fi

exec python "$COMFYUI_DIR/main.py" \
  --listen 0.0.0.0 \
  --port "$COMFYUI_PORT" \
  --enable-cors-header
