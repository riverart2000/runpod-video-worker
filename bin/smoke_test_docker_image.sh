#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-runpod-video-worker:smoke}"
PRELOAD_DIFFUSERS_MODELS="${PRELOAD_DIFFUSERS_MODELS:-false}"
PRELOAD_COMFYUI_MODELS="${PRELOAD_COMFYUI_MODELS:-false}"
SECRET_ARGS=()

if [[ -n "${HF_TOKEN_FILE:-}" ]]; then
    SECRET_ARGS+=(--secret "id=hf_token,src=${HF_TOKEN_FILE}")
elif [[ -n "${HF_TOKEN:-}" ]]; then
    hf_token_tmp="$(mktemp)"
    trap 'rm -f "${hf_token_tmp:-}"' EXIT
    printf '%s' "$HF_TOKEN" > "$hf_token_tmp"
    SECRET_ARGS+=(--secret "id=hf_token,src=${hf_token_tmp}")
fi

cd "$ROOT_DIR"

docker_build_cmd=(
    docker build
    -t "$IMAGE_TAG"
)

if (( ${#SECRET_ARGS[@]} > 0 )); then
        docker_build_cmd+=("${SECRET_ARGS[@]}")
fi

docker_build_cmd+=(
    --build-arg PRELOAD_DIFFUSERS_MODELS="$PRELOAD_DIFFUSERS_MODELS"
    --build-arg PRELOAD_COMFYUI_MODELS="$PRELOAD_COMFYUI_MODELS"
    .
)

optional_build_args=(
    COMFYUI_CKPT_NAME
    COMFYUI_MOTION_MODEL_NAME
    COMFYUI_LORA_NAME
    COMFYUI_CKPT_SOURCE_REPO
    COMFYUI_CKPT_SOURCE_FILENAME
    COMFYUI_MOTION_MODEL_SOURCE_REPO
    COMFYUI_MOTION_MODEL_SOURCE_FILENAME
    COMFYUI_LORA_SOURCE_REPO
    COMFYUI_LORA_SOURCE_FILENAME
)

for arg_name in "${optional_build_args[@]}"; do
        arg_value="${!arg_name:-}"
        if [[ -n "$arg_value" ]]; then
                docker_build_cmd+=(--build-arg "$arg_name=$arg_value")
        fi
done

"${docker_build_cmd[@]}"

docker run --rm \
  -e PRELOAD_COMFYUI_MODELS="$PRELOAD_COMFYUI_MODELS" \
  -e COMFYUI_CKPT_NAME="${COMFYUI_CKPT_NAME:-}" \
  -e COMFYUI_MOTION_MODEL_NAME="${COMFYUI_MOTION_MODEL_NAME:-}" \
  -e COMFYUI_LORA_NAME="${COMFYUI_LORA_NAME:-}" \
  "$IMAGE_TAG" \
  python - <<'PY'
import os
from pathlib import Path

import comfyui_backend
import runpod_video_worker

required_paths = [
    Path("/opt/ComfyUI/main.py"),
    Path("/opt/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved"),
    Path("/opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite"),
    Path("/opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/video_formats/runpod_h264_mp4.json"),
    Path("/worker/handler.py"),
    Path("/worker/runpod_video_worker.py"),
    Path("/worker/comfyui_backend.py"),
]

missing = [str(path) for path in required_paths if not path.exists()]
if missing:
    raise SystemExit(f"missing required runtime paths: {missing}")

compiled = list(Path("/worker").rglob("*.pyc"))
if not compiled:
    raise SystemExit("expected compiled Python bytecode under /worker")

if os.environ.get("PRELOAD_COMFYUI_MODELS", "false").lower() not in {"0", "false", "no", "off"}:
    expected_files = [
        Path("/opt/ComfyUI/models/checkpoints") / os.environ["COMFYUI_CKPT_NAME"],
        Path("/opt/ComfyUI/models/animatediff_models") / os.environ["COMFYUI_MOTION_MODEL_NAME"],
        Path("/opt/ComfyUI/models/loras") / os.environ["COMFYUI_LORA_NAME"],
    ]
    missing_models = [str(path) for path in expected_files if not path.exists()]
    if missing_models:
        raise SystemExit(f"missing baked ComfyUI model files: {missing_models}")

print("smoke test passed")
PY