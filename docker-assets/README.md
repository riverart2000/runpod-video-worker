# Docker Assets

This folder is for build-time assets that should be baked into the worker image so the RunPod worker avoids cold-start downloads.

Supported optional subfolders:

- `hf-cache/` - pre-populated Hugging Face cache content copied directly into `MODEL_CACHE_DIR`
- `comfyui-models/checkpoints/` - local checkpoint files copied into `ComfyUI/models/checkpoints`
- `comfyui-models/animatediff_models/` - local motion model files copied into `ComfyUI/models/animatediff_models`
- `comfyui-models/loras/` - local LoRA files copied into `ComfyUI/models/loras`

If these folders are empty, the Docker build can still fetch assets from Hugging Face when the related build args are provided.