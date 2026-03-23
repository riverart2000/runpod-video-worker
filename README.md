# RunPod Queue Worker

This folder is a standalone queue-based RunPod Serverless worker repository for generating either one raw MP4 or one image per job.

It now supports two video backends:

- `wan` - the new default quality-first text-to-video path using `Wan-AI/Wan2.1-T2V-14B-Diffusers`
- `diffusers` - the original direct Python pipeline path
- `comfyui` - a ComfyUI + AnimateLCM + FP16 path for lower-overhead video orchestration

The worker is built around a quality-first Wan text-to-video path for larger GPUs, while preserving the older backends for fallback:

- cached Wan T2V pipeline for higher-fidelity text-to-video scene generation
- optional ComfyUI runtime with `ComfyUI-AnimateDiff-Evolved` and `ComfyUI-VideoHelperSuite`
- one cached direct diffusers video pipeline and one cached image pipeline
- RTX 5090 oriented default native render resolution and step budget
- raw/native video upload from the worker
- final video formatting can still be handled locally after download if needed

## Contract

- Each call to the RunPod `/run` API returns a `job_id` immediately.
- The worker uses that `job_id` as the S3 object name: `{job_id}.mp4` for video or `{job_id}.png` for image by default.
- Video uploads are raw/native MP4s from the model output.
- The file is uploaded under `RUNPOD_S3_URL`, for example `https://revenuemindproai.s3.eu-west-1.amazonaws.com/runpod_videos/{job_id}.mp4`.
- The handler only reports success after the S3 upload succeeds.

## Runtime design

- Base model: `emilianJR/epiCRealism`
- Video motion adapter: `wangfuyun/AnimateLCM`
- Video LoRA: `AnimateLCM_sd15_t2v_lora.safetensors`
- Default quality-first video path: `Wan-AI/Wan2.1-T2V-14B-Diffusers`
- Image pipeline: `StableDiffusionPipeline` using the same base model family
- ComfyUI video path: `CheckpointLoaderSimple` + `LoraLoader` + `ADE_AnimateDiffLoaderGen1` + `KSampler(lcm)` + `VHS_VideoCombine`
- Default native render size: `720x1280`
- Default requested final size: `720x1280`
- Default video frames: `49`
- Default video fps: `16`
- Default video steps: `30`
- Default video guidance scale: `5.0`
- Default image steps: `20`
- Default image format: `png`

The Docker image is now designed to bake runtime assets in during image build. By default it uses `/opt/models/hf-cache` as the baked Hugging Face cache inside the image, so the worker can start without re-downloading the default diffusers stack.

For GitHub-to-RunPod remote builds, the default image path now keeps all model preload flags disabled. This is intentional: large baked model layers can push the remote build over RunPod's 30 minute build limit even if the Python code is correct.

When the `wan` backend is enabled, the worker loads a cached `WanPipeline` directly in Python and renders clips without ComfyUI. When the `comfyui` backend is enabled, ComfyUI is started headlessly inside the worker container and the worker submits a generated AnimateLCM workflow to the local ComfyUI API.

For RunPod deployment, use a persistent volume when possible. The worker now prefers `/runpod-volume/hf-cache` over the baked `/opt/models/hf-cache` path whenever `/runpod-volume` is mounted, unless you explicitly set `MODEL_CACHE_DIR` to something else. It also disables the Hugging Face Xet download path and checks free space in the cache directory before loading models so low-disk failures are clearer.

## Files

- `handler.py` - RunPod queue worker entrypoint.
- `runpod_video_worker.py` - direct video generation, image generation, and S3 upload.
- `comfyui_backend.py` - headless ComfyUI launcher and AnimateLCM workflow submitter.
- `bin/preload_runtime_assets.py` - build-time downloader/copier for diffusers and ComfyUI assets.
- `Dockerfile` - slim runtime image for GitHub-to-RunPod deployment.
- `docker-assets/README.md` - layout for local model assets copied into the image at build time.
- `workflows/comfyui_video_formats/runpod_h264_mp4.json` - repo-owned MP4 output format copied into VideoHelperSuite.

Local workspace helper:

- `../clipflow/bin/finalize_runpod_video.sh` - local script that downloads a raw RunPod video and turns it into the final 720x1280 MP4.

## Input schema

Send job input as JSON in `event.input`.

Video example:

```json
{
  "type": "video",
  "prompt": "cinematic slow motion shot of a lion walking at sunset",
  "width": 448,
  "height": 768,
  "output_width": 720,
  "output_height": 1280,
  "frames": 16,
  "fps": 8,
  "negative_prompt": "blurry, low quality, distorted",
  "steps": 8,
  "guidance_scale": 1.5,
  "seed": 12345
}
```

Image example:

```json
{
  "type": "image",
  "prompt": "portrait of a fox in a misty forest at sunrise, cinematic lighting",
  "width": 448,
  "height": 768,
  "output_width": 720,
  "output_height": 1280,
  "steps": 20,
  "guidance_scale": 7,
  "image_format": "png",
  "seed": 12345
}
```

Notes:

- `type` may be `video` or `image`. It defaults to `video`.
- `backend` may be `wan`, `diffusers`, or `comfyui` for video jobs. If omitted, `WORKER_BACKEND` / `VIDEO_BACKEND` decides.
- `prompt` is required unless `video_prompt` or `image_prompt` is supplied.
- `width` and `height` are the native generation size.
- `output_width` and `output_height` are preserved in the job metadata for downstream local post-processing.
- If `seed` is omitted, a deterministic seed is derived from `job_id`.
- Width and height are normalized to multiples of 8.
- Video frames and steps are capped to keep the worker fast and predictable.
- Images support `png` and `jpg` output.

## Output schema

Video completion example:

```json
{
  "status": "COMPLETED",
  "job_id": "rp-123456",
  "media_type": "video",
  "model_id": "emilianJR/epiCRealism",
  "video_backend": "comfyui",
  "motion_adapter_id": "wangfuyun/AnimateLCM",
  "native_width": 448,
  "native_height": 768,
  "output_width": 720,
  "output_height": 1280,
  "frames": 16,
  "fps": 8,
  "steps": 8,
  "guidance_scale": 1.5,
  "seed": 12345,
  "s3_bucket": "revenuemindproai",
  "s3_key": "runpod_videos/rp-123456.mp4",
  "artifact_url": "https://revenuemindproai.s3.eu-west-1.amazonaws.com/runpod_videos/rp-123456.mp4",
  "expected_artifact_url": "https://revenuemindproai.s3.eu-west-1.amazonaws.com/runpod_videos/rp-123456.mp4",
  "artifact_extension": "mp4",
  "video_url": "https://revenuemindproai.s3.eu-west-1.amazonaws.com/runpod_videos/rp-123456.mp4",
  "expected_video_url": "https://revenuemindproai.s3.eu-west-1.amazonaws.com/runpod_videos/rp-123456.mp4",
  "video_is_raw_native": true,
  "local_postprocess_required": true,
  "local_output_deleted": true
}
```

Image completion example:

```json
{
  "status": "COMPLETED",
  "job_id": "rp-7890",
  "media_type": "image",
  "model_id": "emilianJR/epiCRealism",
  "native_width": 448,
  "native_height": 768,
  "output_width": 720,
  "output_height": 1280,
  "frames": 1,
  "fps": 1,
  "steps": 20,
  "guidance_scale": 7,
  "seed": 12345,
  "s3_bucket": "revenuemindproai",
  "s3_key": "runpod_videos/rp-7890.png",
  "artifact_url": "https://revenuemindproai.s3.eu-west-1.amazonaws.com/runpod_videos/rp-7890.png",
  "expected_artifact_url": "https://revenuemindproai.s3.eu-west-1.amazonaws.com/runpod_videos/rp-7890.png",
  "artifact_extension": "png",
  "image_url": "https://revenuemindproai.s3.eu-west-1.amazonaws.com/runpod_videos/rp-7890.png",
  "expected_image_url": "https://revenuemindproai.s3.eu-west-1.amazonaws.com/runpod_videos/rp-7890.png",
  "local_output_deleted": true
}
```

## ComfyUI backend

The ComfyUI path is intended for video jobs only. Image jobs still use the direct diffusers image pipeline.

The Docker image now provisions:

- `ComfyUI`
- `ComfyUI-AnimateDiff-Evolved`
- `ComfyUI-VideoHelperSuite`
- `ffmpeg`
- baked Python bytecode for the worker, ComfyUI, and installed site-packages
- a build-time asset preload pass for Hugging Face cache content and ComfyUI model files

To use the `comfyui` backend you must provide local ComfyUI filenames, not Hugging Face repo ids:

- `COMFYUI_CKPT_NAME` - SD1.5 checkpoint filename under `ComfyUI/models/checkpoints`
- `COMFYUI_MOTION_MODEL_NAME` - AnimateLCM motion model filename under `ComfyUI/models/animatediff_models`
- `COMFYUI_LORA_NAME` - AnimateLCM LoRA filename under `ComfyUI/models/loras`

The worker submits a generated workflow equivalent to:

1. load SD checkpoint
2. load AnimateLCM LoRA
3. encode positive / negative prompts
4. create latent batch with `frames == batch_size`
5. inject AnimateLCM motion model using `ADE_AnimateDiffLoaderGen1`
6. sample with `KSampler` using `sampler_name=lcm`
7. decode frames
8. assemble MP4 with `VHS_VideoCombine`

FP16 is enabled for the ComfyUI server by default through `COMFYUI_FORCE_FP16=true`, which starts ComfyUI with `--force-fp16`.

The baked defaults are now tuned for an RTX 5090 32GB class GPU using the Wan backend:

- native render size `720x1280`
- `49` default frames and `81` max frames
- `16` default fps
- `30` default steps and `40` max steps
- `5.0` default CFG / guidance scale
- `5.0` Wan flow shift for 720P portrait generation
- ComfyUI remains available as a legacy fallback backend

## Build-time asset baking

Remote build default:

- `PRELOAD_DIFFUSERS_MODELS=false`
- `PRELOAD_WAN_MODELS=false`
- `PRELOAD_COMFYUI_MODELS=false`

Use those defaults for GitHub-triggered RunPod builds unless you are certain the baked image can still be exported within the provider's build time limit.

Local strict bake:

- enable the preload flags only when you are intentionally validating or producing a locally baked image
- this is appropriate for local Docker builds, not for the constrained remote builder

The Docker build now shifts as much cold-start work as possible into the image build:

1. clones ComfyUI and required custom nodes during build
2. installs Python dependencies during build
3. preloads default diffusers assets into `/opt/models/hf-cache`
4. copies local ComfyUI model files from `docker-assets/` when present
5. optionally downloads missing ComfyUI model files from Hugging Face during build
6. compiles Python bytecode for `/worker`, `/opt/ComfyUI`, and installed site-packages

Local asset layout is documented in `docker-assets/README.md`.

Useful Docker build args:

- `PRELOAD_DIFFUSERS_MODELS=true|false`
- `PRELOAD_COMFYUI_MODELS=true|false`
- `HF_TOKEN_FILE=/path/to/token-file` for gated/private Hugging Face access during the smoke test
- `COMFYUI_CKPT_NAME`, `COMFYUI_MOTION_MODEL_NAME`, `COMFYUI_LORA_NAME`
- `COMFYUI_CKPT_SOURCE_REPO` and `COMFYUI_CKPT_SOURCE_FILENAME`
- `COMFYUI_MOTION_MODEL_SOURCE_REPO` and `COMFYUI_MOTION_MODEL_SOURCE_FILENAME`
- `COMFYUI_LORA_SOURCE_REPO` and `COMFYUI_LORA_SOURCE_FILENAME`

For the fastest worker startup, put the exact ComfyUI files you want under `docker-assets/comfyui-models/` before building the image. That avoids any model download during container boot.

Current known public source mappings used in the repo examples:

- checkpoint: `Syimbiote/v1-5-pruned-emaonly` / `v1-5-pruned-emaonly.safetensors`
- motion model: `guoyww/animatediff` / `mm_sd_v15_v2.ckpt`
- LoRA: `latent-consistency/lcm-lora-sdv1-5` / `pytorch_lora_weights.safetensors`

## Image smoke test

Use the repo-local smoke test to verify that the image builds and contains the expected runtime pieces:

```bash
bin/smoke_test_docker_image.sh
```

By default it runs the fastest validation mode with both preload paths disabled, which verifies:

- the image builds successfully
- the Wan backend module imports successfully inside the container
- ComfyUI and custom nodes are present
- the repo-owned video format file is present
- worker modules import correctly inside the container
- Python bytecode was compiled into the image

For the fastest local build validation after code changes, keep all preload paths disabled:

```bash
PRELOAD_DIFFUSERS_MODELS=false PRELOAD_WAN_MODELS=false PRELOAD_COMFYUI_MODELS=false bin/smoke_test_docker_image.sh
```

That path is the best way to catch Python, pip, Dockerfile, and import-level breakage without waiting for the full baked-model build.

To smoke test a fully baked ComfyUI image, export the exact ComfyUI filenames and enable the stricter mode:

```bash
export PRELOAD_COMFYUI_MODELS=true
export COMFYUI_CKPT_NAME=your-checkpoint.safetensors
export COMFYUI_MOTION_MODEL_NAME=your-motion-model.safetensors
export COMFYUI_LORA_NAME=AnimateLCM_sd15_t2v_lora.safetensors
bin/smoke_test_docker_image.sh
```

If the image is meant to fetch missing ComfyUI files at build time, also export the matching `*_SOURCE_REPO` and `*_SOURCE_FILENAME` values before running the smoke test.

The smoke test also accepts `HF_TOKEN` or `HF_TOKEN_FILE`; it passes the token into `docker build` as a BuildKit secret instead of baking it into the image configuration.

To validate the baked Wan path specifically, enable Wan preload as well:

```bash
PRELOAD_WAN_MODELS=true PRELOAD_COMFYUI_MODELS=false PRELOAD_DIFFUSERS_MODELS=false bin/smoke_test_docker_image.sh
```

## Local post-processing

For video jobs, use the local script after the raw MP4 is available in S3:

```bash
../clipflow/bin/finalize_runpod_video.sh "https://revenuemindproai.s3.eu-west-1.amazonaws.com/runpod_videos/<job_id>.mp4"
```

This script can also take a local file path instead of a URL and will output a final H.264 MP4 padded/scaled to 720x1280 by default.

## Required environment variables

- `RUNPOD_S3_URL` - Base S3 URL including prefix, for example `https://revenuemindproai.s3.eu-west-1.amazonaws.com/runpod_videos/`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

Optional:

- `AWS_REGION` or `AWS_DEFAULT_REGION`
- `RUNPOD_S3_UPLOAD_ACL` - for example `public-read` if the bucket policy expects ACLs
- `RUNPOD_S3_PRESIGN=true` - return a presigned download URL instead of the deterministic public URL
- `MODEL_CACHE_DIR` - explicit cache path override. If left at the baked default `/opt/models/hf-cache`, runtime will automatically switch to `/runpod-volume/hf-cache` when a RunPod volume is mounted.
- `MIN_CACHE_FREE_GB` - minimum free space required in the model cache path before model download starts. Defaults to `12`.
- `WORKER_BACKEND` or `VIDEO_BACKEND` - `wan`, `diffusers`, or `comfyui`
- `COMFYUI_ROOT`, `COMFYUI_HOST`, `COMFYUI_PORT`
- `COMFYUI_FORCE_FP16`
- `COMFYUI_STARTUP_TIMEOUT_SECONDS`, `COMFYUI_JOB_TIMEOUT_SECONDS`, `COMFYUI_POLL_INTERVAL_SECONDS`
- `COMFYUI_VIDEO_FORMAT`, `COMFYUI_VIDEO_PRESET`, `COMFYUI_VIDEO_CRF`, `COMFYUI_VIDEO_PIX_FMT`
- `COMFYUI_CKPT_NAME`, `COMFYUI_MOTION_MODEL_NAME`, `COMFYUI_LORA_NAME`
- `DEFAULT_MODEL_ID`
- `DEFAULT_MOTION_ADAPTER_ID`
- `DEFAULT_LORA_REPOSITORY`
- `DEFAULT_LORA_WEIGHT_NAME`
- `DEFAULT_VAE_ID`
- `DEFAULT_NATIVE_WIDTH`, `DEFAULT_NATIVE_HEIGHT`
- `DEFAULT_OUTPUT_WIDTH`, `DEFAULT_OUTPUT_HEIGHT`
- `DEFAULT_VIDEO_FRAMES`, `MAX_VIDEO_FRAMES`
- `DEFAULT_VIDEO_FPS`
- `DEFAULT_STEPS`, `MAX_STEPS`
- `DEFAULT_IMAGE_FORMAT`
- `DEFAULT_IMAGE_STEPS`, `MAX_IMAGE_STEPS`
- `DEFAULT_GUIDANCE_SCALE`
- `DEFAULT_IMAGE_GUIDANCE_SCALE`
- `DEFAULT_LORA_SCALE`

## Deployment notes

This repo targets the queue-based RunPod Serverless worker model. The container entrypoint is `python handler.py`, and the worker processes one job at a time inside the container.

For strict sequential processing, configure the RunPod endpoint to use a single worker.

For local testing inside this workspace, the worker also looks for environment variables in `.env` at the repo root and then `../clipflow/etc/.env` if those files exist. In standalone GitHub/RunPod deployment, set the same environment variables directly on the RunPod endpoint.

This image now bakes the non-secret 3090-target runtime defaults directly into the Dockerfile, including the ComfyUI backend selection, ComfyUI model filenames, source mappings, host/port, and S3 base URL. A RunPod deployment should therefore not need endpoint-level configuration for those non-secret values.

The remaining runtime inputs that still must come from RunPod secrets or another secure secret source are credential-like values such as `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and any optional gated Hugging Face token.