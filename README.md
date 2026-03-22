# RunPod Queue Worker

This folder is a standalone queue-based RunPod Serverless worker repository for generating either one raw MP4 or one image per job with direct diffusers pipelines.

The worker is built for small image size and fast inference:

- no ComfyUI
- no custom nodes
- one cached video pipeline and one cached image pipeline
- low native render resolution by default
- raw/native video upload from the worker
- final video formatting handled locally via `../clipflow/bin/finalize_runpod_video.sh`

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
- Image pipeline: `StableDiffusionPipeline` using the same base model family
- Default native render size: `360x640`
- Default requested final size: `720x1280`
- Default video frames: `16`
- Default video fps: `6`
- Default video steps: `8`
- Default image steps: `20`
- Default image format: `png`

The model weights are downloaded at runtime and cached on first use. The cache defaults to `/runpod-volume/hf-cache` when that path exists, otherwise a local `models_cache/` folder is used.

## Files

- `handler.py` - RunPod queue worker entrypoint.
- `runpod_video_worker.py` - direct video generation, image generation, and S3 upload.
- `Dockerfile` - slim runtime image for GitHub-to-RunPod deployment.

Local workspace helper:

- `../clipflow/bin/finalize_runpod_video.sh` - local script that downloads a raw RunPod video and turns it into the final 720x1280 MP4.

## Input schema

Send job input as JSON in `event.input`.

Video example:

```json
{
  "type": "video",
  "prompt": "cinematic slow motion shot of a lion walking at sunset",
  "width": 360,
  "height": 640,
  "output_width": 720,
  "output_height": 1280,
  "frames": 16,
  "fps": 6,
  "negative_prompt": "blurry, low quality, distorted",
  "steps": 8,
  "guidance_scale": 1.8,
  "seed": 12345
}
```

Image example:

```json
{
  "type": "image",
  "prompt": "portrait of a fox in a misty forest at sunrise, cinematic lighting",
  "width": 360,
  "height": 640,
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
  "motion_adapter_id": "wangfuyun/AnimateLCM",
  "native_width": 360,
  "native_height": 640,
  "output_width": 720,
  "output_height": 1280,
  "frames": 16,
  "fps": 6,
  "steps": 8,
  "guidance_scale": 1.8,
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
  "native_width": 360,
  "native_height": 640,
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
- `MODEL_CACHE_DIR`
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