# RunPod Queue Worker

This folder is a standalone queue-based RunPod Serverless worker repository for generating one MP4 per job with a direct diffusers pipeline.

The worker is built for small image size and fast inference:

- no ComfyUI
- no custom nodes
- one cached AnimateLCM pipeline
- low native render resolution by default
- fixed upscale and pad to 720x1280 MP4

## Contract

- Each call to the RunPod `/run` API returns a `job_id` immediately.
- The worker uses that `job_id` as the S3 object name: `{job_id}.mp4`.
- The file is uploaded under `RUNPOD_S3_URL`, for example:
  `https://revenuemindproai.s3.eu-west-1.amazonaws.com/runpod_videos/{job_id}.mp4`
- The handler only reports success after the S3 upload succeeds.

## Runtime design

- Base model: `emilianJR/epiCRealism`
- Motion adapter: `wangfuyun/AnimateLCM`
- LoRA: `AnimateLCM_sd15_t2v_lora.safetensors`
- Default native render size: `360x640`
- Default final output size: `720x1280`
- Default frames: `16`
- Default fps: `6`
- Default steps: `8`

The model weights are downloaded at runtime and cached on first use. The cache defaults to `/runpod-volume/hf-cache` when that path exists, otherwise a local `models_cache/` folder is used.

## Files

- `handler.py` - RunPod queue worker entrypoint.
- `runpod_video_worker.py` - direct text-to-video generation, MP4 transcoding, and S3 upload.
- `Dockerfile` - slim CUDA runtime image for GitHub-to-RunPod deployment.

## Input schema

Send job input as JSON in `event.input`.

```json
{
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

Notes:

- `prompt` is required unless `video_prompt` is supplied.
- `width` and `height` are the native generation size, not the final MP4 size.
- `output_width` and `output_height` control the final transcoded MP4 size.
- If `seed` is omitted, a deterministic seed is derived from `job_id`.
- Width and height are normalized to multiples of 8.
- Frames and steps are capped to keep the worker fast and predictable.

## Output schema

```json
{
  "status": "COMPLETED",
  "job_id": "rp-123456",
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
  "video_url": "https://revenuemindproai.s3.eu-west-1.amazonaws.com/runpod_videos/rp-123456.mp4",
  "expected_video_url": "https://revenuemindproai.s3.eu-west-1.amazonaws.com/runpod_videos/rp-123456.mp4",
  "local_output_deleted": true
}
```

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
- `DEFAULT_GUIDANCE_SCALE`
- `DEFAULT_LORA_SCALE`

## Deployment notes

This repo targets the queue-based RunPod Serverless worker model. The container entrypoint is `python handler.py`, and the worker processes one job at a time inside the container.

For strict sequential processing, configure the RunPod endpoint to use a single worker.

For local testing inside this workspace, the worker also looks for environment variables in `.env` at the repo root and then `../clipflow/etc/.env` if those files exist. In standalone GitHub/RunPod deployment, set the same environment variables directly on the RunPod endpoint.