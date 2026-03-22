# RunPod Queue Worker

This folder is a standalone queue-based RunPod Serverless worker repository for generating one MP4 per job.

## Contract

- Each call to the RunPod `/run` API returns a `job_id` immediately.
- The worker uses that `job_id` as the S3 object name: `{job_id}.mp4`.
- The file is uploaded under `RUNPOD_S3_URL`, for example:
  `https://revenuemindproai.s3.eu-west-1.amazonaws.com/runpod_videos/{job_id}.mp4`
- The handler only reports success after the S3 upload succeeds.

## Files

- `handler.py` - RunPod queue worker entrypoint.
- `runpod_video_worker.py` - ComfyUI startup, workflow submission, output detection, and S3 upload.
- `startup.sh` - Launches local ComfyUI inside the container.
- `workflows/default-video-workflow.json` - Bundled default AnimateDiff workflow template.
- `Dockerfile` - Image definition for GitHub-to-RunPod deployment.

## Input schema

Send job input as JSON in `event.input`.

```json
{
  "prompt": "cinematic slow motion shot of a lion walking at sunset",
  "width": 512,
  "height": 512,
  "frames": 24,
  "fps": 6,
  "negative_prompt": "blurry, low quality, distorted",
  "steps": 20,
  "cfg": 7,
  "seed": 12345,
  "workflow": null
}
```

Notes:

- `prompt` is required unless `video_prompt` is supplied.
- `workflow` is optional. If omitted, the bundled default workflow is used.
- If `workflow` is supplied, it can be a plain workflow object or a full payload containing `input.workflow`.
- Workflow templates may use placeholders such as `__VIDEO_PROMPT__`, `__VIDEO_WIDTH__`, `__VIDEO_HEIGHT__`, `__VIDEO_FRAMES__`, and `__VIDEO_FPS__`.

## Output schema

```json
{
  "status": "COMPLETED",
  "job_id": "rp-123456",
  "prompt_id": "prompt-id-from-comfyui",
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
- `DEFAULT_VIDEO_WIDTH`, `DEFAULT_VIDEO_HEIGHT`, `DEFAULT_VIDEO_FRAMES`, `DEFAULT_VIDEO_FPS`
- `DEFAULT_CHECKPOINT_NAME`, `DEFAULT_MOTION_MODULE`

## Deployment notes

This repo targets the queue-based RunPod Serverless worker model. The container entrypoint is `python handler.py`, and the worker processes one job per handler invocation.

For strict sequential processing, configure the RunPod endpoint to use a single worker.

For local testing inside this workspace, the worker also looks for environment variables in `.env` at the repo root and then `../clipflow/etc/.env` if those files exist. In standalone GitHub/RunPod deployment, set the same environment variables directly on the RunPod endpoint.