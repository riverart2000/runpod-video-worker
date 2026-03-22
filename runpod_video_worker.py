from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import boto3
import torch
from diffusers import AnimateDiffPipeline, AutoencoderKL, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_video


ROOT_DIR = Path(__file__).resolve().parent
ENV_CANDIDATE_PATHS = (
    ROOT_DIR / ".env",
    ROOT_DIR.parent / "clipflow" / "etc" / ".env",
)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_key = key.strip()
        if not env_key or env_key in os.environ:
            continue
        os.environ[env_key] = value.strip().strip('"').strip("'")


for env_path in ENV_CANDIDATE_PATHS:
    load_env_file(env_path)


DELETE_LOCAL_OUTPUT_AFTER_UPLOAD = os.environ.get("DELETE_LOCAL_OUTPUT_AFTER_UPLOAD", "true").strip().lower() not in {"0", "false", "no", "off"}
DEFAULT_MODEL_ID = os.environ.get("DEFAULT_MODEL_ID", "emilianJR/epiCRealism")
DEFAULT_MOTION_ADAPTER_ID = os.environ.get("DEFAULT_MOTION_ADAPTER_ID", "wangfuyun/AnimateLCM")
DEFAULT_LORA_REPOSITORY = os.environ.get("DEFAULT_LORA_REPOSITORY", DEFAULT_MOTION_ADAPTER_ID)
DEFAULT_LORA_WEIGHT_NAME = os.environ.get("DEFAULT_LORA_WEIGHT_NAME", "AnimateLCM_sd15_t2v_lora.safetensors")
DEFAULT_VAE_ID = os.environ.get("DEFAULT_VAE_ID", "stabilityai/sd-vae-ft-mse")
DEFAULT_NEGATIVE_PROMPT = os.environ.get(
    "DEFAULT_NEGATIVE_PROMPT",
    "blurry, low quality, distorted, flicker, jitter, warped anatomy, bad hands, text, watermark",
)
DEFAULT_NATIVE_WIDTH = int(os.environ.get("DEFAULT_NATIVE_WIDTH", "360"))
DEFAULT_NATIVE_HEIGHT = int(os.environ.get("DEFAULT_NATIVE_HEIGHT", "640"))
DEFAULT_OUTPUT_WIDTH = int(os.environ.get("DEFAULT_OUTPUT_WIDTH", "720"))
DEFAULT_OUTPUT_HEIGHT = int(os.environ.get("DEFAULT_OUTPUT_HEIGHT", "1280"))
DEFAULT_VIDEO_FRAMES = int(os.environ.get("DEFAULT_VIDEO_FRAMES", "16"))
MAX_VIDEO_FRAMES = int(os.environ.get("MAX_VIDEO_FRAMES", "24"))
DEFAULT_VIDEO_FPS = int(os.environ.get("DEFAULT_VIDEO_FPS", "6"))
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "8"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "12"))
DEFAULT_GUIDANCE_SCALE = float(os.environ.get("DEFAULT_GUIDANCE_SCALE", "1.8"))
DEFAULT_LORA_SCALE = float(os.environ.get("DEFAULT_LORA_SCALE", "0.8"))
DEFAULT_SEED = int(os.environ.get("DEFAULT_SEED", "12345"))
DEFAULT_DECODE_CHUNK_SIZE = int(os.environ.get("DEFAULT_DECODE_CHUNK_SIZE", "8"))
FFMPEG_PRESET = os.environ.get("FFMPEG_PRESET", "veryfast")
FFMPEG_CRF = int(os.environ.get("FFMPEG_CRF", "23"))

PIPELINE_LOCK = threading.Lock()
JOB_LOCK = threading.Lock()
PIPELINE: AnimateDiffPipeline | None = None


@dataclass(frozen=True)
class JobSpec:
    prompt: str
    negative_prompt: str
    native_width: int
    native_height: int
    output_width: int
    output_height: int
    frames: int
    fps: int
    steps: int
    guidance_scale: float
    seed: int


def process_runpod_job(event: dict[str, Any]) -> dict[str, Any]:
    job_input = dict(event.get("input") or {})
    job_id = resolve_job_id(event, job_input)
    job_spec = build_job_spec(job_input, job_id)

    with JOB_LOCK:
        pipeline = get_pipeline()
        output_path = render_video(job_id, job_spec, pipeline)
        s3_result = upload_video_to_s3(job_id, output_path)
        cleaned_up = cleanup_local_output(output_path)

    return {
        "status": "COMPLETED",
        "job_id": job_id,
        "model_id": DEFAULT_MODEL_ID,
        "motion_adapter_id": DEFAULT_MOTION_ADAPTER_ID,
        "native_width": job_spec.native_width,
        "native_height": job_spec.native_height,
        "output_width": job_spec.output_width,
        "output_height": job_spec.output_height,
        "frames": job_spec.frames,
        "fps": job_spec.fps,
        "steps": job_spec.steps,
        "guidance_scale": job_spec.guidance_scale,
        "seed": job_spec.seed,
        "s3_bucket": s3_result["bucket"],
        "s3_key": s3_result["key"],
        "video_url": s3_result["url"],
        "expected_video_url": s3_result["expected_url"],
        "local_output_deleted": cleaned_up,
    }


def resolve_job_id(event: dict[str, Any], job_input: dict[str, Any]) -> str:
    candidates = [
        event.get("id"),
        event.get("job_id"),
        job_input.get("job_id"),
        job_input.get("request_id"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return uuid.uuid4().hex


def build_job_spec(job_input: dict[str, Any], job_id: str) -> JobSpec:
    prompt = str(job_input.get("prompt") or job_input.get("video_prompt") or "").strip()
    if not prompt:
        raise ValueError("input.prompt or input.video_prompt is required.")

    native_width = normalize_dimension(job_input.get("width"), DEFAULT_NATIVE_WIDTH)
    native_height = normalize_dimension(job_input.get("height"), DEFAULT_NATIVE_HEIGHT)
    output_width = normalize_dimension(job_input.get("output_width") or job_input.get("target_width"), DEFAULT_OUTPUT_WIDTH)
    output_height = normalize_dimension(job_input.get("output_height") or job_input.get("target_height"), DEFAULT_OUTPUT_HEIGHT)
    frames = clamp(to_int(job_input.get("frames"), DEFAULT_VIDEO_FRAMES), 8, MAX_VIDEO_FRAMES)
    fps = clamp(to_int(job_input.get("fps"), DEFAULT_VIDEO_FPS), 1, 24)
    steps = clamp(to_int(job_input.get("steps"), DEFAULT_STEPS), 4, MAX_STEPS)
    guidance_scale = clamp_float(
        to_float(job_input.get("guidance_scale"), to_float(job_input.get("cfg"), DEFAULT_GUIDANCE_SCALE)),
        1.0,
        4.0,
    )
    negative_prompt = str(job_input.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT).strip()

    if job_input.get("seed") is None:
        seed = stable_seed(job_id)
    else:
        seed = to_int(job_input.get("seed"), DEFAULT_SEED)

    return JobSpec(
        prompt=prompt,
        negative_prompt=negative_prompt,
        native_width=native_width,
        native_height=native_height,
        output_width=output_width,
        output_height=output_height,
        frames=frames,
        fps=fps,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )


def get_pipeline() -> AnimateDiffPipeline:
    global PIPELINE
    with PIPELINE_LOCK:
        if PIPELINE is not None:
            return PIPELINE

        cache_dir = resolve_cache_dir()
        use_cuda = torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32

        if use_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        motion_adapter = MotionAdapter.from_pretrained(
            DEFAULT_MOTION_ADAPTER_ID,
            cache_dir=str(cache_dir),
            torch_dtype=dtype,
        )
        vae = AutoencoderKL.from_pretrained(
            DEFAULT_VAE_ID,
            cache_dir=str(cache_dir),
            torch_dtype=dtype,
        )
        pipeline = AnimateDiffPipeline.from_pretrained(
            DEFAULT_MODEL_ID,
            motion_adapter=motion_adapter,
            vae=vae,
            cache_dir=str(cache_dir),
            torch_dtype=dtype,
        )
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config, beta_schedule="linear")
        pipeline.load_lora_weights(
            DEFAULT_LORA_REPOSITORY,
            weight_name=DEFAULT_LORA_WEIGHT_NAME,
            adapter_name="lcm",
            cache_dir=str(cache_dir),
        )
        pipeline.set_adapters(["lcm"], [DEFAULT_LORA_SCALE])
        pipeline.enable_vae_slicing()
        pipeline.enable_attention_slicing()

        if hasattr(pipeline.unet, "enable_forward_chunking"):
            pipeline.unet.enable_forward_chunking(chunk_size=1, dim=1)

        if use_cuda:
            pipeline.to("cuda")
        else:
            pipeline.to("cpu")

        PIPELINE = pipeline
        return PIPELINE


def render_video(job_id: str, job_spec: JobSpec, pipeline: AnimateDiffPipeline) -> Path:
    use_cuda = torch.cuda.is_available()
    generator_device = "cuda" if use_cuda else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(job_spec.seed)

    temp_dir = Path(tempfile.mkdtemp(prefix=f"runpod-video-{job_id}-", dir=str(ROOT_DIR)))
    native_video_path = temp_dir / f"{job_id}-native.mp4"
    output_video_path = temp_dir / f"{job_id}.mp4"

    try:
        result = pipeline(
            prompt=job_spec.prompt,
            negative_prompt=job_spec.negative_prompt,
            num_frames=job_spec.frames,
            guidance_scale=job_spec.guidance_scale,
            num_inference_steps=job_spec.steps,
            width=job_spec.native_width,
            height=job_spec.native_height,
            decode_chunk_size=min(job_spec.frames, DEFAULT_DECODE_CHUNK_SIZE),
            generator=generator,
            output_type="pil",
        )
        frames = result.frames[0]
        export_to_video(frames, str(native_video_path), fps=job_spec.fps)
        transcode_video(native_video_path, output_video_path, job_spec)
        return output_video_path
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    finally:
        if use_cuda:
            torch.cuda.empty_cache()


def transcode_video(input_path: Path, output_path: Path, job_spec: JobSpec) -> None:
    scale_filter = (
        f"scale={job_spec.output_width}:{job_spec.output_height}:force_original_aspect_ratio=decrease,"
        f"pad={job_spec.output_width}:{job_spec.output_height}:(ow-iw)/2:(oh-ih)/2:black,"
        f"fps={job_spec.fps},format=yuv420p"
    )
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        scale_filter,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        FFMPEG_PRESET,
        "-crf",
        str(FFMPEG_CRF),
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip() or completed.stdout.strip()}")

    input_path.unlink(missing_ok=True)


def resolve_cache_dir() -> Path:
    configured = os.environ.get("MODEL_CACHE_DIR", "").strip()
    if configured:
        cache_dir = Path(configured)
    elif Path("/runpod-volume").exists():
        cache_dir = Path("/runpod-volume/hf-cache")
    else:
        cache_dir = ROOT_DIR / "models_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def stable_seed(job_id: str) -> int:
    digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def normalize_dimension(value: Any, default: int) -> int:
    dimension = max(64, to_int(value, default))
    return dimension - (dimension % 8)


def upload_video_to_s3(job_id: str, output_path: Path) -> dict[str, str]:
    base_url = os.environ.get("RUNPOD_S3_URL", "").strip()
    if not base_url:
        raise RuntimeError("RUNPOD_S3_URL is required.")

    destination = parse_s3_destination(base_url)
    key = f"{destination['prefix']}{job_id}.mp4"
    bucket = destination["bucket"]
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or destination.get("region") or None

    client = boto3.client("s3", region_name=region)
    extra_args: dict[str, str] = {"ContentType": "video/mp4"}
    acl = os.environ.get("RUNPOD_S3_UPLOAD_ACL", "").strip()
    if acl:
        extra_args["ACL"] = acl
    client.upload_file(str(output_path), bucket, key, ExtraArgs=extra_args)

    expected_url = f"{destination['base_url']}{quote(job_id)}.mp4"
    presign_enabled = os.environ.get("RUNPOD_S3_PRESIGN", "false").strip().lower() in {"1", "true", "yes", "on"}
    url = expected_url
    if presign_enabled:
        expires_in = to_int(os.environ.get("RUNPOD_S3_PRESIGN_EXPIRES_SECONDS"), 86400)
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    return {
        "bucket": bucket,
        "key": key,
        "url": url,
        "expected_url": expected_url,
    }


def parse_s3_destination(base_url: str) -> dict[str, str]:
    normalized = base_url.strip()
    if not normalized.endswith("/"):
        normalized = f"{normalized}/"
    parsed = urlparse(normalized)
    host_parts = parsed.netloc.split(".")
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix = f"{prefix}/"

    bucket = ""
    region = ""
    if len(host_parts) >= 4 and host_parts[1] == "s3":
        bucket = host_parts[0]
        region = host_parts[2]
    elif parsed.path.strip("/"):
        path_parts = parsed.path.strip("/").split("/", 1)
        bucket = path_parts[0]
        prefix = f"{path_parts[1]}/" if len(path_parts) > 1 and path_parts[1] else ""
        if len(host_parts) >= 3 and host_parts[0] == "s3":
            region = host_parts[1]

    if not bucket:
        raise ValueError(f"Unable to determine S3 bucket from RUNPOD_S3_URL={base_url}")

    if prefix and not normalized.endswith(prefix):
        normalized = normalized.rstrip("/") + "/"

    return {
        "bucket": bucket,
        "prefix": prefix,
        "region": region,
        "base_url": normalized,
    }


def cleanup_local_output(output_path: Path) -> bool:
    output_dir = output_path.parent
    try:
        if DELETE_LOCAL_OUTPUT_AFTER_UPLOAD:
            shutil.rmtree(output_dir, ignore_errors=True)
            return True
        return False
    except OSError:
        return False


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default