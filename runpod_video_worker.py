from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import boto3
import torch
from diffusers import AnimateDiffPipeline, AutoencoderKL, LCMScheduler, MotionAdapter, StableDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image, ImageOps


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


def bootstrap_huggingface_cache_env() -> None:
    configured = os.environ.get("MODEL_CACHE_DIR", "").strip()
    if configured:
        cache_dir = Path(configured)
    elif Path("/runpod-volume").exists():
        cache_dir = Path("/runpod-volume/hf-cache")
    else:
        cache_dir = ROOT_DIR / "models_cache"

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    os.environ.setdefault("DIFFUSERS_CACHE", str(cache_dir))


bootstrap_huggingface_cache_env()

from comfyui_backend import ComfyVideoJobSpec, render_video_with_comfyui


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
DEFAULT_NATIVE_WIDTH = int(os.environ.get("DEFAULT_NATIVE_WIDTH", "448"))
DEFAULT_NATIVE_HEIGHT = int(os.environ.get("DEFAULT_NATIVE_HEIGHT", "768"))
DEFAULT_OUTPUT_WIDTH = int(os.environ.get("DEFAULT_OUTPUT_WIDTH", "720"))
DEFAULT_OUTPUT_HEIGHT = int(os.environ.get("DEFAULT_OUTPUT_HEIGHT", "1280"))
DEFAULT_VIDEO_FRAMES = int(os.environ.get("DEFAULT_VIDEO_FRAMES", "16"))
MAX_VIDEO_FRAMES = int(os.environ.get("MAX_VIDEO_FRAMES", "24"))
DEFAULT_VIDEO_FPS = int(os.environ.get("DEFAULT_VIDEO_FPS", "8"))
DEFAULT_VIDEO_STEPS = int(os.environ.get("DEFAULT_STEPS", "8"))
MAX_VIDEO_STEPS = int(os.environ.get("MAX_STEPS", "12"))
DEFAULT_VIDEO_GUIDANCE_SCALE = float(os.environ.get("DEFAULT_GUIDANCE_SCALE", "1.5"))
DEFAULT_IMAGE_STEPS = int(os.environ.get("DEFAULT_IMAGE_STEPS", "20"))
MAX_IMAGE_STEPS = int(os.environ.get("MAX_IMAGE_STEPS", "30"))
DEFAULT_IMAGE_GUIDANCE_SCALE = float(os.environ.get("DEFAULT_IMAGE_GUIDANCE_SCALE", "7.0"))
DEFAULT_MEDIA_TYPE = os.environ.get("DEFAULT_MEDIA_TYPE", "video").strip().lower() or "video"
DEFAULT_LORA_SCALE = float(os.environ.get("DEFAULT_LORA_SCALE", "0.9"))
DEFAULT_SEED = int(os.environ.get("DEFAULT_SEED", "12345"))
DEFAULT_DECODE_CHUNK_SIZE = int(os.environ.get("DEFAULT_DECODE_CHUNK_SIZE", "12"))
MIN_CACHE_FREE_GB = float(os.environ.get("MIN_CACHE_FREE_GB", "12"))
DEFAULT_VIDEO_BACKEND = os.environ.get("VIDEO_BACKEND", os.environ.get("WORKER_BACKEND", "comfyui")).strip().lower() or "comfyui"

PIPELINE_LOCK = threading.Lock()
JOB_LOCK = threading.Lock()
VIDEO_PIPELINE: AnimateDiffPipeline | None = None
IMAGE_PIPELINE: StableDiffusionPipeline | None = None


def normalize_image_format(value: str | None) -> str:
    normalized = str(value or "png").strip().lower()
    if normalized in {"jpg", "jpeg"}:
        return "jpg"
    return "png"


DEFAULT_IMAGE_FORMAT = normalize_image_format(os.environ.get("DEFAULT_IMAGE_FORMAT", "png"))


@dataclass(frozen=True)
class JobSpec:
    media_type: str
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
    image_format: str


def log_runtime(message: str) -> None:
    print(f"[runpod-video-worker] {message}", flush=True)


def get_runtime_device_info() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_name = torch.cuda.get_device_name(0) if cuda_available and device_count > 0 else "cpu"
    return {
        "cuda_available": cuda_available,
        "device_count": device_count,
        "device_name": device_name,
    }


def process_runpod_job(event: dict[str, Any]) -> dict[str, Any]:
    started_at = time.perf_counter()
    job_input = dict(event.get("input") or {})
    job_id = resolve_job_id(event, job_input)
    job_spec = build_job_spec(job_input, job_id)
    device_info = get_runtime_device_info()

    log_runtime(
        "job_start "
        f"job_id={job_id} media_type={job_spec.media_type} prompt_chars={len(job_spec.prompt)} "
        f"frames={job_spec.frames} fps={job_spec.fps} steps={job_spec.steps} "
        f"native={job_spec.native_width}x{job_spec.native_height} output={job_spec.output_width}x{job_spec.output_height} "
        f"backend={resolve_video_backend(job_input, job_spec.media_type)} "
        f"cuda_available={device_info['cuda_available']} device_count={device_info['device_count']} device_name={device_info['device_name']}"
    )

    with JOB_LOCK:
        if job_spec.media_type == "image":
            render_started_at = time.perf_counter()
            output_path = render_image(job_id, job_spec, get_image_pipeline())
            render_elapsed = time.perf_counter() - render_started_at
            log_runtime(f"image_render_complete job_id={job_id} elapsed_seconds={render_elapsed:.2f} output_path={output_path}")
            upload_started_at = time.perf_counter()
            s3_result = upload_artifact_to_s3(job_id, output_path, job_spec.image_format, image_content_type(job_spec.image_format))
        else:
            render_started_at = time.perf_counter()
            backend = resolve_video_backend(job_input, job_spec.media_type)
            output_path = render_video_for_backend(job_id, job_spec, backend)
            render_elapsed = time.perf_counter() - render_started_at
            log_runtime(f"video_render_complete job_id={job_id} backend={backend} elapsed_seconds={render_elapsed:.2f} output_path={output_path}")
            upload_started_at = time.perf_counter()
            output_extension = normalize_video_extension(output_path.suffix)
            s3_result = upload_artifact_to_s3(job_id, output_path, output_extension, video_content_type(output_extension))
        upload_elapsed = time.perf_counter() - upload_started_at
        log_runtime(f"artifact_upload_complete job_id={job_id} elapsed_seconds={upload_elapsed:.2f} artifact_url={s3_result['url']}")
        cleaned_up = cleanup_local_output(output_path)

    response = {
        "status": "COMPLETED",
        "job_id": job_id,
        "media_type": job_spec.media_type,
        "model_id": DEFAULT_MODEL_ID,
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
        "artifact_url": s3_result["url"],
        "expected_artifact_url": s3_result["expected_url"],
        "artifact_extension": s3_result["extension"],
        "local_output_deleted": cleaned_up,
        "cuda_available": device_info["cuda_available"],
        "device_name": device_info["device_name"],
    }
    if job_spec.media_type == "video":
        response["motion_adapter_id"] = DEFAULT_MOTION_ADAPTER_ID
        response["video_backend"] = resolve_video_backend(job_input, job_spec.media_type)
        response["video_url"] = s3_result["url"]
        response["expected_video_url"] = s3_result["expected_url"]
        response["video_is_raw_native"] = True
        response["local_postprocess_required"] = True
    else:
        response["image_url"] = s3_result["url"]
        response["expected_image_url"] = s3_result["expected_url"]
    total_elapsed = time.perf_counter() - started_at
    response["total_elapsed_seconds"] = round(total_elapsed, 2)
    log_runtime(f"job_complete job_id={job_id} total_elapsed_seconds={total_elapsed:.2f}")
    return response


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
    media_type = str(job_input.get("type") or job_input.get("media_type") or DEFAULT_MEDIA_TYPE).strip().lower()
    if media_type not in {"video", "image"}:
        raise ValueError("input.type must be either 'video' or 'image'.")

    prompt = str(job_input.get("prompt") or job_input.get("video_prompt") or job_input.get("image_prompt") or "").strip()
    if not prompt:
        raise ValueError("input.prompt, input.video_prompt, or input.image_prompt is required.")

    native_width = normalize_dimension(job_input.get("width"), DEFAULT_NATIVE_WIDTH)
    native_height = normalize_dimension(job_input.get("height"), DEFAULT_NATIVE_HEIGHT)
    output_width = normalize_dimension(job_input.get("output_width") or job_input.get("target_width"), DEFAULT_OUTPUT_WIDTH)
    output_height = normalize_dimension(job_input.get("output_height") or job_input.get("target_height"), DEFAULT_OUTPUT_HEIGHT)
    negative_prompt = str(job_input.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT).strip()

    if media_type == "image":
        frames = 1
        fps = 1
        steps = clamp(to_int(job_input.get("steps"), DEFAULT_IMAGE_STEPS), 4, MAX_IMAGE_STEPS)
        guidance_scale = clamp_float(
            to_float(job_input.get("guidance_scale"), to_float(job_input.get("cfg"), DEFAULT_IMAGE_GUIDANCE_SCALE)),
            1.0,
            20.0,
        )
        image_format = normalize_image_format(job_input.get("image_format") or job_input.get("format") or DEFAULT_IMAGE_FORMAT)
    else:
        frames = clamp(to_int(job_input.get("frames"), DEFAULT_VIDEO_FRAMES), 8, MAX_VIDEO_FRAMES)
        fps = clamp(to_int(job_input.get("fps"), DEFAULT_VIDEO_FPS), 1, 24)
        steps = clamp(to_int(job_input.get("steps"), DEFAULT_VIDEO_STEPS), 4, MAX_VIDEO_STEPS)
        guidance_scale = clamp_float(
            to_float(job_input.get("guidance_scale"), to_float(job_input.get("cfg"), DEFAULT_VIDEO_GUIDANCE_SCALE)),
            1.0,
            4.0,
        )
        image_format = DEFAULT_IMAGE_FORMAT

    seed = stable_seed(job_id) if job_input.get("seed") is None else to_int(job_input.get("seed"), DEFAULT_SEED)

    return JobSpec(
        media_type=media_type,
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
        image_format=image_format,
    )


def get_video_pipeline() -> AnimateDiffPipeline:
    global VIDEO_PIPELINE
    with PIPELINE_LOCK:
        if VIDEO_PIPELINE is not None:
            log_runtime("video_pipeline_cache_hit")
            return VIDEO_PIPELINE

        load_started_at = time.perf_counter()
        cache_dir = resolve_cache_dir()
        use_cuda = torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32
        log_runtime(f"video_pipeline_init_start cache_dir={cache_dir} use_cuda={use_cuda} dtype={dtype}")

        configure_torch_backends(use_cuda)
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
        move_pipeline_to_device(pipeline, use_cuda)
        VIDEO_PIPELINE = pipeline
        elapsed = time.perf_counter() - load_started_at
        log_runtime(f"video_pipeline_init_complete elapsed_seconds={elapsed:.2f} device={'cuda' if use_cuda else 'cpu'}")
        return VIDEO_PIPELINE


def get_image_pipeline() -> StableDiffusionPipeline:
    global IMAGE_PIPELINE
    with PIPELINE_LOCK:
        if IMAGE_PIPELINE is not None:
            log_runtime("image_pipeline_cache_hit")
            return IMAGE_PIPELINE

        load_started_at = time.perf_counter()
        cache_dir = resolve_cache_dir()
        use_cuda = torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32
        log_runtime(f"image_pipeline_init_start cache_dir={cache_dir} use_cuda={use_cuda} dtype={dtype}")

        configure_torch_backends(use_cuda)
        vae = AutoencoderKL.from_pretrained(
            DEFAULT_VAE_ID,
            cache_dir=str(cache_dir),
            torch_dtype=dtype,
        )
        pipeline = StableDiffusionPipeline.from_pretrained(
            DEFAULT_MODEL_ID,
            vae=vae,
            cache_dir=str(cache_dir),
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline.enable_vae_slicing()
        pipeline.enable_attention_slicing()
        move_pipeline_to_device(pipeline, use_cuda)
        IMAGE_PIPELINE = pipeline
        elapsed = time.perf_counter() - load_started_at
        log_runtime(f"image_pipeline_init_complete elapsed_seconds={elapsed:.2f} device={'cuda' if use_cuda else 'cpu'}")
        return IMAGE_PIPELINE


def configure_torch_backends(use_cuda: bool) -> None:
    if not use_cuda:
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def move_pipeline_to_device(pipeline: Any, use_cuda: bool) -> None:
    pipeline.to("cuda" if use_cuda else "cpu")


def render_video(job_id: str, job_spec: JobSpec, pipeline: AnimateDiffPipeline) -> Path:
    use_cuda = torch.cuda.is_available()
    generator = torch.Generator(device=("cuda" if use_cuda else "cpu")).manual_seed(job_spec.seed)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"runpod-video-{job_id}-", dir=str(ROOT_DIR)))
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
        export_to_video(result.frames[0], str(output_video_path), fps=job_spec.fps)
        return output_video_path
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    finally:
        clear_cuda_cache(use_cuda)


def render_video_for_backend(job_id: str, job_spec: JobSpec, backend: str) -> Path:
    if backend == "comfyui":
        return render_video_with_comfyui(
            job_id,
            ComfyVideoJobSpec(
                prompt=job_spec.prompt,
                negative_prompt=job_spec.negative_prompt,
                native_width=job_spec.native_width,
                native_height=job_spec.native_height,
                frames=job_spec.frames,
                fps=job_spec.fps,
                steps=job_spec.steps,
                guidance_scale=job_spec.guidance_scale,
                seed=job_spec.seed,
            ),
        )
    return render_video(job_id, job_spec, get_video_pipeline())


def resolve_video_backend(job_input: dict[str, Any], media_type: str) -> str:
    if media_type != "video":
        return "diffusers"
    backend = str(job_input.get("backend") or job_input.get("video_backend") or DEFAULT_VIDEO_BACKEND).strip().lower()
    if backend not in {"diffusers", "comfyui"}:
        raise ValueError("input.backend must be either 'diffusers' or 'comfyui'.")
    return backend


def render_image(job_id: str, job_spec: JobSpec, pipeline: StableDiffusionPipeline) -> Path:
    use_cuda = torch.cuda.is_available()
    generator = torch.Generator(device=("cuda" if use_cuda else "cpu")).manual_seed(job_spec.seed)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"runpod-image-{job_id}-", dir=str(ROOT_DIR)))
    output_image_path = temp_dir / f"{job_id}.{job_spec.image_format}"

    try:
        result = pipeline(
            prompt=job_spec.prompt,
            negative_prompt=job_spec.negative_prompt,
            guidance_scale=job_spec.guidance_scale,
            num_inference_steps=job_spec.steps,
            width=job_spec.native_width,
            height=job_spec.native_height,
            generator=generator,
            output_type="pil",
        )
        image = result.images[0].convert("RGB")
        image = ImageOps.pad(image, (job_spec.output_width, job_spec.output_height), method=Image.Resampling.LANCZOS, color="black")
        image.save(output_image_path, format=image_save_format(job_spec.image_format), optimize=True)
        return output_image_path
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    finally:
        clear_cuda_cache(use_cuda)


def resolve_cache_dir() -> Path:
    configured = os.environ.get("MODEL_CACHE_DIR", "").strip()
    if configured:
        cache_dir = Path(configured)
    elif Path("/runpod-volume").exists():
        cache_dir = Path("/runpod-volume/hf-cache")
    else:
        cache_dir = ROOT_DIR / "models_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ensure_cache_has_free_space(cache_dir)
    return cache_dir


def ensure_cache_has_free_space(cache_dir: Path) -> None:
    try:
        usage = shutil.disk_usage(cache_dir)
    except OSError:
        return

    required_bytes = int(MIN_CACHE_FREE_GB * 1024 * 1024 * 1024)
    if usage.free >= required_bytes:
        return

    free_gb = usage.free / (1024 * 1024 * 1024)
    raise RuntimeError(
        "Insufficient disk space for Hugging Face model download. "
        f"Cache path '{cache_dir}' has only {free_gb:.2f} GB free, but at least {MIN_CACHE_FREE_GB:.0f} GB is recommended. "
        "Mount a larger /runpod-volume or set MODEL_CACHE_DIR to a larger disk path."
    )


def stable_seed(job_id: str) -> int:
    digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def normalize_dimension(value: Any, default: int) -> int:
    dimension = max(64, to_int(value, default))
    return dimension - (dimension % 8)


def upload_artifact_to_s3(job_id: str, output_path: Path, extension: str, content_type: str) -> dict[str, str]:
    base_url = os.environ.get("RUNPOD_S3_URL", "").strip()
    if not base_url:
        raise RuntimeError("RUNPOD_S3_URL is required.")

    destination = parse_s3_destination(base_url)
    key = f"{destination['prefix']}{job_id}.{extension}"
    bucket = destination["bucket"]
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or destination.get("region") or None
    client = boto3.client("s3", region_name=region)
    extra_args: dict[str, str] = {"ContentType": content_type}
    acl = os.environ.get("RUNPOD_S3_UPLOAD_ACL", "").strip()
    if acl:
        extra_args["ACL"] = acl
    client.upload_file(str(output_path), bucket, key, ExtraArgs=extra_args)

    expected_url = f"{destination['base_url']}{quote(job_id)}.{extension}"
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
        "extension": extension,
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


def clear_cuda_cache(use_cuda: bool) -> None:
    if use_cuda:
        torch.cuda.empty_cache()


def image_content_type(image_format: str) -> str:
    return "image/jpeg" if image_format == "jpg" else "image/png"


def video_content_type(video_format: str) -> str:
    if video_format == "webm":
        return "video/webm"
    if video_format == "mkv":
        return "video/x-matroska"
    if video_format == "mov":
        return "video/quicktime"
    return "video/mp4"


def normalize_video_extension(value: str) -> str:
    normalized = value.strip().lower().lstrip(".")
    if normalized in {"webm", "mkv", "mov", "mp4"}:
        return normalized
    return "mp4"


def image_save_format(image_format: str) -> str:
    return "JPEG" if image_format == "jpg" else "PNG"


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