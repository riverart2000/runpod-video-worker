from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline
from diffusers.utils import export_to_video
from runtime_cache import ensure_cache_has_free_space, resolve_runtime_cache_dir, resolve_runtime_tmp_dir


ROOT_DIR = Path(__file__).resolve().parent

WAN_MODEL_ID = os.environ.get("WAN_MODEL_ID", "Wan-AI/Wan2.1-T2V-14B-Diffusers").strip() or "Wan-AI/Wan2.1-T2V-14B-Diffusers"
WAN_FLOW_SHIFT = float(os.environ.get("WAN_FLOW_SHIFT", "5.0"))
WAN_ENABLE_VAE_TILING = os.environ.get("WAN_ENABLE_VAE_TILING", "true").strip().lower() not in {"0", "false", "no", "off"}
WAN_ENABLE_VAE_SLICING = os.environ.get("WAN_ENABLE_VAE_SLICING", "false").strip().lower() not in {"0", "false", "no", "off"}
WAN_USE_CPU_OFFLOAD = os.environ.get("WAN_USE_CPU_OFFLOAD", "false").strip().lower() not in {"0", "false", "no", "off"}

PIPELINE_LOCK = threading.Lock()
VIDEO_PIPELINE: WanPipeline | None = None


@dataclass(frozen=True)
class WanVideoJobSpec:
    prompt: str
    negative_prompt: str
    native_width: int
    native_height: int
    frames: int
    fps: int
    steps: int
    guidance_scale: float
    seed: int


def log_runtime(message: str) -> None:
    print(f"[runpod-video-worker][wan] {message}", flush=True)


def resolve_cache_dir() -> Path:
    cache_dir = resolve_runtime_cache_dir()
    minimum_free_gb = float(os.environ.get("MIN_CACHE_FREE_GB", "12"))
    ensure_cache_has_free_space(cache_dir, minimum_free_gb, context="Wan model download")
    return cache_dir


def normalize_wan_frame_count(frames: int) -> int:
    rounded = int(round((max(5, frames) - 1) / 4.0)) * 4 + 1
    return max(5, rounded)


def get_video_pipeline() -> WanPipeline:
    global VIDEO_PIPELINE
    with PIPELINE_LOCK:
        if VIDEO_PIPELINE is not None:
            log_runtime("video_pipeline_cache_hit")
            return VIDEO_PIPELINE

        cache_dir = resolve_cache_dir()
        use_cuda = torch.cuda.is_available()
        transformer_dtype = torch.bfloat16 if use_cuda else torch.float32
        log_runtime(
            f"video_pipeline_init_start model_id={WAN_MODEL_ID} cache_dir={cache_dir} use_cuda={use_cuda} "
            f"transformer_dtype={transformer_dtype} cpu_offload={WAN_USE_CPU_OFFLOAD}"
        )

        vae = AutoencoderKLWan.from_pretrained(
            WAN_MODEL_ID,
            subfolder="vae",
            cache_dir=str(cache_dir),
            torch_dtype=torch.float32,
        )
        pipeline = WanPipeline.from_pretrained(
            WAN_MODEL_ID,
            vae=vae,
            cache_dir=str(cache_dir),
            torch_dtype=transformer_dtype,
        )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, flow_shift=WAN_FLOW_SHIFT)
        pipeline.set_progress_bar_config(disable=True)
        if WAN_ENABLE_VAE_TILING:
            pipeline.enable_vae_tiling()
        if WAN_ENABLE_VAE_SLICING:
            pipeline.enable_vae_slicing()

        if use_cuda:
            if WAN_USE_CPU_OFFLOAD:
                pipeline.enable_model_cpu_offload()
            else:
                pipeline.to("cuda")

        VIDEO_PIPELINE = pipeline
        log_runtime("video_pipeline_init_complete")
        return VIDEO_PIPELINE


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def render_video_with_wan(job_id: str, job_spec: WanVideoJobSpec) -> Path:
    pipeline = get_video_pipeline()
    temp_dir = Path(tempfile.mkdtemp(prefix=f"runpod-wan-video-{job_id}-", dir=str(resolve_runtime_tmp_dir())))
    output_video_path = temp_dir / f"{job_id}.mp4"

    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(job_spec.seed)
    frame_count = normalize_wan_frame_count(job_spec.frames)

    try:
        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False):
                frames = pipeline(
                    prompt=job_spec.prompt,
                    negative_prompt=job_spec.negative_prompt,
                    height=job_spec.native_height,
                    width=job_spec.native_width,
                    num_frames=frame_count,
                    num_inference_steps=job_spec.steps,
                    guidance_scale=job_spec.guidance_scale,
                    generator=generator,
                    output_type="pil",
                ).frames[0]
        else:
            frames = pipeline(
                prompt=job_spec.prompt,
                negative_prompt=job_spec.negative_prompt,
                height=job_spec.native_height,
                width=job_spec.native_width,
                num_frames=frame_count,
                num_inference_steps=job_spec.steps,
                guidance_scale=job_spec.guidance_scale,
                generator=generator,
                output_type="pil",
            ).frames[0]

        export_to_video(frames, str(output_video_path), fps=job_spec.fps)
        log_runtime(
            f"render_complete job_id={job_id} frames={frame_count} fps={job_spec.fps} steps={job_spec.steps} "
            f"native={job_spec.native_width}x{job_spec.native_height} output_path={output_video_path}"
        )
        return output_video_path
    except Exception:
        output_video_path.unlink(missing_ok=True)
        temp_dir.rmdir()
        raise
    finally:
        clear_cuda_cache()