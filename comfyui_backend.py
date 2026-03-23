from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent


def env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped or default


COMFYUI_ROOT = Path(os.environ.get("COMFYUI_ROOT", "/opt/ComfyUI")).resolve()
COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "127.0.0.1").strip() or "127.0.0.1"
COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
COMFYUI_STARTUP_TIMEOUT_SECONDS = float(os.environ.get("COMFYUI_STARTUP_TIMEOUT_SECONDS", "240"))
COMFYUI_JOB_TIMEOUT_SECONDS = float(os.environ.get("COMFYUI_JOB_TIMEOUT_SECONDS", "1800"))
COMFYUI_POLL_INTERVAL_SECONDS = float(os.environ.get("COMFYUI_POLL_INTERVAL_SECONDS", "2"))
COMFYUI_FORCE_FP16 = os.environ.get("COMFYUI_FORCE_FP16", "true").strip().lower() not in {"0", "false", "no", "off"}
COMFYUI_VIDEO_FORMAT = os.environ.get("COMFYUI_VIDEO_FORMAT", "video/runpod_h264_mp4").strip() or "video/runpod_h264_mp4"
COMFYUI_VIDEO_PRESET = os.environ.get("COMFYUI_VIDEO_PRESET", "veryfast").strip() or "veryfast"
COMFYUI_VIDEO_CRF = int(os.environ.get("COMFYUI_VIDEO_CRF", "23"))
COMFYUI_VIDEO_PIX_FMT = os.environ.get("COMFYUI_VIDEO_PIX_FMT", "yuv420p").strip() or "yuv420p"
COMFYUI_LORA_STRENGTH_MODEL = float(os.environ.get("COMFYUI_LORA_STRENGTH_MODEL", "0.9"))
COMFYUI_LORA_STRENGTH_CLIP = float(os.environ.get("COMFYUI_LORA_STRENGTH_CLIP", "1.0"))

COMFYUI_CKPT_NAME = env_or_default("COMFYUI_CKPT_NAME", "v1-5-pruned-emaonly.safetensors")
COMFYUI_MOTION_MODEL_NAME = env_or_default("COMFYUI_MOTION_MODEL_NAME", "mm_sd_v15_v2.ckpt")
COMFYUI_LORA_NAME = env_or_default("COMFYUI_LORA_NAME", "lcm-lora-sdv1-5.safetensors")

SERVER_LOCK = threading.Lock()
SERVER_PROCESS: subprocess.Popen[str] | None = None


@dataclass(frozen=True)
class ComfyVideoJobSpec:
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
    print(f"[runpod-video-worker][comfyui] {message}", flush=True)


def require_comfyui_configuration() -> None:
    missing = []
    if not COMFYUI_CKPT_NAME:
        missing.append("COMFYUI_CKPT_NAME")
    if not COMFYUI_MOTION_MODEL_NAME:
        missing.append("COMFYUI_MOTION_MODEL_NAME")
    if not COMFYUI_LORA_NAME:
        missing.append("COMFYUI_LORA_NAME")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"ComfyUI backend requires these environment variables: {joined}")


def render_video_with_comfyui(job_id: str, job_spec: ComfyVideoJobSpec) -> Path:
    require_comfyui_configuration()
    ensure_comfyui_server()
    prompt_id = queue_prompt(build_animatelcm_prompt(job_id, job_spec))
    history_entry = wait_for_history(prompt_id)
    output_path = resolve_output_path_from_history(history_entry)
    if not output_path.exists():
        raise RuntimeError(f"ComfyUI completed prompt {prompt_id}, but output file was not found at {output_path}")
    log_runtime(f"render_complete prompt_id={prompt_id} output_path={output_path}")
    return output_path


def ensure_comfyui_server() -> None:
    global SERVER_PROCESS

    with SERVER_LOCK:
        if SERVER_PROCESS is not None and SERVER_PROCESS.poll() is None and comfyui_server_ready():
            return

        if not COMFYUI_ROOT.exists():
            raise RuntimeError(f"ComfyUI root does not exist: {COMFYUI_ROOT}")

        command = [
            sys.executable,
            "main.py",
            "--listen",
            COMFYUI_HOST,
            "--port",
            str(COMFYUI_PORT),
            "--disable-auto-launch",
        ]
        if COMFYUI_FORCE_FP16:
            command.append("--force-fp16")

        log_runtime(f"server_start cwd={COMFYUI_ROOT} command={' '.join(command)}")
        SERVER_PROCESS = subprocess.Popen(
            command,
            cwd=str(COMFYUI_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    wait_for_server_ready()


def wait_for_server_ready() -> None:
    started_at = time.monotonic()
    while time.monotonic() - started_at <= COMFYUI_STARTUP_TIMEOUT_SECONDS:
        if SERVER_PROCESS is not None and SERVER_PROCESS.poll() is not None:
            raise RuntimeError(f"ComfyUI server exited before becoming ready with code {SERVER_PROCESS.poll()}")
        if comfyui_server_ready():
            return
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for ComfyUI server after {COMFYUI_STARTUP_TIMEOUT_SECONDS:.0f} seconds")


def comfyui_server_ready() -> bool:
    try:
        request_json("GET", f"http://{COMFYUI_HOST}:{COMFYUI_PORT}/history")
        return True
    except Exception:
        return False


def queue_prompt(prompt: dict[str, Any]) -> str:
    client_id = uuid.uuid4().hex
    payload = {"prompt": prompt, "client_id": client_id}
    response = request_json("POST", f"http://{COMFYUI_HOST}:{COMFYUI_PORT}/prompt", payload)
    prompt_id = str(response.get("prompt_id") or "").strip()
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return a prompt_id. Response: {response}")
    node_errors = response.get("node_errors")
    if node_errors:
        raise RuntimeError(f"ComfyUI prompt validation failed: {node_errors}")
    log_runtime(f"prompt_queued prompt_id={prompt_id}")
    return prompt_id


def extract_status_string(payload: Any) -> str:
    if isinstance(payload, dict):
        status = payload.get("status")
        if isinstance(status, dict):
            return str(status.get("status_str") or status.get("status") or "").lower()
        if isinstance(status, str):
            return status.lower()
        return str(payload.get("status_str") or "").lower()
    if isinstance(payload, str):
        return payload.lower()
    return ""


def wait_for_history(prompt_id: str) -> dict[str, Any]:
    started_at = time.monotonic()
    history_url = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}/history/{urllib.parse.quote(prompt_id)}"
    job_url = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}/api/jobs/{urllib.parse.quote(prompt_id)}"

    while time.monotonic() - started_at <= COMFYUI_JOB_TIMEOUT_SECONDS:
        history = request_json("GET", history_url)
        entry = history.get(prompt_id)
        if entry:
            status = extract_status_string(entry)
            if status == "error":
                raise RuntimeError(f"ComfyUI prompt {prompt_id} failed: {json.dumps(entry, ensure_ascii=True)}")
            return entry

        try:
            job_info = request_json("GET", job_url)
            status = extract_status_string(job_info)
            if status == "error":
                raise RuntimeError(f"ComfyUI prompt {prompt_id} failed while queued: {json.dumps(job_info, ensure_ascii=True)}")
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise

        time.sleep(COMFYUI_POLL_INTERVAL_SECONDS)

    raise TimeoutError(f"Timed out waiting for ComfyUI prompt {prompt_id} after {COMFYUI_JOB_TIMEOUT_SECONDS:.0f} seconds")


def resolve_output_path_from_history(history_entry: dict[str, Any]) -> Path:
    outputs = history_entry.get("outputs") or {}
    for node_output in outputs.values():
        gifs = node_output.get("gifs") or []
        for gif in gifs:
            fullpath = gif.get("fullpath")
            if isinstance(fullpath, str) and fullpath.strip():
                return Path(fullpath)
            filename = gif.get("filename")
            subfolder = gif.get("subfolder") or ""
            output_type = gif.get("type") or "output"
            if isinstance(filename, str) and filename.strip():
                base_dir = COMFYUI_ROOT / ("temp" if output_type == "temp" else "output")
                return (base_dir / subfolder / filename).resolve()
    raise RuntimeError(f"ComfyUI history did not include a video output: {json.dumps(history_entry, ensure_ascii=True)}")


def build_animatelcm_prompt(job_id: str, job_spec: ComfyVideoJobSpec) -> dict[str, Any]:
    filename_prefix = f"runpod/{job_id}"
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": COMFYUI_CKPT_NAME,
            },
        },
        "2": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": COMFYUI_LORA_NAME,
                "strength_model": COMFYUI_LORA_STRENGTH_MODEL,
                "strength_clip": COMFYUI_LORA_STRENGTH_CLIP,
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 1],
                "text": job_spec.prompt,
            },
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 1],
                "text": job_spec.negative_prompt,
            },
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": job_spec.native_width,
                "height": job_spec.native_height,
                "batch_size": job_spec.frames,
            },
        },
        "6": {
            "class_type": "ADE_AnimateDiffLoaderGen1",
            "inputs": {
                "model": ["2", 0],
                "model_name": COMFYUI_MOTION_MODEL_NAME,
                "beta_schedule": "autoselect",
            },
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "seed": job_spec.seed,
                "steps": job_spec.steps,
                "cfg": job_spec.guidance_scale,
                "sampler_name": "lcm",
                "scheduler": "normal",
                "denoise": 1,
                "model": ["6", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["7", 0],
                "vae": ["1", 2],
            },
        },
        "9": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["8", 0],
                "frame_rate": job_spec.fps,
                "loop_count": 0,
                "filename_prefix": filename_prefix,
                "format": COMFYUI_VIDEO_FORMAT,
                "pingpong": False,
                "save_output": True,
                "crf": COMFYUI_VIDEO_CRF,
                "preset": COMFYUI_VIDEO_PRESET,
                "pix_fmt": COMFYUI_VIDEO_PIX_FMT,
                "save_metadata": False,
            },
        },
    }


def request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))