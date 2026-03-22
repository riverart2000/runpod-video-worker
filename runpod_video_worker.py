from __future__ import annotations

import copy
import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import boto3
import requests


ROOT_DIR = Path(__file__).resolve().parent
ENV_CANDIDATE_PATHS = (
    ROOT_DIR / ".env",
    ROOT_DIR.parent / "clipflow" / "etc" / ".env",
)
STARTUP_SCRIPT_PATH = ROOT_DIR / "startup.sh"
DEFAULT_WORKFLOW_PATH = ROOT_DIR / "workflows" / "default-video-workflow.json"


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


COMFYUI_DIR = Path(os.environ.get("COMFYUI_DIR", "/comfyui")).resolve()
COMFYUI_OUTPUT_DIR = COMFYUI_DIR / "output"
COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
COMFYUI_BASE_URL = os.environ.get("COMFYUI_BASE_URL", f"http://127.0.0.1:{COMFYUI_PORT}").rstrip("/")
COMFYUI_STARTUP_TIMEOUT_SECONDS = int(os.environ.get("COMFYUI_STARTUP_TIMEOUT_SECONDS", "600"))
COMFYUI_JOB_TIMEOUT_SECONDS = int(os.environ.get("COMFYUI_JOB_TIMEOUT_SECONDS", "3600"))
COMFYUI_POLL_INTERVAL_SECONDS = float(os.environ.get("COMFYUI_POLL_INTERVAL_SECONDS", "2"))
DELETE_LOCAL_OUTPUT_AFTER_UPLOAD = os.environ.get("DELETE_LOCAL_OUTPUT_AFTER_UPLOAD", "true").strip().lower() not in {"0", "false", "no", "off"}

DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted"
DEFAULT_CHECKPOINT_NAME = os.environ.get("DEFAULT_CHECKPOINT_NAME", "sdxl.safetensors")
DEFAULT_MOTION_MODULE = os.environ.get("DEFAULT_MOTION_MODULE", "mm_sd_v15_v2.ckpt")
DEFAULT_VIDEO_WIDTH = int(os.environ.get("DEFAULT_VIDEO_WIDTH", "512"))
DEFAULT_VIDEO_HEIGHT = int(os.environ.get("DEFAULT_VIDEO_HEIGHT", "512"))
DEFAULT_VIDEO_FRAMES = int(os.environ.get("DEFAULT_VIDEO_FRAMES", "24"))
DEFAULT_VIDEO_FPS = int(os.environ.get("DEFAULT_VIDEO_FPS", "6"))
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "20"))
DEFAULT_CFG = float(os.environ.get("DEFAULT_CFG", "7"))
DEFAULT_SAMPLER_NAME = os.environ.get("DEFAULT_SAMPLER_NAME", "euler")
DEFAULT_SCHEDULER = os.environ.get("DEFAULT_SCHEDULER", "normal")
DEFAULT_SEED = int(os.environ.get("DEFAULT_SEED", "12345"))

HTTP_SESSION = requests.Session()
PROCESS_LOCK = threading.Lock()
JOB_LOCK = threading.Lock()
COMFYUI_PROCESS: subprocess.Popen[str] | None = None


def process_runpod_job(event: dict[str, Any]) -> dict[str, Any]:
    job_input = dict(event.get("input") or {})
    job_id = resolve_job_id(event, job_input)

    with JOB_LOCK:
        ensure_comfyui_server()
        workflow, workflow_source = build_workflow(job_input)
        prompt_id = submit_workflow(workflow)
        output_path = wait_for_video_output(prompt_id)
        s3_result = upload_video_to_s3(job_id, output_path)
        cleaned_up = cleanup_local_output(output_path)

    return {
        "status": "COMPLETED",
        "job_id": job_id,
        "prompt_id": prompt_id,
        "workflow_source": workflow_source,
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


def ensure_comfyui_server() -> None:
    global COMFYUI_PROCESS
    with PROCESS_LOCK:
        if COMFYUI_PROCESS is not None and COMFYUI_PROCESS.poll() is None:
            wait_for_comfyui_ready()
            return

        COMFYUI_PROCESS = subprocess.Popen(
            ["/bin/bash", str(STARTUP_SCRIPT_PATH)],
            cwd=str(ROOT_DIR),
            env=os.environ.copy(),
        )
        wait_for_comfyui_ready()


def wait_for_comfyui_ready() -> None:
    deadline = time.time() + COMFYUI_STARTUP_TIMEOUT_SECONDS
    while time.time() < deadline:
        if COMFYUI_PROCESS is not None and COMFYUI_PROCESS.poll() is not None:
            raise RuntimeError(f"ComfyUI exited during startup with code {COMFYUI_PROCESS.returncode}.")
        try:
            response = HTTP_SESSION.get(f"{COMFYUI_BASE_URL}/system_stats", timeout=5)
            if response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for ComfyUI to start after {COMFYUI_STARTUP_TIMEOUT_SECONDS} seconds.")


def build_workflow(job_input: dict[str, Any]) -> tuple[dict[str, Any], str]:
    workflow_payload = job_input.get("workflow")
    workflow_source = "default"
    if workflow_payload is None:
        loaded = json.loads(DEFAULT_WORKFLOW_PATH.read_text(encoding="utf-8"))
    else:
        workflow_source = "request"
        if isinstance(workflow_payload, str):
            workflow_payload = json.loads(workflow_payload)
        if not isinstance(workflow_payload, dict):
            raise TypeError("input.workflow must be a JSON object or JSON string.")
        loaded = copy.deepcopy(workflow_payload)

    workflow = loaded.get("input", {}).get("workflow") if "input" in loaded else loaded
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("Workflow payload must contain a non-empty workflow object.")

    replacements = build_template_replacements(job_input)
    resolved_workflow = substitute_placeholders(copy.deepcopy(workflow), replacements)
    return resolved_workflow, workflow_source


def build_template_replacements(job_input: dict[str, Any]) -> dict[str, Any]:
    prompt = str(job_input.get("prompt") or job_input.get("video_prompt") or "").strip()
    if not prompt:
        raise ValueError("input.prompt or input.video_prompt is required.")

    width = to_int(job_input.get("width"), DEFAULT_VIDEO_WIDTH)
    height = to_int(job_input.get("height"), DEFAULT_VIDEO_HEIGHT)
    frames = to_int(job_input.get("frames"), DEFAULT_VIDEO_FRAMES)
    fps = to_int(job_input.get("fps"), DEFAULT_VIDEO_FPS)
    steps = to_int(job_input.get("steps"), DEFAULT_STEPS)
    seed = to_int(job_input.get("seed"), DEFAULT_SEED)
    cfg = to_float(job_input.get("cfg"), DEFAULT_CFG)

    return {
        "__VIDEO_PROMPT__": prompt,
        "__NEGATIVE_PROMPT__": str(job_input.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT),
        "__VIDEO_WIDTH__": width,
        "__VIDEO_HEIGHT__": height,
        "__VIDEO_FRAMES__": frames,
        "__VIDEO_FPS__": fps,
        "__STEPS__": steps,
        "__SEED__": seed,
        "__CFG__": cfg,
        "__SAMPLER_NAME__": str(job_input.get("sampler_name") or DEFAULT_SAMPLER_NAME),
        "__SCHEDULER__": str(job_input.get("scheduler") or DEFAULT_SCHEDULER),
        "__CHECKPOINT_NAME__": str(job_input.get("checkpoint_name") or DEFAULT_CHECKPOINT_NAME),
        "__MOTION_MODULE__": str(job_input.get("motion_module") or DEFAULT_MOTION_MODULE),
    }


def substitute_placeholders(value: Any, replacements: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: substitute_placeholders(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [substitute_placeholders(item, replacements) for item in value]
    if isinstance(value, str):
        if value in replacements:
            return replacements[value]
        resolved = value
        for placeholder, replacement in replacements.items():
            if placeholder in resolved:
                resolved = resolved.replace(placeholder, str(replacement))
        return resolved
    return value


def submit_workflow(workflow: dict[str, Any]) -> str:
    response = HTTP_SESSION.post(
        f"{COMFYUI_BASE_URL}/prompt",
        json={"prompt": workflow, "client_id": uuid.uuid4().hex},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    prompt_id = str(payload.get("prompt_id") or "").strip()
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return prompt_id: {payload}")
    return prompt_id


def wait_for_video_output(prompt_id: str) -> Path:
    deadline = time.time() + COMFYUI_JOB_TIMEOUT_SECONDS
    while time.time() < deadline:
        history = fetch_prompt_history(prompt_id)
        file_refs = collect_output_file_references(history)
        if file_refs:
            output_path = resolve_output_file(file_refs)
            if output_path is not None and output_path.exists():
                return output_path
        time.sleep(COMFYUI_POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"Timed out waiting for ComfyUI job {prompt_id} after {COMFYUI_JOB_TIMEOUT_SECONDS} seconds.")


def fetch_prompt_history(prompt_id: str) -> dict[str, Any]:
    response = HTTP_SESSION.get(f"{COMFYUI_BASE_URL}/history/{prompt_id}", timeout=30)
    response.raise_for_status()
    payload = response.json()
    history = payload.get(prompt_id)
    if isinstance(history, dict):
        return history
    return {}


def collect_output_file_references(history: dict[str, Any]) -> list[dict[str, str]]:
    outputs = history.get("outputs")
    if not isinstance(outputs, dict):
        return []

    collected: list[dict[str, str]] = []
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for key in ("gifs", "videos", "images", "files"):
            items = node_output.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                filename = str(item.get("filename") or "").strip()
                if not filename:
                    continue
                collected.append({
                    "filename": filename,
                    "subfolder": str(item.get("subfolder") or "").strip(),
                    "type": str(item.get("type") or "output").strip() or "output",
                })
    return collected


def resolve_output_file(file_refs: list[dict[str, str]]) -> Path | None:
    preferred = sorted(
        file_refs,
        key=lambda item: (0 if item["filename"].lower().endswith(".mp4") else 1, item["filename"]),
    )
    for ref in preferred:
        base_dir = COMFYUI_DIR / ref["type"]
        candidate = (base_dir / ref["subfolder"] / ref["filename"]).resolve()
        if candidate.exists():
            return candidate
    return None


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
    if not DELETE_LOCAL_OUTPUT_AFTER_UPLOAD:
        return False
    try:
        output_path.unlink(missing_ok=True)
        return True
    except OSError:
        return False


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