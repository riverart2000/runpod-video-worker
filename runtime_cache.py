from __future__ import annotations

import os
import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_CACHE_DIR = Path("/opt/models/hf-cache")
RUNPOD_VOLUME_CACHE_DIR = Path("/runpod-volume/hf-cache")
RUNPOD_VOLUME_STATE_DIR = Path("/runpod-volume/runpod-worker-state")
RUNPOD_VOLUME_TMP_DIR = Path("/runpod-volume/tmp")


def resolve_runtime_cache_dir() -> Path:
    configured = os.environ.get("MODEL_CACHE_DIR", "").strip()
    configured_path = Path(configured) if configured else None

    if RUNPOD_VOLUME_CACHE_DIR.parent.exists() and (
        configured_path is None or configured_path.resolve() == DEFAULT_IMAGE_CACHE_DIR.resolve()
    ):
        cache_dir = RUNPOD_VOLUME_CACHE_DIR
    elif configured_path is not None:
        cache_dir = configured_path
    else:
        cache_dir = ROOT_DIR / "models_cache"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def bootstrap_huggingface_cache_env() -> Path:
    cache_dir = resolve_runtime_cache_dir()
    os.environ["MODEL_CACHE_DIR"] = str(cache_dir)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["DIFFUSERS_CACHE"] = str(cache_dir)
    return cache_dir


def resolve_runtime_state_dir() -> Path:
    if RUNPOD_VOLUME_STATE_DIR.parent.exists():
        state_dir = RUNPOD_VOLUME_STATE_DIR
    else:
        state_dir = Path("/tmp/runpod-worker-state")

    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def resolve_runtime_tmp_dir() -> Path:
    if RUNPOD_VOLUME_TMP_DIR.parent.exists():
        tmp_dir = RUNPOD_VOLUME_TMP_DIR
    else:
        tmp_dir = Path("/tmp")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def bootstrap_runtime_dirs() -> Path:
    state_dir = resolve_runtime_state_dir()
    tmp_dir = resolve_runtime_tmp_dir()
    os.environ.setdefault("TMPDIR", str(tmp_dir))
    os.chdir(state_dir)
    return state_dir


def ensure_cache_has_free_space(cache_dir: Path, minimum_free_gb: float, context: str = "Hugging Face model download") -> None:
    try:
        usage = shutil.disk_usage(cache_dir)
    except OSError:
        return

    required_bytes = int(minimum_free_gb * 1024 * 1024 * 1024)
    if usage.free >= required_bytes:
        return

    free_gb = usage.free / (1024 * 1024 * 1024)
    raise RuntimeError(
        f"Insufficient disk space for {context}. "
        f"Cache path '{cache_dir}' has only {free_gb:.2f} GB free, but at least {minimum_free_gb:.0f} GB is required. "
        "Mount a larger /runpod-volume, increase the container disk, or set MODEL_CACHE_DIR to a larger path."
    )