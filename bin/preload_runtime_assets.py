from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


ROOT_DIR = Path(__file__).resolve().parent.parent


def env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped or default


MODEL_CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", "/opt/models/hf-cache")).resolve()
COMFYUI_ROOT = Path(os.environ.get("COMFYUI_ROOT", "/opt/ComfyUI")).resolve()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
HF_TOKEN_FILE = os.environ.get("HF_TOKEN_FILE", "").strip()

DEFAULT_MODEL_ID = env_or_default("DEFAULT_MODEL_ID", "emilianJR/epiCRealism")
DEFAULT_MOTION_ADAPTER_ID = env_or_default("DEFAULT_MOTION_ADAPTER_ID", "wangfuyun/AnimateLCM")
DEFAULT_LORA_REPOSITORY = env_or_default("DEFAULT_LORA_REPOSITORY", DEFAULT_MOTION_ADAPTER_ID)
DEFAULT_LORA_WEIGHT_NAME = env_or_default("DEFAULT_LORA_WEIGHT_NAME", "AnimateLCM_sd15_t2v_lora.safetensors")
DEFAULT_VAE_ID = env_or_default("DEFAULT_VAE_ID", "stabilityai/sd-vae-ft-mse")
WAN_MODEL_ID = env_or_default("WAN_MODEL_ID", "Wan-AI/Wan2.1-T2V-14B-Diffusers")

COMFYUI_CKPT_NAME = env_or_default("COMFYUI_CKPT_NAME", "v1-5-pruned-emaonly.safetensors")
COMFYUI_MOTION_MODEL_NAME = env_or_default("COMFYUI_MOTION_MODEL_NAME", "mm_sd_v15_v2.ckpt")
COMFYUI_LORA_NAME = env_or_default("COMFYUI_LORA_NAME", "lcm-lora-sdv1-5.safetensors")


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def log(message: str) -> None:
    print(f"[build-preload] {message}", flush=True)


def resolve_hf_token() -> str | None:
    if HF_TOKEN:
        return HF_TOKEN
    if HF_TOKEN_FILE:
        token_path = Path(HF_TOKEN_FILE)
        if token_path.exists():
            token = token_path.read_text(encoding="utf-8").strip()
            if token:
                return token
    return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_directory_contents(source_dir: Path, target_dir: Path) -> None:
    if not source_dir.exists():
        return
    ensure_dir(target_dir)
    for item in source_dir.iterdir():
        destination = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)
    log(f"copied_local_assets source={source_dir} target={target_dir}")


def preload_diffusers_assets() -> None:
    token = resolve_hf_token()
    log(f"preloading_diffusers model={DEFAULT_MODEL_ID} motion_adapter={DEFAULT_MOTION_ADAPTER_ID} vae={DEFAULT_VAE_ID}")
    snapshot_download(
        repo_id=DEFAULT_MODEL_ID,
        cache_dir=str(MODEL_CACHE_DIR),
        token=token,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    snapshot_download(
        repo_id=DEFAULT_MOTION_ADAPTER_ID,
        cache_dir=str(MODEL_CACHE_DIR),
        token=token,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    snapshot_download(
        repo_id=DEFAULT_VAE_ID,
        cache_dir=str(MODEL_CACHE_DIR),
        token=token,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    hf_hub_download(
        repo_id=DEFAULT_LORA_REPOSITORY,
        filename=DEFAULT_LORA_WEIGHT_NAME,
        cache_dir=str(MODEL_CACHE_DIR),
        token=token,
        resume_download=True,
    )


def preload_wan_assets() -> None:
    token = resolve_hf_token()
    log(f"preloading_wan model={WAN_MODEL_ID}")
    snapshot_download(
        repo_id=WAN_MODEL_ID,
        cache_dir=str(MODEL_CACHE_DIR),
        token=token,
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def maybe_download_comfyui_model(target_subdir: str, target_name: str, source_repo: str, source_filename: str) -> None:
    token = resolve_hf_token()
    if not target_name:
        raise RuntimeError(
            f"Missing target filename for ComfyUI {target_subdir}. Set the corresponding COMFYUI_*_NAME value before enabling PRELOAD_COMFYUI_MODELS."
        )
    target_path = COMFYUI_ROOT / "models" / target_subdir / target_name
    if target_path.exists():
        log(f"comfyui_asset_present path={target_path}")
        return
    if not source_repo or not source_filename:
        raise RuntimeError(
            f"Missing ComfyUI model asset for {target_path}. Provide a local file under docker-assets/comfyui-models/{target_subdir}/ or set source repo and filename build args."
        )
    ensure_dir(target_path.parent)
    downloaded_path = Path(
        hf_hub_download(
            repo_id=source_repo,
            filename=source_filename,
            local_dir=str(target_path.parent),
            local_dir_use_symlinks=False,
            token=token,
            resume_download=True,
        )
    )
    if downloaded_path.name != target_name:
        final_path = target_path.parent / target_name
        downloaded_path.replace(final_path)
        downloaded_path = final_path
    log(f"downloaded_comfyui_asset path={downloaded_path}")


def preload_comfyui_assets() -> None:
    if not COMFYUI_CKPT_NAME or not COMFYUI_MOTION_MODEL_NAME or not COMFYUI_LORA_NAME:
        raise RuntimeError(
            "PRELOAD_COMFYUI_MODELS=true requires COMFYUI_CKPT_NAME, COMFYUI_MOTION_MODEL_NAME, and COMFYUI_LORA_NAME so the Docker build can bake the exact files into the image."
        )

    local_models_root = ROOT_DIR / "docker-assets" / "comfyui-models"
    copy_directory_contents(local_models_root / "checkpoints", COMFYUI_ROOT / "models" / "checkpoints")
    copy_directory_contents(local_models_root / "animatediff_models", COMFYUI_ROOT / "models" / "animatediff_models")
    copy_directory_contents(local_models_root / "loras", COMFYUI_ROOT / "models" / "loras")

    maybe_download_comfyui_model(
        "checkpoints",
        COMFYUI_CKPT_NAME,
        env_or_default("COMFYUI_CKPT_SOURCE_REPO", "Syimbiote/v1-5-pruned-emaonly"),
        env_or_default("COMFYUI_CKPT_SOURCE_FILENAME", "v1-5-pruned-emaonly.safetensors"),
    )
    maybe_download_comfyui_model(
        "animatediff_models",
        COMFYUI_MOTION_MODEL_NAME,
        env_or_default("COMFYUI_MOTION_MODEL_SOURCE_REPO", "guoyww/animatediff"),
        env_or_default("COMFYUI_MOTION_MODEL_SOURCE_FILENAME", "mm_sd_v15_v2.ckpt"),
    )
    maybe_download_comfyui_model(
        "loras",
        COMFYUI_LORA_NAME,
        env_or_default("COMFYUI_LORA_SOURCE_REPO", "latent-consistency/lcm-lora-sdv1-5"),
        env_or_default("COMFYUI_LORA_SOURCE_FILENAME", "pytorch_lora_weights.safetensors"),
    )


def main() -> None:
    ensure_dir(MODEL_CACHE_DIR)
    copy_directory_contents(ROOT_DIR / "docker-assets" / "hf-cache", MODEL_CACHE_DIR)

    if env_flag("PRELOAD_DIFFUSERS_MODELS", True):
        preload_diffusers_assets()
    else:
        log("skipping_diffusers_preload")

    if env_flag("PRELOAD_WAN_MODELS", True):
        preload_wan_assets()
    else:
        log("skipping_wan_preload")

    if env_flag("PRELOAD_COMFYUI_MODELS", True):
        preload_comfyui_assets()
    else:
        log("skipping_comfyui_preload")


if __name__ == "__main__":
    main()