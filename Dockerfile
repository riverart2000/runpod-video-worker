# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm

ARG COMFYUI_REF=master
ARG COMFYUI_ANIMATEDIFF_REF=main
ARG COMFYUI_VHS_REF=main
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG PRELOAD_DIFFUSERS_MODELS=false
ARG PRELOAD_WAN_MODELS=false
ARG PRELOAD_COMFYUI_MODELS=false
ARG COMFYUI_CKPT_NAME=v1-5-pruned-emaonly.safetensors
ARG COMFYUI_MOTION_MODEL_NAME=mm_sd_v15_v2.ckpt
ARG COMFYUI_LORA_NAME=lcm-lora-sdv1-5.safetensors
ARG COMFYUI_CKPT_SOURCE_REPO=Syimbiote/v1-5-pruned-emaonly
ARG COMFYUI_CKPT_SOURCE_FILENAME=v1-5-pruned-emaonly.safetensors
ARG COMFYUI_MOTION_MODEL_SOURCE_REPO=guoyww/animatediff
ARG COMFYUI_MOTION_MODEL_SOURCE_FILENAME=mm_sd_v15_v2.ckpt
ARG COMFYUI_LORA_SOURCE_REPO=latent-consistency/lcm-lora-sdv1-5
ARG COMFYUI_LORA_SOURCE_FILENAME=pytorch_lora_weights.safetensors

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_XET=1 \
    WORKER_BACKEND=comfyui \
    VIDEO_BACKEND=comfyui \
    RUNPOD_S3_URL=https://revenuemindproai.s3.eu-west-1.amazonaws.com/runpod_videos/ \
    AWS_REGION=eu-west-1 \
    DEFAULT_NATIVE_WIDTH=768 \
    DEFAULT_NATIVE_HEIGHT=1344 \
    DEFAULT_VIDEO_FRAMES=61 \
    MAX_VIDEO_FRAMES=81 \
    DEFAULT_VIDEO_FPS=12 \
    DEFAULT_STEPS=20 \
    MAX_STEPS=28 \
    DEFAULT_GUIDANCE_SCALE=3.0 \
    DEFAULT_LORA_SCALE=0.9 \
    DEFAULT_DECODE_CHUNK_SIZE=12 \
    WAN_MODEL_ID=Wan-AI/Wan2.1-T2V-14B-Diffusers \
    WAN_FLOW_SHIFT=5.0 \
    WAN_ENABLE_VAE_TILING=true \
    WAN_ENABLE_VAE_SLICING=false \
    WAN_USE_CPU_OFFLOAD=false \
    COMFYUI_ROOT=/opt/ComfyUI \
    COMFYUI_HOST=127.0.0.1 \
    COMFYUI_PORT=8188 \
    COMFYUI_FORCE_FP16=true \
    COMFYUI_STARTUP_TIMEOUT_SECONDS=240 \
    COMFYUI_JOB_TIMEOUT_SECONDS=2400 \
    COMFYUI_POLL_INTERVAL_SECONDS=2 \
    COMFYUI_VIDEO_FORMAT=video/runpod_h264_mp4 \
    COMFYUI_VIDEO_PRESET=slow \
    COMFYUI_VIDEO_CRF=18 \
    COMFYUI_VIDEO_PIX_FMT=yuv420p \
    COMFYUI_LORA_STRENGTH_MODEL=0.85 \
    COMFYUI_LORA_STRENGTH_CLIP=0.95 \
    MODEL_CACHE_DIR=/opt/models/hf-cache \
    HF_HOME=/opt/models/hf-cache \
    HUGGINGFACE_HUB_CACHE=/opt/models/hf-cache \
    TRANSFORMERS_CACHE=/opt/models/hf-cache \
    DIFFUSERS_CACHE=/opt/models/hf-cache \
    PRELOAD_DIFFUSERS_MODELS=${PRELOAD_DIFFUSERS_MODELS} \
    PRELOAD_WAN_MODELS=${PRELOAD_WAN_MODELS} \
    PRELOAD_COMFYUI_MODELS=${PRELOAD_COMFYUI_MODELS} \
    HF_TOKEN_FILE=/run/secrets/hf_token \
    COMFYUI_CKPT_NAME=${COMFYUI_CKPT_NAME} \
    COMFYUI_MOTION_MODEL_NAME=${COMFYUI_MOTION_MODEL_NAME} \
    COMFYUI_LORA_NAME=${COMFYUI_LORA_NAME} \
    COMFYUI_CKPT_SOURCE_REPO=${COMFYUI_CKPT_SOURCE_REPO} \
    COMFYUI_CKPT_SOURCE_FILENAME=${COMFYUI_CKPT_SOURCE_FILENAME} \
    COMFYUI_MOTION_MODEL_SOURCE_REPO=${COMFYUI_MOTION_MODEL_SOURCE_REPO} \
    COMFYUI_MOTION_MODEL_SOURCE_FILENAME=${COMFYUI_MOTION_MODEL_SOURCE_FILENAME} \
    COMFYUI_LORA_SOURCE_REPO=${COMFYUI_LORA_SOURCE_REPO} \
    COMFYUI_LORA_SOURCE_FILENAME=${COMFYUI_LORA_SOURCE_FILENAME}

WORKDIR /worker

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

COPY requirements.txt /tmp/runpod-worker-requirements.txt
RUN python -m pip install --index-url ${PYTORCH_INDEX_URL} torch==2.5.1 && \
    python -m pip install -r /tmp/runpod-worker-requirements.txt && \
    rm /tmp/runpod-worker-requirements.txt

RUN git clone --depth 1 --branch ${COMFYUI_REF} https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI && \
    git clone --depth 1 --branch ${COMFYUI_ANIMATEDIFF_REF} https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git /opt/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved && \
    git clone --depth 1 --branch ${COMFYUI_VHS_REF} https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && \
    python -m pip install -r /opt/ComfyUI/requirements.txt && \
    if [ -f /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt ]; then python -m pip install -r /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt; fi

COPY . /worker
RUN --mount=type=secret,id=hf_token,required=false \
    mkdir -p /opt/models/hf-cache \
    /opt/ComfyUI/models/checkpoints \
    /opt/ComfyUI/models/animatediff_models \
    /opt/ComfyUI/models/loras \
    /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/video_formats && \
    cp /worker/workflows/comfyui_video_formats/runpod_h264_mp4.json /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/video_formats/runpod_h264_mp4.json && \
    python /worker/bin/preload_runtime_assets.py && \
    python -m compileall -q -j 0 /worker /opt/ComfyUI

CMD ["python", "handler.py"]