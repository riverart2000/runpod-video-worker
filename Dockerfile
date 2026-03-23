# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm

ARG COMFYUI_REF=master
ARG COMFYUI_ANIMATEDIFF_REF=main
ARG COMFYUI_VHS_REF=main
ARG PRELOAD_DIFFUSERS_MODELS=true
ARG PRELOAD_COMFYUI_MODELS=true
ARG COMFYUI_CKPT_NAME=
ARG COMFYUI_MOTION_MODEL_NAME=
ARG COMFYUI_LORA_NAME=
ARG COMFYUI_CKPT_SOURCE_REPO=
ARG COMFYUI_CKPT_SOURCE_FILENAME=
ARG COMFYUI_MOTION_MODEL_SOURCE_REPO=
ARG COMFYUI_MOTION_MODEL_SOURCE_FILENAME=
ARG COMFYUI_LORA_SOURCE_REPO=
ARG COMFYUI_LORA_SOURCE_FILENAME=

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_XET=1 \
    COMFYUI_ROOT=/opt/ComfyUI \
    MODEL_CACHE_DIR=/opt/models/hf-cache \
    HF_HOME=/opt/models/hf-cache \
    HUGGINGFACE_HUB_CACHE=/opt/models/hf-cache \
    TRANSFORMERS_CACHE=/opt/models/hf-cache \
    DIFFUSERS_CACHE=/opt/models/hf-cache \
    PRELOAD_DIFFUSERS_MODELS=${PRELOAD_DIFFUSERS_MODELS} \
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
RUN python -m pip install torch==2.5.1 && \
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
    python -m compileall -q -j 0 /worker /opt/ComfyUI /usr/local/lib/python3.11/site-packages

CMD ["python", "handler.py"]