FROM runpod/worker-comfyui:5.8.5-base

ENV COMFYUI_DIR=/comfyui \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /

RUN test -d "$COMFYUI_DIR"

RUN pip install --upgrade pip

RUN cd "$COMFYUI_DIR/custom_nodes" && \
    git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git && \
    git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git

RUN pip install -r "$COMFYUI_DIR/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt"

COPY requirements.txt /tmp/runpod-worker-requirements.txt
RUN pip install -r /tmp/runpod-worker-requirements.txt && rm /tmp/runpod-worker-requirements.txt

COPY . /worker

RUN chmod +x /worker/startup.sh

WORKDIR /worker

CMD ["python", "handler.py"]