FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_XET=1

WORKDIR /worker

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

COPY requirements.txt /tmp/runpod-worker-requirements.txt
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 && \
    python -m pip install -r /tmp/runpod-worker-requirements.txt && \
    rm /tmp/runpod-worker-requirements.txt

COPY . /worker

CMD ["python", "handler.py"]