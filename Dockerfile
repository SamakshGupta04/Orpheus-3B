# Voice Cloning Dashboard — GPU image.
#
# Bakes in the same layout as setup_vm.sh: a base Streamlit env plus one
# isolated venv per model (torch/transformer versions conflict between them)
# and one for the heavy metrics stack. Each model runs as its own warm HTTP
# server (infer_model{1,2,3}.py --serve), started from the app's "Model
# Servers" panel — this image just makes sure the right interpreter, ports,
# and deps exist for each.
#
# Model 1 (Orpheus) and Model 3 (VibeVoice) are currently DISABLED — their
# venv-build RUN blocks are commented out below — because VM disk space is
# tight. Only Model 2 (VoxCPM2) installs. DISABLED_MODELS=orpheus,vibevoice
# (set near the bottom) tells app.py to grey those out in the UI instead of
# ever trying to start them. To bring one back: uncomment its RUN block,
# remove it from DISABLED_MODELS, and rebuild.
#
# Only ONE port is published: 8512, an in-container nginx gateway (see
# nginx.conf.template + docker-entrypoint.sh) that routes "/" to Streamlit
# and "/api/<orpheus|voxcpm2|vibevoice>/..." to that model's own HTTP API
# (proxying to 127.0.0.1:8001/8002/8003, which stay unpublished). The /api/*
# routes require an "X-API-Key" header matching the API_KEY env var — the
# container refuses to start without it set. "/" (the dashboard) needs no
# key; app.py's own calls to the model servers bypass nginx entirely (same
# container, loopback) so they don't need one either.
#
# Build (adjust CUDA_VERSION build-arg to match `nvidia-smi` on the host —
# cu118 | cu121 | cu124):
#   docker build --build-arg CUDA_VERSION=cu121 -t voice-cloning:full .
#
# Run (needs nvidia-container-toolkit on the host):
#   docker run --gpus all -p 8512:8512 -e API_KEY=$(openssl rand -hex 32) \
#       -v $(pwd)/outputs:/app/outputs \
#       -v hf-cache:/root/.cache/huggingface \
#       voice-cloning:full
#
# Model weights are NOT baked into the image — they download from
# HuggingFace on first use of each model and are cached under
# /root/.cache/huggingface, so mount that as a volume or they re-download
# every container recreation.
#
# Uses the CUDA "devel" base (not "runtime") because unsloth's Triton JIT
# and bitsandbytes' 4-bit kernels are unverified against a runtime-only
# image on this GPU/driver combo — devel guarantees nvcc/headers are
# present. If you confirm "runtime" works on your driver, switching saves
# several GB.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG CUDA_VERSION=cu121
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── System deps + Python 3.12 (matches .python-version; Ubuntu 22.04 ships 3.10) ──
# nginx-light (not full nginx): just a reverse proxy + WebSocket upgrade for
# Streamlit here, none of the extra modules (image-filter, xslt, geoip...)
# the full package pulls in — disk space on this VM is already tight.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        build-essential ffmpeg libsndfile1 git curl tini nginx-light \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m ensurepip --upgrade

WORKDIR /app

# ── 0. Base app deps ──────────────────────────────────────────────────────────
# In its own venv, not system python3.12: software-properties-common (needed
# above for the deadsnakes PPA) drags in an apt-managed python3-blinker, and
# pip refuses to upgrade a distutils-installed package it can't safely
# uninstall. A venv sidesteps that entirely instead of pip-flag-fighting it.
RUN python3.12 -m venv /app/.venv-app && \
    /app/.venv-app/bin/pip install --upgrade pip && \
    /app/.venv-app/bin/pip install "streamlit>=1.31" librosa soundfile numpy

# ── 1. Model 1 — Orpheus 3B (unsloth + SNAC) ─────────────────────────────────
# DISABLED — disk space on the VM is tight. To re-enable: uncomment this block
# AND drop "orpheus" from DISABLED_MODELS below.
# RUN python3.12 -m venv /app/.venv-model1 && \
#     /app/.venv-model1/bin/pip install --upgrade pip && \
#     /app/.venv-model1/bin/pip install torch torchvision torchaudio \
#         --index-url https://download.pytorch.org/whl/${CUDA_VERSION} && \
#     /app/.venv-model1/bin/pip install "unsloth[colab-new]" snac librosa soundfile numpy

# ── 2. Model 2 — VoxCPM2 2B ───────────────────────────────────────────────────
RUN python3.12 -m venv /app/.venv-model2 && \
    /app/.venv-model2/bin/pip install --upgrade pip && \
    /app/.venv-model2/bin/pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/${CUDA_VERSION} && \
    /app/.venv-model2/bin/pip install voxcpm librosa soundfile numpy

# ── 3. Model 3 — VibeVoice Hindi 1.5B (not on PyPI) ──────────────────────────
# DISABLED — disk space on the VM is tight. To re-enable: uncomment this block
# AND drop "vibevoice" from DISABLED_MODELS below.
# RUN python3.12 -m venv /app/.venv-model3 && \
#     /app/.venv-model3/bin/pip install --upgrade pip && \
#     /app/.venv-model3/bin/pip install torch torchvision torchaudio \
#         --index-url https://download.pytorch.org/whl/${CUDA_VERSION} && \
#     /app/.venv-model3/bin/pip install \
#         "git+https://github.com/vibevoice-community/VibeVoice.git" \
#         librosa soundfile numpy

# ── 4. Metrics stack (compute_metrics.py — SECS / WER / UTMOS) ───────────────
COPY requirements-metrics.txt .
RUN python3.12 -m venv /app/.venv-metrics && \
    /app/.venv-metrics/bin/pip install --upgrade pip && \
    /app/.venv-metrics/bin/pip install -r requirements-metrics.txt

# ── 5. App source (copied last so code changes don't invalidate venv layers) ─
COPY . .

RUN chmod +x /app/docker-entrypoint.sh \
    && rm -f /etc/nginx/sites-enabled/default

# MODEL1_PYTHON / MODEL3_PYTHON intentionally unset — .venv-model1/3 aren't
# built (see the commented-out RUN blocks above). app.py checks
# DISABLED_MODELS explicitly before ever touching those interpreter paths, so
# leaving them unset here is enough; it never falls through to a broken path.
#
# API_KEY has NO default — docker-entrypoint.sh refuses to start without one
# set at `docker run`/compose time. Don't bake a real key into the image.
ENV MODEL2_PYTHON=/app/.venv-model2/bin/python \
    METRICS_PYTHON=/app/.venv-metrics/bin/python \
    MODEL1_PORT=8001 \
    MODEL2_PORT=8002 \
    MODEL3_PORT=8003 \
    AUTOSTART_MODELS=all \
    DISABLED_MODELS=orpheus,vibevoice \
    API_KEY=

# Only 8512 (the nginx gateway) is meant to be published. 8501/8001-8003
# stay on 127.0.0.1 inside the container — nginx and app.py reach them over
# loopback; nothing outside the container can reach them directly.
EXPOSE 8512

# tini (PID 1) reaps both nginx's forked worker processes and any detached
# model-server subprocesses app.py spawns. docker-entrypoint.sh starts nginx
# + Streamlit and exits (taking the container down for a clean restart) if
# either one dies.
ENTRYPOINT ["tini", "--", "/app/docker-entrypoint.sh"]
