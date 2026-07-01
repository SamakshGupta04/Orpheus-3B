# Voice Cloning Dashboard — full GPU image.
#
# Bakes in the same layout as setup_vm.sh: a base Streamlit env plus three
# isolated venvs (one per model, matching torch/transformer versions that
# conflict with each other) and one for the heavy metrics stack. Each model
# runs as its own warm HTTP server (infer_model{1,2,3}.py --serve), started
# from the app's "Model Servers" panel — this image just makes sure the
# right interpreter, ports, and deps exist for each.
#
# Build (adjust CUDA_VERSION build-arg to match `nvidia-smi` on the host —
# cu118 | cu121 | cu124):
#   docker build --build-arg CUDA_VERSION=cu121 -t voice-cloning:full .
#
# Run (needs nvidia-container-toolkit on the host):
#   docker run --gpus all -p 8501:8501 \
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
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        build-essential ffmpeg libsndfile1 git curl tini \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m ensurepip --upgrade

WORKDIR /app

# ── 0. Base app deps (system Python — the container itself is the isolation) ──
RUN python3.12 -m pip install --upgrade pip && \
    python3.12 -m pip install "streamlit>=1.31" librosa soundfile numpy

# ── 1. Model 1 — Orpheus 3B (unsloth + SNAC) ─────────────────────────────────
RUN python3.12 -m venv /app/.venv-model1 && \
    /app/.venv-model1/bin/pip install --upgrade pip && \
    /app/.venv-model1/bin/pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/${CUDA_VERSION} && \
    /app/.venv-model1/bin/pip install "unsloth[colab-new]" snac librosa soundfile numpy

# ── 2. Model 2 — VoxCPM2 2B ───────────────────────────────────────────────────
RUN python3.12 -m venv /app/.venv-model2 && \
    /app/.venv-model2/bin/pip install --upgrade pip && \
    /app/.venv-model2/bin/pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/${CUDA_VERSION} && \
    /app/.venv-model2/bin/pip install voxcpm librosa soundfile numpy

# ── 3. Model 3 — VibeVoice Hindi 1.5B (not on PyPI) ──────────────────────────
RUN python3.12 -m venv /app/.venv-model3 && \
    /app/.venv-model3/bin/pip install --upgrade pip && \
    /app/.venv-model3/bin/pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/${CUDA_VERSION} && \
    /app/.venv-model3/bin/pip install \
        "git+https://github.com/vibevoice-community/VibeVoice.git" \
        librosa soundfile numpy

# ── 4. Metrics stack (compute_metrics.py — SECS / WER / UTMOS) ───────────────
COPY requirements-metrics.txt .
RUN python3.12 -m venv /app/.venv-metrics && \
    /app/.venv-metrics/bin/pip install --upgrade pip && \
    /app/.venv-metrics/bin/pip install -r requirements-metrics.txt

# ── 5. App source (copied last so code changes don't invalidate venv layers) ─
COPY . .

ENV MODEL1_PYTHON=/app/.venv-model1/bin/python \
    MODEL2_PYTHON=/app/.venv-model2/bin/python \
    MODEL3_PYTHON=/app/.venv-model3/bin/python \
    METRICS_PYTHON=/app/.venv-metrics/bin/python \
    MODEL1_PORT=8001 \
    MODEL2_PORT=8002 \
    MODEL3_PORT=8003 \
    AUTOSTART_MODELS=all

# 8001-8003 are consumed internally over 127.0.0.1 by app.py — not published
# by docker-compose, only documented here for anyone attaching a debugger.
EXPOSE 8501 8001 8002 8003

# tini reaps the detached model-server subprocesses app.py spawns (it isn't
# their parent by the time they're orphaned on restart/crash) since streamlit
# runs as PID 1 otherwise.
ENTRYPOINT ["tini", "--"]
CMD ["python3.12", "-m", "streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
