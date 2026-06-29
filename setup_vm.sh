#!/usr/bin/env bash
# setup_vm.sh — Install all three voice-cloning model stacks on a GPU VM.
#
# Usage:
#   chmod +x setup_vm.sh
#   ./setup_vm.sh
#
# What it does:
#   1. Installs the Streamlit app's base deps (streamlit, librosa, soundfile).
#   2. Creates three separate Python venvs (.venv-model1/2/3) to avoid
#      transformer-version conflicts between the models.
#   3. Writes a helper script (activate_models.sh) that exports MODEL1/2/3_PYTHON
#      so the app knows which interpreter to use per model.
#
# Adjust CUDA_VERSION below to match your driver (check: nvidia-smi).

set -euo pipefail

CUDA_VERSION="cu121"          # cu118 | cu121 | cu124 — match your driver
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_VERSION}"
PYTHON="${PYTHON:-python3}"   # override with: PYTHON=/usr/bin/python3.11 ./setup_vm.sh

echo "=== Voice Cloning VM Setup ==="
echo "CUDA version : ${CUDA_VERSION}"
echo "Python       : $(${PYTHON} --version)"
echo ""

# ── 0. Base app deps (in the current env / system Python) ────────────────────
echo "[0/4] Installing base app dependencies..."
${PYTHON} -m pip install --quiet --upgrade pip
${PYTHON} -m pip install --quiet \
    "streamlit>=1.31" \
    librosa \
    soundfile \
    numpy

echo "      Base deps installed."
echo ""

# ── 1. Model 1 — Orpheus 3B (unsloth + SNAC) ────────────────────────────────
echo "[1/4] Setting up Model 1 venv (.venv-model1) — Orpheus 3B..."
${PYTHON} -m venv .venv-model1
.venv-model1/bin/pip install --quiet --upgrade pip

# unsloth must be installed AFTER torch (it detects the CUDA version at install time)
.venv-model1/bin/pip install --quiet \
    torch torchvision torchaudio \
    --index-url "${TORCH_INDEX}"

.venv-model1/bin/pip install --quiet \
    "unsloth[colab-new]" \
    snac \
    librosa \
    soundfile \
    numpy

echo "      Model 1 venv ready."
echo ""

# ── 2. Model 2 — VoxCPM2 2B ──────────────────────────────────────────────────
echo "[2/4] Setting up Model 2 venv (.venv-model2) — VoxCPM2..."
${PYTHON} -m venv .venv-model2
.venv-model2/bin/pip install --quiet --upgrade pip

.venv-model2/bin/pip install --quiet \
    torch torchvision torchaudio \
    --index-url "${TORCH_INDEX}"

.venv-model2/bin/pip install --quiet \
    voxcpm \
    librosa \
    soundfile \
    numpy

echo "      Model 2 venv ready."
echo ""

# ── 3. Model 3 — VibeVoice Hindi 1.5B (not on PyPI) ─────────────────────────
echo "[3/4] Setting up Model 3 venv (.venv-model3) — VibeVoice Hindi..."
${PYTHON} -m venv .venv-model3
.venv-model3/bin/pip install --quiet --upgrade pip

.venv-model3/bin/pip install --quiet \
    torch torchvision torchaudio \
    --index-url "${TORCH_INDEX}"

.venv-model3/bin/pip install --quiet \
    "git+https://github.com/vibevoice-community/VibeVoice.git" \
    librosa \
    soundfile \
    numpy

echo "      Model 3 venv ready."
echo ""

# ── 4. Write the env-var helper ──────────────────────────────────────────────
echo "[4/4] Writing activate_models.sh..."
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

cat > activate_models.sh <<EOF
#!/usr/bin/env bash
# Source this file before running the Streamlit app to point each model
# at its own venv:
#   source activate_models.sh
#   streamlit run app.py
export MODEL1_PYTHON="${REPO_DIR}/.venv-model1/bin/python"
export MODEL2_PYTHON="${REPO_DIR}/.venv-model2/bin/python"
export MODEL3_PYTHON="${REPO_DIR}/.venv-model3/bin/python"
echo "Model interpreters set:"
echo "  MODEL1_PYTHON=\${MODEL1_PYTHON}"
echo "  MODEL2_PYTHON=\${MODEL2_PYTHON}"
echo "  MODEL3_PYTHON=\${MODEL3_PYTHON}"
EOF

chmod +x activate_models.sh
echo "      activate_models.sh written."
echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
echo "=== Setup complete ==="
echo ""
echo "Quick-smoke-test each inference script:"
echo "  .venv-model1/bin/python infer_model1.py --ref-audio outputs/sample_reference_audio.mp3 --target-text 'test' --output /tmp/m1.wav --dry-run"
echo "  .venv-model2/bin/python infer_model2.py --ref-audio outputs/sample_reference_audio.mp3 --target-text 'test' --output /tmp/m2.wav --dry-run"
echo "  .venv-model3/bin/python infer_model3.py --ref-audio outputs/sample_reference_audio.mp3 --target-text 'test' --output /tmp/m3.wav --dry-run"
echo ""
echo "Then start the app:"
echo "  source activate_models.sh"
echo "  streamlit run app.py"
