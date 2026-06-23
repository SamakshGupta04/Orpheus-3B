"""
🗣️ Voice Cloning Model Comparison Dashboard
=============================================
Streamlit app to compare audio outputs from Orpheus, VoxCPM2, and VibeVoice
across 12 test texts.

Usage:
    1. Run the 3 batch notebooks in Colab → download the zip files
    2. Unzip into an `outputs/` folder so the structure looks like:
         outputs/
           orpheus/   (test_01.wav ... test_12.wav + metadata.json)
           voxcpm2/   (test_01.wav ... test_12.wav + metadata.json)
           vibevoice/ (test_01.wav ... test_12.wav + metadata.json)
    3. Run:  streamlit run app.py
"""

import streamlit as st
import json
import os
import sys
import subprocess
import zipfile
import tempfile
import shutil
from pathlib import Path

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Voice Cloning Comparison",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        color: white !important;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        color: #e0e0ff;
    }

    .test-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .test-card:hover {
        border-color: rgba(102, 126, 234, 0.4);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }

    .test-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 0.8rem;
        flex-wrap: wrap;
    }
    .test-id {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.85rem;
        min-width: 36px;
        text-align: center;
    }
    .badge-lang {
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-english {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .badge-hindi {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-category {
        background: rgba(139, 92, 246, 0.12);
        color: #a78bfa;
        border: 1px solid rgba(139, 92, 246, 0.25);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    .test-text {
        color: #d1d5db;
        font-size: 0.92rem;
        line-height: 1.5;
        margin: 0.6rem 0 1rem 0;
        padding: 0.6rem 1rem;
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #667eea;
        border-radius: 0 8px 8px 0;
    }

    .model-label {
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 0.4rem;
        text-align: center;
    }
    .model-orpheus { color: #f472b6; }
    .model-voxcpm2 { color: #34d399; }
    .model-vibevoice { color: #60a5fa; }

    .gen-time {
        font-size: 0.72rem;
        color: #9ca3af;
        text-align: center;
        margin-top: 0.2rem;
    }

    .status-success {
        color: #34d399;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-error {
        color: #f87171;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-missing {
        color: #6b7280;
        font-size: 0.75rem;
        font-style: italic;
    }

    .reference-card {
        background: linear-gradient(145deg, #1a2332 0%, #1e2d3d 100%);
        border: 1px solid rgba(96, 165, 250, 0.25);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        flex-wrap: wrap;
    }
    .reference-card .ref-icon {
        font-size: 2.2rem;
    }
    .reference-card .ref-info h3 {
        margin: 0;
        font-size: 1.05rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .reference-card .ref-info p {
        margin: 0.2rem 0 0 0;
        font-size: 0.8rem;
        color: #94a3b8;
    }

    .stats-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #222238 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .stats-card h3 {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .stats-card p {
        font-size: 0.8rem;
        color: #9ca3af;
        margin: 0.3rem 0 0 0;
    }

    .sidebar .stSelectbox label,
    .sidebar .stTextInput label {
        font-weight: 600;
        color: #e2e8f0;
    }

    div[data-testid="stExpander"] {
        border: none !important;
        background: transparent !important;
    }

    .metric-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        justify-content: center;
        margin-top: 0.5rem;
    }
    .metric-chip {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 9px;
        border-radius: 999px;
        border: 1px solid;
        background: rgba(255,255,255,0.03);
        white-space: nowrap;
        cursor: help;
    }
    .score-chip {
        font-size: 0.82rem;
        font-weight: 800;
        padding: 3px 12px;
        border-width: 2px;
        background: rgba(255,255,255,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
MODELS = {
    "orpheus": {"name": "Orpheus 3B", "color": "#f472b6", "css_class": "model-orpheus"},
    "voxcpm2": {"name": "VoxCPM2 2B", "color": "#34d399", "css_class": "model-voxcpm2"},
    "vibevoice": {"name": "VibeVoice Hindi 1.5B", "color": "#60a5fa", "css_class": "model-vibevoice"},
}

TEST_TEXTS = [
    {"id": 1, "text": "Your ticket number is B 4 7 2 9 and the fare is rupees three thousand two hundred.", "language": "English", "category": "Booking Confirmation"},
    {"id": 2, "text": "Thank you for calling customer support. Your query has been registered and our team will get back to you within twenty four hours. We apologise for the inconvenience caused.", "language": "English", "category": "Customer Support"},
    {"id": 3, "text": "Departure at 06:45 AM on 3rd February 2025", "language": "English", "category": "Flight Details"},
    {"id": 4, "text": "Aapki booking confirm ho gayi. Reference number note kar lijiye: B 4 9 2 1.", "language": "Hindi", "category": "Booking Confirmation"},
    {"id": 5, "text": "Namaskar aur hamare service mein aapka swagat hai. Aapka loan application approved ho gaya hai. Amount aapke registered account mein do se teen working days mein credit ho jayega. Kisi bhi sahayta ke liye humse contact karein.", "language": "Hindi", "category": "Loan / Finance"},
    {"id": 6, "text": "Flight booking ke liye 1 dabayen. Flight status ke liye 2 dabayen. Cancellation ke liye 3 dabayen.", "language": "Hindi", "category": "IVR Menu"},
    {"id": 7, "text": "Dhanyavaad IndiGo ko call karne ke liye. Aapka din mangalmay ho.", "language": "Hindi", "category": "Call Closing"},
    {"id": 8, "text": "Aapka PNR number hai A B 1 2 3 4. Ise save kar lijiye.", "language": "Hindi", "category": "Booking Confirmation"},
    {"id": 9, "text": "Kya aap travel insurance add karna chahenge? Yeh sirf rupees 299 mein available hai.", "language": "Hindi", "category": "Upsell / Add-on"},
    {"id": 10, "text": "Yeh final boarding call hai passengers Mr. Sharma aur Mrs. Gupta ke liye, flight 6E 888 ke liye gate C 3 par.", "language": "Hindi", "category": "Boarding Announcement"},
    {"id": 11, "text": "IndiGo BluChip Gold members aur business class passengers priority boarding le sakte hain.", "language": "Hindi", "category": "Boarding Announcement"},
    {"id": 12, "text": "IndiGo wallet mein minimum rupees 500 add kar sakte hain future bookings ke liye.", "language": "Hindi", "category": "Wallet / Payment"},
]


# ─── Helper Functions ────────────────────────────────────────────────────────
def load_metadata(model_dir: str) -> dict | None:
    """Load metadata.json from a model output directory."""
    meta_path = os.path.join(model_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def extract_zip_to_dir(zip_file, target_dir: str):
    """Extract uploaded zip file to target directory."""
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(target_dir)


def find_result_for_id(metadata: dict, test_id: int) -> dict | None:
    """Find the result entry matching a given test ID."""
    if metadata is None:
        return None
    for r in metadata.get("results", []):
        if r.get("id") == test_id:
            return r
    return None


def get_outputs_dir() -> str:
    """Get the outputs directory path."""
    return st.session_state.get("outputs_dir", "outputs")


def resolve_model_dir(outputs_dir: str, model_key: str) -> str:
    """Resolve a model's output directory, tolerating naming variants.

    Tries, in order: `<model_key>_outputs/`, `<model_key>/`, then case
    variants. Falls back to `<model_key>/` if none exist.
    """
    candidates = [
        f"{model_key}_outputs",
        model_key,
        f"{model_key}-outputs",
        model_key.upper(),
    ]
    for name in candidates:
        path = os.path.join(outputs_dir, name)
        if os.path.isdir(path):
            return path
    return os.path.join(outputs_dir, model_key)


# Default reference audio bundled in the outputs folder
DEFAULT_REF_AUDIO = os.path.join("outputs", "orpheus_voxcpm_sample_audio.mp3")


# ─── Metrics helpers ─────────────────────────────────────────────────────────
# Definitions surfaced in the UI so the numbers are self-explanatory.
METRIC_INFO = {
    "secs":  {"label": "Similarity",  "help": "SECS — neural speaker-embedding cosine similarity to the reference voice. Higher = more like the original speaker.", "fmt": "{:.2f}", "good": "high", "range": (0, 1)},
    "intel": {"label": "WER",         "help": "Intelligibility error — WER (English) or CER (Hindi). Lower = clearer / correct words. For Hindi, Whisper's Devanagari output is romanized and compared phonetically.", "fmt": "{:.0%}", "good": "low", "range": (0, 1)},
    "utmos": {"label": "Naturalness", "help": "UTMOS — predicted mean opinion score (1–5) for how human the audio sounds. Higher = more natural.", "fmt": "{:.2f}", "good": "high", "range": (1, 5)},
}


def metrics_path(outputs_dir: str) -> str:
    return os.path.join(outputs_dir, "metrics.json")


def load_metrics(outputs_dir: str) -> dict | None:
    """Load metrics.json if present."""
    p = metrics_path(outputs_dir)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def get_clip_metrics(metrics: dict | None, model_key: str, test_id: int) -> dict | None:
    """Pull the metric dict for one (model, test) pair."""
    if not metrics:
        return None
    return metrics.get("models", {}).get(model_key, {}).get("clips", {}).get(str(test_id))


def resolve_metrics_python() -> str:
    """Interpreter that has the heavy ML deps installed.

    Order: METRICS_PYTHON env -> local .venv-metrics -> current interpreter.
    """
    env = os.environ.get("METRICS_PYTHON")
    if env and os.path.exists(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (
        os.path.join(here, ".venv-metrics", "bin", "python"),
        os.path.join(here, ".venv-metrics", "Scripts", "python.exe"),
    ):
        if os.path.exists(cand):
            return cand
    return sys.executable


def metrics_regen_available() -> bool:
    """True only where metrics can actually be computed (heavy ML deps present).

    On a lightweight deploy (e.g. Railway) there's no `.venv-metrics` and torch
    isn't installed, so we hide the Regenerate button and just show precomputed
    metrics. Set METRICS_PYTHON to force-enable.
    """
    if os.environ.get("METRICS_PYTHON"):
        return True
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, ".venv-metrics", "bin", "python")) or \
       os.path.exists(os.path.join(here, ".venv-metrics", "Scripts", "python.exe")):
        return True
    import importlib.util
    return (importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("faster_whisper") is not None)


def regenerate_metrics(outputs_dir: str, reference: str | None, whisper_model: str):
    """Run compute_metrics.py as a subprocess; stream its log to a generator."""
    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        resolve_metrics_python(),
        os.path.join(here, "compute_metrics.py"),
        "--outputs-dir", outputs_dir,
        "--whisper-model", whisper_model,
    ]
    if reference and os.path.exists(reference):
        cmd += ["--reference", reference]
    proc = subprocess.Popen(
        cmd, cwd=here, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in iter(proc.stdout.readline, ""):
        yield line.rstrip("\n")
    proc.stdout.close()
    rc = proc.wait()
    yield f"__EXIT__ {rc}"


# Composite "/10" score weights. Speaker similarity is the headline of voice
# *cloning*, so it dominates; naturalness next; intelligibility last (and it's
# dropped for clips whose transcript is unreliable, weights renormalized).
SCORE_WEIGHTS = {"secs": 0.5, "utmos": 0.3, "intel": 0.2}
SCORE_FORMULA_HELP = (
    "Composite quality score out of 10 (higher = better), combining:\n"
    "• Similarity (SECS), weight 50%\n"
    "• Naturalness (UTMOS→0–1), weight 30%\n"
    "• Intelligibility (1−error: WER for English, CER for Hindi), weight 20%.\n"
    "Weights renormalize over whichever metrics are available."
)


def composite_score(secs, intel_error, utmos, reliable: bool = True):
    """Blend the three axes into a single 0–10 score. Returns None if nothing usable.

    intel_error is the primary intelligibility error (WER for English, CER for
    Hindi); lower is better, so it contributes 1 − error.
    """
    parts, weights = [], []
    if isinstance(secs, (int, float)):
        parts.append(min(max(secs, 0.0), 1.0))
        weights.append(SCORE_WEIGHTS["secs"])
    if isinstance(utmos, (int, float)):
        parts.append(min(max((utmos - 1) / 4.0, 0.0), 1.0))  # 1–5 -> 0–1
        weights.append(SCORE_WEIGHTS["utmos"])
    if isinstance(intel_error, (int, float)) and reliable:
        parts.append(min(max(1.0 - min(intel_error, 1.0), 0.0), 1.0))
        weights.append(SCORE_WEIGHTS["intel"])
    if not parts:
        return None
    total_w = sum(weights)
    blended = sum(p * w for p, w in zip(parts, weights)) / total_w
    return round(blended * 10, 1)


def clean_romanization(s: str) -> str:
    """Make a raw IAST romanization readable for display (does NOT affect metrics).

    1. Drop word-final inherent schwa ('a' after a consonant) — Hindi almost always
       deletes it. We do this *before* stripping diacritics so long vowels (ā, ī…)
       are preserved (e.g. "kā" stays "ka", not "k").
    2. Strip diacritics to plain ASCII (ā→a, ṃ→m, ṭ→t, ś→s …).
    "aba vokiṃga … karalijagā kā" -> "ab vokimg … karalijaga ka".
    """
    import unicodedata
    vowels = set("aeiouāīūēōṛṝḷ")
    out = []
    for word in s.split():
        if len(word) >= 2 and word[-1] == "a" and word[-2].isalpha() and word[-2].lower() not in vowels:
            word = word[:-1]  # trim final schwa
        word = "".join(c for c in unicodedata.normalize("NFKD", word) if not unicodedata.combining(c))
        out.append(word)
    return " ".join(out).lower()


def score_color(score) -> str:
    if score is None:
        return "#6b7280"
    if score >= 7.5:
        return "#34d399"
    if score >= 5.0:
        return "#fbbf24"
    return "#f87171"


def metric_color(key: str, value) -> str:
    """Green/amber/red hint based on whether higher or lower is better."""
    if value is None:
        return "#6b7280"
    info = METRIC_INFO[key]
    lo, hi = info["range"]
    norm = (value - lo) / (hi - lo) if hi != lo else 0.5
    if info["good"] == "low":
        norm = 1 - norm
    if norm >= 0.66:
        return "#34d399"
    if norm >= 0.4:
        return "#fbbf24"
    return "#f87171"


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # Data source
    st.markdown("### 📂 Data Source")
    data_mode = st.radio(
        "How to load audio outputs?",
        ["Local directory", "Upload zip files"],
        index=0,
        label_visibility="collapsed",
    )

    if data_mode == "Local directory":
        outputs_dir = st.text_input(
            "Outputs directory path",
            value="outputs",
            help="Path to the directory containing orpheus/, voxcpm2/, vibevoice/ subdirs",
        )
        st.session_state["outputs_dir"] = outputs_dir
    else:
        st.markdown("Upload the 3 zip files from Colab:")
        outputs_dir = os.path.join(tempfile.gettempdir(), "voice_comparison_outputs")
        st.session_state["outputs_dir"] = outputs_dir

        for model_key, model_info in MODELS.items():
            uploaded = st.file_uploader(
                f"📦 {model_info['name']} zip",
                type=["zip"],
                key=f"zip_{model_key}",
            )
            if uploaded:
                target = os.path.join(outputs_dir, model_key)
                extract_zip_to_dir(uploaded, target)
                st.success(f"✅ Extracted {model_info['name']}")

    st.divider()

    # Original reference audio upload
    st.markdown("### 🎤 Original Reference Audio")
    ref_audio = st.file_uploader(
        "Upload the original voice sample",
        type=["mp3", "wav", "ogg", "flac"],
        key="ref_audio",
        help="The original voice clip used as reference for cloning",
    )
    if ref_audio is not None:
        st.session_state["ref_audio_data"] = ref_audio.read()
        st.session_state["ref_audio_name"] = ref_audio.name
        st.session_state["ref_audio_type"] = ref_audio.type
        st.success(f"✅ Loaded: {ref_audio.name}")

    st.divider()

    # Evaluation metrics
    st.markdown("### 📊 Evaluation Metrics")
    _regen_available = metrics_regen_available()
    if _regen_available:
        whisper_choice = st.selectbox(
            "Whisper model (WER accuracy vs. speed)",
            ["tiny", "base", "small", "medium"],
            index=2,
            help="Larger = more accurate transcription (better WER) but slower. 'small' is a good balance.",
        )
        regen_clicked = st.button(
            "🔄 Regenerate Metrics",
            use_container_width=True,
            help="Recompute SECS / WER / UTMOS for every clip in the outputs folder. "
                 "Unchanged clips are served from cache, so this is fast after the first run.",
        )
    else:
        whisper_choice = "small"
        regen_clicked = False
        st.caption(
            "ℹ️ Metrics are **precomputed** here. Regeneration needs the ML deps "
            "(torch/Whisper) — run `compute_metrics.py` locally, commit "
            "`outputs/metrics.json`, and redeploy. See METRICS.md."
        )
    _existing_metrics = load_metrics(st.session_state.get("outputs_dir", "outputs"))
    if _existing_metrics:
        st.caption(f"Last computed: {_existing_metrics.get('computed_at', 'unknown')}")
    elif _regen_available:
        st.caption("No metrics computed yet — click above.")

    st.divider()

    # Filters
    st.markdown("### 🔍 Filters")
    languages = ["All", "English", "Hindi"]
    selected_language = st.selectbox("Language", languages, index=0)

    categories = sorted(set(t["category"] for t in TEST_TEXTS))
    categories.insert(0, "All")
    selected_category = st.selectbox("Category", categories, index=0)

    st.divider()

    # Model visibility
    st.markdown("### 👁️ Models to Show")
    show_models = {}
    for model_key, model_info in MODELS.items():
        show_models[model_key] = st.checkbox(
            model_info["name"], value=True, key=f"show_{model_key}"
        )

    st.divider()
    st.markdown(
        "<p style='font-size:0.75rem;color:#6b7280;text-align:center;'>"
        "Voice Cloning Comparison Dashboard<br>Built for batch testing evaluation</p>",
        unsafe_allow_html=True,
    )


# ─── Load Data ───────────────────────────────────────────────────────────────
outputs_dir = get_outputs_dir()

# Resolve the reference audio path to feed the metrics harness. If the user
# uploaded one, persist it to a stable temp file so the subprocess can read it.
def _resolve_reference_path() -> str | None:
    if "ref_audio_data" in st.session_state:
        ext = os.path.splitext(st.session_state.get("ref_audio_name", "ref.mp3"))[1] or ".mp3"
        ref_tmp = os.path.join(tempfile.gettempdir(), f"voice_ref{ext}")
        with open(ref_tmp, "wb") as f:
            f.write(st.session_state["ref_audio_data"])
        return ref_tmp
    if os.path.exists(DEFAULT_REF_AUDIO):
        return DEFAULT_REF_AUDIO
    return None


# ─── Regenerate Metrics (triggered from sidebar) ─────────────────────────────
if regen_clicked:
    ref_path = _resolve_reference_path()
    if ref_path is None:
        st.warning("⚠️ No reference audio available — SECS (speaker similarity) will be skipped.")
    st.markdown("### 🔄 Computing metrics…")
    status = st.status("Starting metrics computation…", expanded=True)
    log_lines: list[str] = []
    log_box = status.empty()
    exit_code = None
    try:
        for line in regenerate_metrics(outputs_dir, ref_path, whisper_choice):
            if line.startswith("__EXIT__"):
                exit_code = int(line.split()[1])
                break
            log_lines.append(line)
            # Keep the tail visible; full log can grow large.
            log_box.code("\n".join(log_lines[-25:]))
    except FileNotFoundError as e:
        status.update(label="❌ Could not launch metrics harness", state="error")
        st.error(
            f"Failed to run compute_metrics.py: {e}\n\n"
            "Ensure the metrics venv exists (`.venv-metrics`) or set the "
            "`METRICS_PYTHON` env var to a Python with the ML deps installed."
        )
        exit_code = -1
    if exit_code == 0:
        status.update(label="✅ Metrics computed", state="complete")
        st.cache_data.clear()
        st.rerun()
    elif exit_code not in (None, -1):
        status.update(label=f"❌ Metrics failed (exit {exit_code})", state="error")

model_metadata = {}
model_available = {}

model_dirs = {}
for model_key in MODELS:
    model_dir = resolve_model_dir(outputs_dir, model_key)
    model_dirs[model_key] = model_dir
    meta = load_metadata(model_dir)
    model_metadata[model_key] = meta
    model_available[model_key] = meta is not None

# Evaluation metrics (computed by compute_metrics.py -> outputs/metrics.json)
metrics = load_metrics(outputs_dir)


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🗣️ Voice Cloning Model Comparison</h1>
        <p>Compare audio outputs from Orpheus, VoxCPM2, and VibeVoice across 12 test texts</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Original Reference Audio Player ────────────────────────────────────────
# Prefer an uploaded clip; otherwise fall back to the bundled sample audio.
if "ref_audio_data" in st.session_state:
    ref_data = st.session_state["ref_audio_data"]
    ref_type = st.session_state.get("ref_audio_type", "audio/mpeg")
elif os.path.exists(DEFAULT_REF_AUDIO):
    with open(DEFAULT_REF_AUDIO, "rb") as f:
        ref_data = f.read()
    ref_type = "audio/mpeg"
else:
    ref_data = None

if ref_data is not None:
    st.markdown(
        """
        <div class="reference-card">
            <div class="ref-icon">🎤</div>
            <div class="ref-info">
                <h3>Original Reference Voice</h3>
                <p>The source voice sample used for cloning</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.audio(ref_data, format=ref_type)
    st.markdown("")

# ─── Stats Row ───────────────────────────────────────────────────────────────
active_models = [k for k, v in show_models.items() if v]
loaded_models = [k for k in active_models if model_available.get(k)]
visible_models = [k for k in active_models if show_models.get(k, False)]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f'<div class="stats-card"><h3 style="color:#667eea">{len(TEST_TEXTS)}</h3>'
        f"<p>Test Texts</p></div>",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="stats-card"><h3 style="color:#34d399">{len(loaded_models)}</h3>'
        f"<p>Models Loaded</p></div>",
        unsafe_allow_html=True,
    )
with col3:
    total_success = 0
    for mk in loaded_models:
        meta = model_metadata[mk]
        if meta:
            total_success += sum(1 for r in meta.get("results", []) if r.get("status") == "success")
    st.markdown(
        f'<div class="stats-card"><h3 style="color:#fbbf24">{total_success}</h3>'
        f"<p>Successful Outputs</p></div>",
        unsafe_allow_html=True,
    )
with col4:
    total_errors = 0
    for mk in loaded_models:
        meta = model_metadata[mk]
        if meta:
            total_errors += sum(1 for r in meta.get("results", []) if r.get("status") == "error")
    st.markdown(
        f'<div class="stats-card"><h3 style="color:#f87171">{total_errors}</h3>'
        f"<p>Errors</p></div>",
        unsafe_allow_html=True,
    )

st.markdown("")

# ─── Data Check ──────────────────────────────────────────────────────────────
if not any(model_available.values()):
    st.warning(
        "⚠️ No model outputs found. Please either:\n"
        "- Set the **outputs directory** to the folder containing `orpheus/`, `voxcpm2/`, `vibevoice/` subdirectories, or\n"
        "- **Upload the zip files** from Colab using the sidebar."
    )
    st.info(
        "**Expected folder structure:**\n"
        "```\n"
        "outputs/\n"
        "  orpheus/\n"
        "    metadata.json\n"
        "    test_01.wav ... test_12.wav\n"
        "  voxcpm2/\n"
        "    metadata.json\n"
        "    test_01.wav ... test_12.wav\n"
        "  vibevoice/\n"
        "    metadata.json\n"
        "    test_01.wav ... test_12.wav\n"
        "```"
    )
    st.stop()


# ─── Evaluation Leaderboard ──────────────────────────────────────────────────
if metrics:
    st.markdown("## 🏆 Evaluation Leaderboard")
    st.caption(
        "Averaged across all clips. **Similarity (SECS)** ↑ better · "
        "**Error (WER·CER)** ↓ better — WER for English, CER for Hindi (Whisper output romanized & phonetically compared); n = reliable clips · "
        "**Naturalness (UTMOS)** ↑ better · "
        f"Reference: `{os.path.basename(metrics.get('reference_audio') or '—')}` · "
        f"Whisper: `{metrics.get('whisper_model', '?')}`"
    )
    board = []
    scored = []  # (model_key, score) for the headline cards
    for model_key in (visible_models or list(MODELS)):
        agg = metrics.get("models", {}).get(model_key, {}).get("aggregate")
        if not agg:
            continue
        def _f(key, fmt):
            v = agg.get(key)
            return fmt.format(v) if isinstance(v, (int, float)) else "—"
        n_rel = agg.get("intel_error_n_reliable")
        intel_label = _f("intel_error_mean", "{:.1%}")
        if isinstance(n_rel, int):
            intel_label += f"  (n={n_rel})"
        score = composite_score(
            agg.get("secs_mean"), agg.get("intel_error_mean"), agg.get("utmos_mean"),
            reliable=bool(n_rel),
        )
        scored.append((model_key, score))
        board.append({
            "Model": MODELS[model_key]["name"],
            "Score /10 ↑": f"{score:.1f}" if score is not None else "—",
            "Similarity (SECS) ↑": _f("secs_mean", "{:.3f}"),
            "Error WER·CER ↓": intel_label,
            "Naturalness (UTMOS) ↑": _f("utmos_mean", "{:.2f}"),
            "SNR (dB) ↑": _f("snr_db_mean", "{:.1f}"),
            "Clips": agg.get("n", 0),
        })

    # Headline score cards — the at-a-glance "how good is each clone".
    if scored:
        best = max((s for _, s in scored if s is not None), default=None)
        score_cols = st.columns(len(scored))
        for col, (mk, score) in zip(score_cols, scored):
            color = score_color(score)
            crown = " 👑" if (score is not None and score == best) else ""
            with col:
                st.markdown(
                    f'<div class="stats-card" title="{SCORE_FORMULA_HELP}">'
                    f'<h3 style="color:{color}">{score:.1f}<span style="font-size:1rem;color:#9ca3af">/10</span></h3>'
                    f'<p>{MODELS[mk]["name"]}{crown}</p></div>'
                    if score is not None else
                    f'<div class="stats-card"><h3 style="color:#6b7280">—</h3>'
                    f'<p>{MODELS[mk]["name"]}</p></div>',
                    unsafe_allow_html=True,
                )
        st.markdown("")

    if board:
        st.dataframe(board, use_container_width=True, hide_index=True)
    st.caption(SCORE_FORMULA_HELP)
    st.markdown("")


# ─── Filter Test Texts ───────────────────────────────────────────────────────
filtered_texts = TEST_TEXTS.copy()
if selected_language != "All":
    filtered_texts = [t for t in filtered_texts if t["language"] == selected_language]
if selected_category != "All":
    filtered_texts = [t for t in filtered_texts if t["category"] == selected_category]

st.markdown(f"**Showing {len(filtered_texts)} of {len(TEST_TEXTS)} test texts**")

# ─── Comparison Cards ────────────────────────────────────────────────────────
for test in filtered_texts:
    test_id = test["id"]
    lang_class = "badge-english" if test["language"] == "English" else "badge-hindi"

    # Card header
    st.markdown(
        f"""
        <div class="test-card">
            <div class="test-header">
                <span class="test-id">#{test_id}</span>
                <span class="badge-lang {lang_class}">{test['language']}</span>
                <span class="badge-category">{test['category']}</span>
            </div>
            <div class="test-text">{test['text']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Audio columns
    cols = st.columns(len(visible_models)) if visible_models else []

    for col_idx, model_key in enumerate(visible_models):
        model_info = MODELS[model_key]
        meta = model_metadata.get(model_key)
        result = find_result_for_id(meta, test_id)

        with cols[col_idx]:
            st.markdown(
                f'<div class="model-label {model_info["css_class"]}">'
                f"{model_info['name']}</div>",
                unsafe_allow_html=True,
            )

            if result is None:
                st.markdown(
                    '<span class="status-missing">No data available</span>',
                    unsafe_allow_html=True,
                )
            elif result.get("status") == "error":
                st.markdown(
                    f'<span class="status-error">❌ Error: {result.get("error", "Unknown")[:60]}</span>',
                    unsafe_allow_html=True,
                )
            else:
                wav_path = os.path.join(
                    model_dirs[model_key], result.get("wav_file", f"test_{test_id:02d}.wav")
                )
                if os.path.exists(wav_path):
                    st.audio(wav_path, format="audio/wav")
                    gen_time = result.get("generation_time_sec", 0)
                    dur = result.get("audio_duration_sec", 0)
                    st.markdown(
                        f'<div class="gen-time">⏱ {gen_time:.1f}s gen &nbsp;|&nbsp; 🔊 {dur:.1f}s audio</div>',
                        unsafe_allow_html=True,
                    )
                    # ── Evaluation metric chips (if computed) ──
                    cm = get_clip_metrics(metrics, model_key, test_id)
                    if cm:
                        clip_score = composite_score(
                            cm.get("secs"), cm.get("intel_error"), cm.get("utmos"),
                            reliable=cm.get("wer_reliable", True),
                        )
                        chips = []
                        if clip_score is not None:
                            sc = score_color(clip_score)
                            chips.append(
                                f'<span class="metric-chip score-chip" style="border-color:{sc};color:{sc};" '
                                f'title="{SCORE_FORMULA_HELP}">★ {clip_score:.1f}/10</span>'
                            )
                        for mk_metric, info in METRIC_INFO.items():
                            # Intelligibility: dynamic label (WER/CER), muted if unreliable.
                            if mk_metric == "intel":
                                val = cm.get("intel_error")
                                label = cm.get("intel_label", "WER")
                                unreliable = not cm.get("wer_reliable", True)
                            else:
                                val = cm.get(mk_metric)
                                label = info["label"]
                                unreliable = False
                            if val is None or unreliable:
                                txt = "n/a ⚠" if unreliable else "—"
                                color = "#6b7280"
                            else:
                                txt = info["fmt"].format(val)
                                # Show the scale for non-percentage metrics: "0.95 / 1".
                                if not info["fmt"].endswith("%}"):
                                    txt += f" / {info['range'][1]:g}"
                                color = metric_color(mk_metric, val)
                            chips.append(
                                f'<span class="metric-chip" style="border-color:{color};color:{color};" '
                                f'title="{info["help"]}">{label}: {txt}</span>'
                            )
                        st.markdown(
                            f'<div class="metric-row">{"".join(chips)}</div>',
                            unsafe_allow_html=True,
                        )
                        # Show the ASR transcript (collapsed) so the error rate is auditable
                        transcript = cm.get("transcript")
                        if transcript:
                            with st.expander("📝 transcript"):
                                st.caption(transcript)
                                romanized = cm.get("transcript_romanized")
                                if romanized:
                                    st.caption(f"↳ romanized: _{clean_romanization(romanized)}_")
                                if not cm.get("wer_reliable", True):
                                    st.caption("⚠️ transcript script differs from reference — error rate approximate.")
                else:
                    st.markdown(
                        f'<span class="status-missing">WAV not found: {wav_path}</span>',
                        unsafe_allow_html=True,
                    )

    st.markdown("---")


# ─── Generation Time Comparison ──────────────────────────────────────────────
if any(model_available.values()):
    st.markdown("## ⏱️ Generation Time Comparison")

    # Build a table
    table_data = []
    for test in TEST_TEXTS:
        row = {"ID": test["id"], "Text": test["text"][:50] + "...", "Language": test["language"], "Category": test["category"]}
        for model_key in visible_models:
            meta = model_metadata.get(model_key)
            result = find_result_for_id(meta, test["id"])
            if result and result.get("status") == "success":
                row[MODELS[model_key]["name"]] = f"{result.get('generation_time_sec', 0):.1f}s"
            elif result and result.get("status") == "error":
                row[MODELS[model_key]["name"]] = "❌ Error"
            else:
                row[MODELS[model_key]["name"]] = "—"
        table_data.append(row)

    st.dataframe(table_data, use_container_width=True, hide_index=True)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#6b7280;font-size:0.8rem;'>"
    "🗣️ Voice Cloning Model Comparison Dashboard &nbsp;•&nbsp; "
    "Orpheus 3B &nbsp;|&nbsp; VoxCPM2 2B &nbsp;|&nbsp; VibeVoice Hindi 1.5B"
    "</p>",
    unsafe_allow_html=True,
)
