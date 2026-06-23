# Voice-Cloning Evaluation Metrics

This adds objective, automatable quality metrics for the cloned voices shown in
the Streamlit dashboard (`app.py`). Update the `outputs/` folder, click
**🔄 Regenerate Metrics** in the sidebar, and the numbers refresh.

## What gets measured

Each generated clip is scored on three independent axes (plus acoustic
descriptors), all relative to the original reference voice
(`outputs/orpheus_voxcpm_sample_audio.mp3`):

| Metric | Question it answers | How | Range |
|---|---|---|---|
| **SECS** (Similarity) | Is it the *same voice*? | Neural speaker embeddings (Resemblyzer) → cosine similarity to the reference | 0–1, **higher** better |
| **WER / CER** | Did it say the *right words*? | Whisper ASR transcript vs. input text. WER is primary for English, **CER for Hindi** | 0–1, **lower** better |
| **UTMOS** (Naturalness) | Does it sound *human*? | Neural MOS predictor (`tarepan/SpeechMOS` via torch.hub) | 1–5, **higher** better |

The app also shows a **composite score out of 10** per clip and per model
(higher = better): 50% similarity + 30% naturalness + 20% intelligibility
(1 − error), with weights renormalized when a metric is missing.

Plus per-clip acoustic descriptors: pitch (F0) mean/std and Δ-vs-reference,
speaking rate (words/sec), loudness (dBFS), silence ratio, voiced ratio, and an
SNR estimate.

These three axes (similarity + intelligibility + naturalness) are exactly what
modern TTS / voice-cloning papers report, and none require human raters.

## One-time setup

The metric models (torch, Whisper, speaker encoder) are heavy and don't have
Python 3.14 wheels, so they live in a **separate Python 3.12 venv**, isolated
from the Streamlit runtime:

```bash
python3.12 -m venv .venv-metrics
.venv-metrics/bin/python -m pip install -r requirements-metrics.txt
```

The first metrics run downloads model weights (~400 MB UTMOS + the Whisper
model) and caches them under `~/.cache`. Subsequent runs are offline.

## Running

**From the app:** click **🔄 Regenerate Metrics** in the sidebar. It shells out
to `compute_metrics.py` using `.venv-metrics`, streams progress, then reloads
`outputs/metrics.json` and renders the leaderboard + per-clip chips.

**From the CLI:**

```bash
.venv-metrics/bin/python compute_metrics.py \
    --outputs-dir outputs \
    --reference outputs/orpheus_voxcpm_sample_audio.mp3 \
    --whisper-model small        # tiny|base|small|medium  (bigger = more accurate, slower)
```

Results are written to `outputs/metrics.json`. Runs are **cached by file
hash** — only clips whose `.wav` changed are recomputed, so regenerating after
swapping a few files is fast. Use `--force` to recompute everything.

## Notes & caveats

- **Hindi intelligibility (CER).** The reference texts are Hindi written in
  Latin script ("Aapki booking confirm..."), but Whisper transcribes Hindi audio
  into Devanagari. We romanize Whisper's output back to Latin (the *deterministic*
  direction, via `indic-transliteration`), phonetically normalize both sides
  (vowel-length, w/v, doubled consonants), and report **CER** — word-level WER
  stays high from schwa/spelling variation, so CER is the primary Hindi number.
  It won't be as clean as English (the content is loanword-heavy: "booking",
  "reference", "PNR"), but it's trustworthy. Both the Devanagari transcript and
  its romanization are shown in the UI so you can audit by ear.
- **Number normalization.** Spelled-out English numbers are converted to digits
  ("three thousand two hundred" → "3200") and alphanumeric runs are joined
  ("B 4 7 2 9" → "b4729") on both reference and transcript, so WER reflects real
  intelligibility instead of formatting differences.
- **`ai4bharat`/IndicXlit (neural transliteration) is *not* used** — it depends
  on `fairseq`, which won't build on Python 3.12 + torch 2.x (and calls torch
  internals removed in 2.x). The rule-based romanization above is the working
  alternative. If you later want neural quality, IndicXlit's ONNX export
  (`onnxruntime`, already installed) is the only fairseq-free path.
- **One reference voice** is assumed for all three models (they cloned the same
  speaker). Upload a different reference in the sidebar to re-score against it.
- **Custom interpreter:** set `METRICS_PYTHON=/path/to/python` to override which
  interpreter the app uses for the subprocess.
- **Deployment:** for Railway/cloud, the ML deps must be installed in the
  serving environment (the `.venv-metrics` split is a local-dev convenience).
  Computing metrics on a small cloud box is slow; prefer committing a
  precomputed `outputs/metrics.json` and treating the button as a local tool.
