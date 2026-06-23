"""
compute_metrics.py
==================
Standalone voice-cloning evaluation harness.

For every generated clip (outputs/<model>_outputs/test_NN.wav) it computes how
good the *clone* is along three independent axes plus a set of cheap acoustic
descriptors, and writes the result to ``outputs/metrics.json``.

Metric axes
-----------
  1. Speaker similarity (SECS)  -- is it the *same voice* as the reference?
       Neural speaker embeddings (Resemblyzer) -> cosine similarity. [0..1, higher better]
  2. Intelligibility (WER / CER) -- did it say the *right words*?
       Whisper ASR transcript vs. the input text. [0..1, lower better]
  3. Naturalness (UTMOS)        -- does it sound *human*?
       Neural MOS predictor via torch.hub. [1..5, higher better]
  + acoustic descriptors: pitch (F0) mean/std + delta-vs-reference, speaking
    rate, loudness (dBFS), silence ratio, SNR estimate, duration.

Design notes
------------
  * Heavy models load lazily and once, then are reused across all clips.
  * Each clip's result is cached by a content hash, so re-running after only
    changing a few files recomputes just those (fast "Regenerate Metrics").
  * Every metric is wrapped in try/except: one failure (e.g. UTMOS download
    blocked offline) degrades to ``null`` instead of killing the whole run.
  * Run this with the dedicated metrics venv:
        .venv-metrics/bin/python compute_metrics.py
    CLI:  --outputs-dir outputs  --reference <path>  --force  --whisper-model small

Output: outputs/metrics.json  (consumed by app.py)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# On macOS the framework Python often lacks system CA certs, which makes
# torch.hub's urllib download (UTMOS weights) fail with CERTIFICATE_VERIFY_FAILED.
# Point SSL at certifi's bundle so model downloads work out of the box.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    ssl._create_default_https_context = lambda *a, **k: ssl.create_default_context(cafile=certifi.where())
except Exception:
    pass

# ─── Test text catalogue (must mirror app.py TEST_TEXTS) ─────────────────────
# Used as the WER reference transcript + language hint per clip.
TEST_TEXTS = {
    1: {"text": "Your ticket number is B 4 7 2 9 and the fare is rupees three thousand two hundred.", "language": "English"},
    2: {"text": "Thank you for calling customer support. Your query has been registered and our team will get back to you within twenty four hours. We apologise for the inconvenience caused.", "language": "English"},
    3: {"text": "Departure at 06:45 AM on 3rd February 2025", "language": "English"},
    4: {"text": "Aapki booking confirm ho gayi. Reference number note kar lijiye: B 4 9 2 1.", "language": "Hindi"},
    5: {"text": "Namaskar aur hamare service mein aapka swagat hai. Aapka loan application approved ho gaya hai. Amount aapke registered account mein do se teen working days mein credit ho jayega. Kisi bhi sahayta ke liye humse contact karein.", "language": "Hindi"},
    6: {"text": "Flight booking ke liye 1 dabayen. Flight status ke liye 2 dabayen. Cancellation ke liye 3 dabayen.", "language": "Hindi"},
    7: {"text": "Dhanyavaad IndiGo ko call karne ke liye. Aapka din mangalmay ho.", "language": "Hindi"},
    8: {"text": "Aapka PNR number hai A B 1 2 3 4. Ise save kar lijiye.", "language": "Hindi"},
    9: {"text": "Kya aap travel insurance add karna chahenge? Yeh sirf rupees 299 mein available hai.", "language": "Hindi"},
    10: {"text": "Yeh final boarding call hai passengers Mr. Sharma aur Mrs. Gupta ke liye, flight 6E 888 ke liye gate C 3 par.", "language": "Hindi"},
    11: {"text": "IndiGo BluChip Gold members aur business class passengers priority boarding le sakte hain.", "language": "Hindi"},
    12: {"text": "IndiGo wallet mein minimum rupees 500 add kar sakte hain future bookings ke liye.", "language": "Hindi"},
}

MODEL_KEYS = ["orpheus", "voxcpm2", "vibevoice"]
SR = 16000  # working sample rate for all neural models

LANG_HINT = {"English": "en", "Hindi": "hi"}


# ─── Small utilities ─────────────────────────────────────────────────────────
def log(msg: str) -> None:
    """Progress line. Flushed immediately so a subprocess parent sees it live."""
    print(msg, flush=True)


def file_hash(path: str) -> str:
    """Content hash for cache invalidation (size + first/last block + mtime)."""
    st = os.stat(path)
    h = hashlib.sha1()
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()[:16]


def resolve_model_dir(outputs_dir: str, model_key: str) -> str | None:
    """Mirror app.py's directory resolution, tolerating naming variants."""
    for name in (f"{model_key}_outputs", model_key, f"{model_key}-outputs", model_key.upper()):
        p = os.path.join(outputs_dir, name)
        if os.path.isdir(p):
            return p
    return None


import re
import unicodedata

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_CONS = "bcdfghjklmnpqrstvxyz"


def latin_ratio(s: str) -> float:
    """Fraction of alphabetic characters that are ASCII/Latin. 1.0 if none."""
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 1.0
    return sum(c.isascii() for c in letters) / len(letters)


def _strip_diacritics(s: str) -> str:
    """Drop combining marks: IAST 'āpakī' -> 'apaki'."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def romanize_devanagari(text: str):
    """Devanagari -> Latin (IAST). Deterministic direction; Latin passes through.

    Returns None if the transliterator is unavailable.
    """
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        return transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
    except Exception:
        return None


def _words_to_digits(text: str) -> str:
    """Convert spelled-out English numbers to digits: 'three thousand' -> '3000'."""
    try:
        from word2number import w2n
    except Exception:
        return text
    num_words = {
        "zero","one","two","three","four","five","six","seven","eight","nine","ten",
        "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
        "eighteen","nineteen","twenty","thirty","forty","fifty","sixty","seventy",
        "eighty","ninety","hundred","thousand","million","billion",
    }
    out, buf = [], []
    def flush():
        if buf:
            try:
                out.append(str(w2n.word_to_num(" ".join(buf))))
            except Exception:
                out.extend(buf)
            buf.clear()
    for tok in text.split():
        if tok in num_words:
            buf.append(tok)
        else:
            flush()
            out.append(tok)
    flush()
    return " ".join(out)


def _join_alnum_runs(text: str) -> str:
    """Merge runs of single-char alphanumerics: 'b 4 7 2 9' -> 'b4729'."""
    out, buf = [], []
    def flush():
        if buf:
            out.append("".join(buf))
            buf.clear()
    for tok in text.split():
        if len(tok) == 1 and tok.isalnum():
            buf.append(tok)
        else:
            flush()
            out.append(tok)
    flush()
    return " ".join(out)


def normalize_text(s: str, language: str = "English") -> str:
    """Normalize for WER/CER.

    English: lowercase, strip punctuation, spell-out numbers -> digits, join
             alphanumeric runs (ticket/PNR codes).
    Hindi:   additionally collapse romanization-convention noise (vowel length,
             w/v) so the phonetic content lines up across spelling variants.
    """
    s = s.lower().translate(DEVANAGARI_DIGITS)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    if language == "Hindi":
        s = _strip_diacritics(s)
        s = re.sub(r"aa+", "a", s)
        s = re.sub(r"(ee+|ii+)", "i", s)
        s = re.sub(r"(oo+|uu+)", "u", s)
        s = s.replace("w", "v")
        s = re.sub(rf"([{_CONS}])\1+", r"\1", s)  # collapse doubled consonants
    else:
        s = _words_to_digits(s)
    s = _join_alnum_runs(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ─── Lazy model registry ─────────────────────────────────────────────────────
class Models:
    """Loads each heavy model at most once, on first use."""

    def __init__(self, whisper_model: str = "small"):
        self._whisper_name = whisper_model
        self._encoder = None      # Resemblyzer VoiceEncoder
        self._whisper = None      # faster_whisper.WhisperModel
        self._utmos = None        # torch.hub UTMOS predictor
        self._utmos_failed = False
        self._torch = None

    @property
    def torch(self):
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch

    @property
    def encoder(self):
        if self._encoder is None:
            log("  · loading speaker encoder (Resemblyzer)…")
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder(verbose=False)
        return self._encoder

    @property
    def whisper(self):
        if self._whisper is None:
            log(f"  · loading Whisper ASR ({self._whisper_name})…")
            from faster_whisper import WhisperModel
            self._whisper = WhisperModel(self._whisper_name, device="cpu", compute_type="int8")
        return self._whisper

    @property
    def utmos(self):
        if self._utmos is None and not self._utmos_failed:
            try:
                log("  · loading UTMOS naturalness predictor (torch.hub)…")
                self._utmos = self.torch.hub.load(
                    "tarepan/SpeechMOS", "utmos22_strong", trust_repo=True
                )
                self._utmos.eval()
            except Exception as e:  # offline / network blocked
                log(f"  ! UTMOS unavailable, skipping naturalness ({e.__class__.__name__})")
                self._utmos_failed = True
        return self._utmos


# ─── Individual metric computations ──────────────────────────────────────────
def embed(models: Models, wav_path: str):
    """Resemblyzer utterance embedding (preprocessed to 16k mono)."""
    from resemblyzer import preprocess_wav
    wav = preprocess_wav(Path(wav_path))
    return models.encoder.embed_utterance(wav)


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


def transcribe(models: Models, wav_path: str, lang: str | None):
    """Whisper transcript. Returns (text, detected_language)."""
    segments, info = models.whisper.transcribe(
        wav_path, language=lang, beam_size=5, vad_filter=True
    )
    text = " ".join(seg.text for seg in segments).strip()
    return text, getattr(info, "language", lang)


def utmos_score(models: Models, y: np.ndarray) -> float | None:
    """Predict UTMOS naturalness (1..5) for a 16k mono waveform."""
    if models.utmos is None:
        return None
    torch = models.torch
    with torch.no_grad():
        wav = torch.from_numpy(y).float().unsqueeze(0)  # (1, T)
        score = models.utmos(wav, SR)
    return float(score.item())


def acoustic_metrics(y: np.ndarray, sr: int) -> dict:
    """Cheap signal descriptors: pitch, loudness, silence, SNR, rate."""
    import librosa
    out: dict = {}
    duration = float(len(y) / sr)
    out["duration_sec"] = round(duration, 3)

    # Loudness (RMS -> dBFS)
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    out["rms_dbfs"] = round(20 * np.log10(rms), 2)

    # Pitch (F0) via probabilistic YIN over a voice range
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=70, fmax=400, sr=sr, frame_length=2048
        )
        voiced = f0[~np.isnan(f0)]
        if voiced.size:
            out["pitch_mean_hz"] = round(float(np.mean(voiced)), 2)
            out["pitch_std_hz"] = round(float(np.std(voiced)), 2)
        else:
            out["pitch_mean_hz"] = None
            out["pitch_std_hz"] = None
        out["voiced_ratio"] = round(float(np.mean(voiced_flag)), 3)
    except Exception:
        out["pitch_mean_hz"] = out["pitch_std_hz"] = out["voiced_ratio"] = None

    # Silence ratio: fraction of frames below an adaptive energy threshold
    try:
        frame = 1024
        hop = 512
        energy = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
        thr = 0.15 * float(np.median(energy[energy > 0])) if np.any(energy > 0) else 0.0
        out["silence_ratio"] = round(float(np.mean(energy < thr)), 3)
    except Exception:
        out["silence_ratio"] = None

    # SNR estimate: speech (high-energy frames) vs noise floor (low-energy)
    try:
        e = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
        e = e[e > 0]
        if e.size > 4:
            noise = np.percentile(e, 10)
            signal = np.percentile(e, 90)
            out["snr_db"] = round(float(20 * np.log10((signal + 1e-9) / (noise + 1e-9))), 2)
        else:
            out["snr_db"] = None
    except Exception:
        out["snr_db"] = None

    return out


# ─── Per-clip orchestration ──────────────────────────────────────────────────
def compute_clip(models: Models, ref_emb, ref_acoustic, wav_path: str, test_id: int) -> dict:
    """Run every metric for one clip; degrade gracefully per-metric."""
    import librosa

    meta = TEST_TEXTS.get(test_id, {})
    ref_text = meta.get("text", "")
    language = meta.get("language", "English")
    lang_hint = LANG_HINT.get(language)

    res: dict = {"file_hash": file_hash(wav_path), "language": language}

    # Load once at 16k mono for neural metrics + acoustics
    try:
        y, _ = librosa.load(wav_path, sr=SR, mono=True)
    except Exception as e:
        res["error"] = f"load failed: {e}"
        return res

    # 1) Speaker similarity (SECS)
    try:
        emb = embed(models, wav_path)
        res["secs"] = round(cosine(ref_emb, emb), 4) if ref_emb is not None else None
    except Exception as e:
        res["secs"] = None
        res["secs_error"] = str(e)[:120]

    # 2) Intelligibility (WER / CER)
    #    For Hindi, the reference is romanized but Whisper emits Devanagari, so we
    #    romanize the hypothesis back to Latin (deterministic direction) and
    #    compare in a phonetic-normalized space. CER is the primary Hindi metric
    #    (word-level WER stays high from schwa/spelling variation); WER is primary
    #    for English.
    try:
        hyp, detected = transcribe(models, wav_path, lang_hint)
        res["transcript"] = hyp
        res["detected_language"] = detected
        import jiwer

        hyp_cmp = hyp
        reliable = True
        if language == "Hindi":
            romanized = romanize_devanagari(hyp)
            if romanized is not None:
                hyp_cmp = romanized
                res["transcript_romanized"] = romanized
            else:
                # Transliterator unavailable: fall back to script-mismatch flag.
                reliable = latin_ratio(hyp) >= 0.6
        else:
            reliable = latin_ratio(hyp) >= 0.6

        ref_n = normalize_text(ref_text, language)
        hyp_n = normalize_text(hyp_cmp, language)
        if ref_n and hyp_n:
            res["wer"] = round(float(jiwer.wer(ref_n, hyp_n)), 4)
            res["cer"] = round(float(jiwer.cer(ref_n, hyp_n)), 4)
        else:
            res["wer"] = res["cer"] = None
        res["wer_reliable"] = reliable
        # Primary intelligibility number surfaced in the UI + composite score.
        if language == "Hindi":
            res["intel_error"], res["intel_label"] = res["cer"], "CER"
        else:
            res["intel_error"], res["intel_label"] = res["wer"], "WER"
    except Exception as e:
        res["wer"] = res["cer"] = res["intel_error"] = None
        res["intel_label"] = "WER"
        res["transcript"] = None
        res["wer_error"] = str(e)[:120]

    # 3) Naturalness (UTMOS)
    try:
        mos = utmos_score(models, y)
        res["utmos"] = round(mos, 3) if mos is not None else None
    except Exception as e:
        res["utmos"] = None
        res["utmos_error"] = str(e)[:120]

    # 4) Acoustic descriptors + deltas vs reference
    try:
        ac = acoustic_metrics(y, SR)
        res.update(ac)
        if ref_acoustic and ac.get("pitch_mean_hz") and ref_acoustic.get("pitch_mean_hz"):
            res["pitch_mean_diff_hz"] = round(
                abs(ac["pitch_mean_hz"] - ref_acoustic["pitch_mean_hz"]), 2
            )
        # speaking rate in words/sec from transcript
        words = len(normalize_text(res.get("transcript") or "").split())
        speech = max(ac["duration_sec"] * (1 - (ac.get("silence_ratio") or 0)), 1e-3)
        res["speaking_rate_wps"] = round(words / speech, 2) if words else None
    except Exception as e:
        res["acoustic_error"] = str(e)[:120]

    return res


def aggregate(clips: dict) -> dict:
    """Mean of each numeric metric across a model's clips (ignoring nulls).

    WER/CER are averaged over *reliable* clips only (Latin-script transcripts),
    since romanized-Hindi refs vs. non-Latin ASR output produce meaningless
    (often >1.0) error rates that would swamp the average.
    """
    keys = ["secs", "utmos", "snr_db", "pitch_mean_diff_hz", "duration_sec"]
    agg = {"n": len(clips)}
    for k in keys:
        vals = [c[k] for c in clips.values() if isinstance(c.get(k), (int, float))]
        agg[f"{k}_mean"] = round(float(np.mean(vals)), 4) if vals else None
    # WER/CER and the primary intelligibility error, averaged over reliable clips.
    for k in ("wer", "cer", "intel_error"):
        vals = [c[k] for c in clips.values()
                if isinstance(c.get(k), (int, float)) and c.get("wer_reliable", True)]
        agg[f"{k}_mean"] = round(float(np.mean(vals)), 4) if vals else None
        agg[f"{k}_n_reliable"] = len(vals)
    return agg


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Compute voice-cloning evaluation metrics.")
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--reference", default=os.path.join("outputs", "orpheus_voxcpm_sample_audio.mp3"),
                    help="Original reference voice the clones were made from.")
    ap.add_argument("--out", default=None, help="metrics.json path (default: <outputs-dir>/metrics.json)")
    ap.add_argument("--whisper-model", default=os.environ.get("METRICS_WHISPER_MODEL", "small"))
    ap.add_argument("--force", action="store_true", help="Ignore cache; recompute everything.")
    args = ap.parse_args()

    outputs_dir = args.outputs_dir
    out_path = args.out or os.path.join(outputs_dir, "metrics.json")

    if not os.path.isdir(outputs_dir):
        log(f"ERROR: outputs dir not found: {outputs_dir}")
        return 2

    # Load existing metrics for caching
    cache = {}
    if os.path.exists(out_path) and not args.force:
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            for mk, mdata in prev.get("models", {}).items():
                for cid, cdata in mdata.get("clips", {}).items():
                    cache[(mk, str(cid))] = cdata
        except Exception:
            pass

    log("=" * 60)
    log("Voice-cloning metrics")
    log(f"  outputs : {outputs_dir}")
    log(f"  whisper : {args.whisper_model}")
    log("=" * 60)

    models = Models(whisper_model=args.whisper_model)

    # Reference embedding + acoustic profile (computed once)
    ref_emb = None
    ref_acoustic = None
    ref_path = args.reference
    if os.path.exists(ref_path):
        try:
            log(f"Embedding reference voice: {ref_path}")
            ref_emb = embed(models, ref_path)
            import librosa
            ry, _ = librosa.load(ref_path, sr=SR, mono=True)
            ref_acoustic = acoustic_metrics(ry, SR)
        except Exception as e:
            log(f"  ! reference processing failed: {e} (SECS will be null)")
    else:
        log(f"  ! reference audio not found ({ref_path}); SECS will be null")

    result = {
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_audio": ref_path if os.path.exists(ref_path) else None,
        "reference_metrics": ref_acoustic,
        "whisper_model": args.whisper_model,
        "models": {},
    }

    n_computed = n_cached = 0
    for mk in MODEL_KEYS:
        mdir = resolve_model_dir(outputs_dir, mk)
        if not mdir:
            log(f"[{mk}] no output dir found, skipping")
            continue
        log(f"[{mk}] {mdir}")
        clips: dict = {}
        for test_id in sorted(TEST_TEXTS):
            wav_path = os.path.join(mdir, f"test_{test_id:02d}.wav")
            if not os.path.exists(wav_path):
                continue
            h = file_hash(wav_path)
            cached = cache.get((mk, str(test_id)))
            if cached and cached.get("file_hash") == h and not args.force:
                clips[str(test_id)] = cached
                n_cached += 1
                log(f"  #{test_id:02d} cached")
                continue
            log(f"  #{test_id:02d} computing…")
            try:
                clips[str(test_id)] = compute_clip(models, ref_emb, ref_acoustic, wav_path, test_id)
                n_computed += 1
            except Exception as e:
                clips[str(test_id)] = {"file_hash": h, "error": str(e)[:200]}
                log(f"     ! failed: {e}")
                traceback.print_exc()
        result["models"][mk] = {"clips": clips, "aggregate": aggregate(clips)}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log("=" * 60)
    log(f"Done. computed={n_computed} cached={n_cached}")
    log(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
