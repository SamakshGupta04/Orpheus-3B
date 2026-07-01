#!/usr/bin/env python3
"""
Model 2 (VoxCPM2 2B) voice cloning inference.

Two modes:

1. One-shot CLI (batch / smoke tests) — loads the model, runs one inference,
   exits. VRAM is freed on exit.

       python infer_model2.py \
           --ref-audio  reference.wav \
           --target-text "Say this sentence." \
           --output  outputs/live/out.wav \
           [--ref-transcript "Words in the reference clip."] \
           [--cfg-value 2.0] \
           [--timesteps 10]

   Exit 0 on success, 1 on failure. Emits "__OUTPUT_SAMPLE_RATE__ XXXX"
   before exit.

2. Persistent server (used by app.py so the model stays warm across
   requests) — loads once, then serves inference requests over HTTP until
   told to shut down.

       python infer_model2.py --serve --port 8002 [--dry-run]

   See model_server.py for the wire protocol.
"""

import argparse
import sys
import os
import gc
import tempfile
import warnings

warnings.filterwarnings("ignore")

DEFAULT_SAMPLE_RATE = 48000  # VoxCPM2 default, used only for the dry-run tone


def _dry_run_audio(output_path: str, sample_rate: int = DEFAULT_SAMPLE_RATE, duration: float = 2.0):
    """Write a 440 Hz test tone to output_path without loading any model."""
    import numpy as np
    import soundfile as sf
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sf.write(output_path, audio, sample_rate)
    return sample_rate, duration


def load_model(log, dry_run: bool = False) -> dict:
    """Load VoxCPM2 once. Returns an objs dict consumed by run_inference."""
    if dry_run:
        log("[Model 2] DRY RUN mode — skipping model loading, will generate test tones.")
        return {"dry_run": True}

    import torch

    try:
        from voxcpm import VoxCPM
    except ImportError as e:
        raise RuntimeError("voxcpm not installed. Run: pip install voxcpm") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log("[Model 2] WARNING: No CUDA GPU detected. Inference will be very slow.")
    else:
        log(f"[Model 2] GPU: {torch.cuda.get_device_name(0)}")
    log(f"[Model 2] Device: {device}")

    log("[Model 2] Loading VoxCPM2 (this may take a few minutes on first run)...")
    try:
        model = VoxCPM.from_pretrained("openbmb/VoxCPM2", device=device)
    except Exception as e:
        raise RuntimeError(f"Could not load VoxCPM2: {e}") from e
    log("[Model 2] VoxCPM2 loaded.")

    return {"dry_run": False, "model": model, "device": device}


def run_inference(objs: dict, payload: dict, log) -> dict:
    """Run one cloning request.

    payload keys: ref_audio, ref_transcript (optional), target_text, output,
    cfg_value (optional), timesteps (optional).
    Returns {"output_path", "sample_rate", "duration"}.
    """
    output_path = payload["output"]

    if objs.get("dry_run"):
        sr, duration = _dry_run_audio(output_path, sample_rate=DEFAULT_SAMPLE_RATE)
        log(f"[Model 2] DRY RUN saved: {output_path}  ({duration:.1f}s @ {sr} Hz)")
        return {"output_path": output_path, "sample_rate": sr, "duration": duration}

    ref_audio = payload["ref_audio"]
    ref_transcript = (payload.get("ref_transcript") or "").strip()
    target_text = payload["target_text"]
    cfg_value = float(payload.get("cfg_value") or 2.0)
    timesteps = int(payload.get("timesteps") or 10)

    log(f"[Model 2] Reference audio : {ref_audio}")
    log(f"[Model 2] Target text     : {target_text[:80]}...")
    if ref_transcript:
        log(f"[Model 2] Transcript      : {ref_transcript[:80]}...")
        log("[Model 2] Using 'ultimate' cloning mode (reference + prompt transcript).")
    else:
        log("[Model 2] No transcript provided — using basic cloning mode.")

    import numpy as np
    import torch

    try:
        import librosa
    except ImportError as e:
        raise RuntimeError("librosa not installed. Run: pip install librosa") from e
    try:
        import soundfile as sf
    except ImportError as e:
        raise RuntimeError("soundfile not installed. Run: pip install soundfile") from e

    model = objs["model"]

    # ── Preprocess reference to 16 kHz mono WAV ────────────────────────────
    log("[Model 2] Preprocessing reference audio to 16 kHz mono...")
    try:
        ref_wav, _ = librosa.load(ref_audio, sr=16000, mono=True)
    except Exception as e:
        raise RuntimeError(f"Could not load reference audio: {e}") from e

    ref_wav, _ = librosa.effects.trim(ref_wav, top_db=30)
    peak = float(np.max(np.abs(ref_wav))) if ref_wav.size else 0.0
    if peak > 0:
        ref_wav = (ref_wav / peak) * 0.95
    ref_wav = ref_wav.astype(np.float32)
    log(f"[Model 2] Reference duration: {len(ref_wav) / 16000:.1f}s @ 16 kHz")

    # Write preprocessed reference to a temp WAV file. Include the PID so two
    # servers/jobs never race on the same path.
    tmp_ref_path = os.path.join(
        tempfile.gettempdir(), f"_voxcpm2_ref_{os.getpid()}.wav"
    )
    sf.write(tmp_ref_path, ref_wav, 16000)
    log(f"[Model 2] Preprocessed reference saved to: {tmp_ref_path}")

    # ── Generate ────────────────────────────────────────────────────────────
    gen_kwargs = dict(
        text=target_text,
        reference_wav_path=tmp_ref_path,
        cfg_value=cfg_value,
        inference_timesteps=timesteps,
    )
    if ref_transcript:
        gen_kwargs["prompt_wav_path"] = tmp_ref_path
        gen_kwargs["prompt_text"] = ref_transcript

    log(f"[Model 2] Generating with cfg={cfg_value}, steps={timesteps}...")
    try:
        audio = model.generate(**gen_kwargs)
    except torch.cuda.OutOfMemoryError as e:
        gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError("CUDA OOM during generation.") from e
    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}") from e
    finally:
        try:
            os.remove(tmp_ref_path)
        except OSError:
            pass

    audio_np = np.asarray(audio, dtype=np.float32).squeeze()
    try:
        sample_rate = int(model.tts_model.sample_rate)
    except AttributeError:
        sample_rate = DEFAULT_SAMPLE_RATE
    duration = len(audio_np) / sample_rate
    log(f"[Model 2] Generated {duration:.1f}s @ {sample_rate} Hz.")

    # ── Save output ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sf.write(output_path, audio_np, sample_rate)
    log(f"[Model 2] Saved: {output_path}  ({duration:.1f}s @ {sample_rate} Hz)")

    return {"output_path": output_path, "sample_rate": sample_rate, "duration": duration}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref-audio")
    p.add_argument("--ref-transcript", default="")
    p.add_argument("--target-text")
    p.add_argument("--output")
    p.add_argument("--cfg-value", type=float, default=2.0)
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--dry-run", action="store_true",
                   help="Skip model loading; write a test tone to --output instead.")
    p.add_argument("--serve", action="store_true",
                   help="Run as a persistent HTTP inference server instead of one-shot CLI.")
    p.add_argument("--port", type=int, default=int(os.environ.get("MODEL2_PORT", 8002)))
    args = p.parse_args()

    if args.serve:
        from model_server import ModelServer, serve
        server = ModelServer(
            name="Model 2",
            load_fn=lambda log: load_model(log, dry_run=args.dry_run),
            infer_fn=run_inference,
        )
        serve(server, args.port, dry_run_label=" (dry-run)" if args.dry_run else "")
        return

    if not args.ref_audio or not args.target_text or not args.output:
        p.error("--ref-audio, --target-text and --output are required in one-shot mode")

    try:
        objs = load_model(print, dry_run=args.dry_run)
        result = run_inference(objs, {
            "ref_audio": args.ref_audio,
            "ref_transcript": args.ref_transcript,
            "target_text": args.target_text,
            "output": args.output,
            "cfg_value": args.cfg_value,
            "timesteps": args.timesteps,
        }, print)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"__OUTPUT_SAMPLE_RATE__ {result['sample_rate']}")


if __name__ == "__main__":
    main()
