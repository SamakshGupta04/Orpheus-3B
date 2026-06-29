#!/usr/bin/env python3
"""
Model 2 (VoxCPM2 2B) voice cloning inference script.

Called as a subprocess by app.py so that VRAM is freed on exit and deps are
isolated from the other models.

Usage:
    python infer_model2.py \
        --ref-audio  reference.wav \
        --target-text "Say this sentence." \
        --output  outputs/live/out.wav \
        [--ref-transcript "Words in the reference clip."] \
        [--cfg-value 2.0] \
        [--timesteps 10]

Exit 0 on success, 1 on failure.
Prints progress to stdout. Emits "__OUTPUT_SAMPLE_RATE__ XXXX" before exit.
"""

import argparse
import sys
import os
import gc
import tempfile
import warnings

warnings.filterwarnings("ignore")


def _dry_run(output_path: str, sample_rate: int = 48000, duration: float = 2.0) -> None:
    """Write a 440 Hz test tone to output_path without loading any model."""
    import numpy as np
    import soundfile as sf
    print("[Model 2] DRY RUN — skipping model loading, generating test tone.")
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sf.write(output_path, audio, sample_rate)
    print(f"[Model 2] DRY RUN saved: {output_path}  ({duration:.1f}s @ {sample_rate} Hz)")
    print(f"__OUTPUT_SAMPLE_RATE__ {sample_rate}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref-audio", required=True)
    p.add_argument("--ref-transcript", default="")
    p.add_argument("--target-text", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cfg-value", type=float, default=2.0)
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--dry-run", action="store_true",
                   help="Skip model loading; write a test tone to --output instead.")
    args = p.parse_args()

    if args.dry_run:
        _dry_run(args.output, sample_rate=48000)
        return

    print(f"[Model 2] Reference audio : {args.ref_audio}")
    print(f"[Model 2] Target text     : {args.target_text[:80]}...")
    if args.ref_transcript:
        print(f"[Model 2] Transcript      : {args.ref_transcript[:80]}...")
        print("[Model 2] Using 'ultimate' cloning mode (reference + prompt transcript).")
    else:
        print("[Model 2] No transcript provided — using basic cloning mode.")

    import numpy as np

    try:
        import torch
    except ImportError:
        print("ERROR: torch not installed.", file=sys.stderr)
        sys.exit(1)

    try:
        import librosa
    except ImportError:
        print("ERROR: librosa not installed. Run: pip install librosa", file=sys.stderr)
        sys.exit(1)

    try:
        import soundfile as sf
    except ImportError:
        print("ERROR: soundfile not installed. Run: pip install soundfile", file=sys.stderr)
        sys.exit(1)

    try:
        from voxcpm import VoxCPM
    except ImportError:
        print("ERROR: voxcpm not installed. Run: pip install voxcpm", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[Model 2] WARNING: No CUDA GPU detected. Inference will be very slow.")
    else:
        print(f"[Model 2] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Model 2] Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("[Model 2] Loading VoxCPM2 (this may take a few minutes on first run)...")
    try:
        model = VoxCPM.from_pretrained("openbmb/VoxCPM2", device=device)
    except Exception as e:
        print(f"ERROR: Could not load VoxCPM2: {e}", file=sys.stderr)
        sys.exit(1)
    print("[Model 2] VoxCPM2 loaded.")

    # ── Preprocess reference to 16 kHz mono WAV ───────────────────────────────
    print("[Model 2] Preprocessing reference audio to 16 kHz mono...")
    try:
        ref_wav, _ = librosa.load(args.ref_audio, sr=16000, mono=True)
    except Exception as e:
        print(f"ERROR: Could not load reference audio: {e}", file=sys.stderr)
        sys.exit(1)

    ref_wav, _ = librosa.effects.trim(ref_wav, top_db=30)
    peak = float(np.max(np.abs(ref_wav))) if ref_wav.size else 0.0
    if peak > 0:
        ref_wav = (ref_wav / peak) * 0.95
    ref_wav = ref_wav.astype(np.float32)
    print(f"[Model 2] Reference duration: {len(ref_wav) / 16000:.1f}s @ 16 kHz")

    # Write preprocessed reference to a temp WAV file
    tmp_ref_path = os.path.join(
        tempfile.gettempdir(), "_voxcpm2_ref.wav"
    )
    sf.write(tmp_ref_path, ref_wav, 16000)
    print(f"[Model 2] Preprocessed reference saved to: {tmp_ref_path}")

    # ── Generate ──────────────────────────────────────────────────────────────
    gen_kwargs = dict(
        text=args.target_text,
        reference_wav_path=tmp_ref_path,
        cfg_value=args.cfg_value,
        inference_timesteps=args.timesteps,
    )
    if args.ref_transcript.strip():
        gen_kwargs["prompt_wav_path"] = tmp_ref_path
        gen_kwargs["prompt_text"] = args.ref_transcript

    print(f"[Model 2] Generating with cfg={args.cfg_value}, steps={args.timesteps}...")
    try:
        audio = model.generate(**gen_kwargs)
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        print("ERROR: CUDA OOM during generation.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Generation failed: {e}", file=sys.stderr)
        sys.exit(1)

    audio_np = np.asarray(audio, dtype=np.float32).squeeze()
    try:
        sample_rate = int(model.tts_model.sample_rate)
    except AttributeError:
        sample_rate = 48000  # VoxCPM2 default
    duration = len(audio_np) / sample_rate
    print(f"[Model 2] Generated {duration:.1f}s @ {sample_rate} Hz.")

    # ── Save output ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sf.write(args.output, audio_np, sample_rate)
    print(f"[Model 2] Saved: {args.output}  ({duration:.1f}s @ {sample_rate} Hz)")
    print(f"__OUTPUT_SAMPLE_RATE__ {sample_rate}")

    # Clean up temp file
    try:
        os.remove(tmp_ref_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
