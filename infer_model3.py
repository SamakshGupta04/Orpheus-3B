#!/usr/bin/env python3
"""
Model 3 (VibeVoice Hindi 1.5B) voice cloning inference script.

Called as a subprocess by app.py so that VRAM is freed on exit and deps are
isolated from the other models.

NOTE: This model is specialised for Hindi (Devanagari). English text will
produce low-quality audio. Use Hindi text for best results.

Usage:
    python infer_model3.py \
        --ref-audio  reference.wav \
        --target-text "यह एक हिन्दी वाक्य है।" \
        --output  outputs/live/out.wav \
        [--cfg-scale 1.3]

Installation (vibevoice is not on PyPI):
    pip install git+https://github.com/vibevoice-community/VibeVoice.git

Exit 0 on success, 1 on failure.
Prints progress to stdout. Emits "__OUTPUT_SAMPLE_RATE__ 24000" before exit.
"""

import argparse
import sys
import os
import gc
import tempfile
import warnings

warnings.filterwarnings("ignore")


def _dry_run(output_path: str, sample_rate: int = 24000, duration: float = 2.0) -> None:
    """Write a 440 Hz test tone to output_path without loading any model."""
    import numpy as np
    import soundfile as sf
    print("[Model 3] DRY RUN — skipping model loading, generating test tone.")
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sf.write(output_path, audio, sample_rate)
    print(f"[Model 3] DRY RUN saved: {output_path}  ({duration:.1f}s @ {sample_rate} Hz)")
    print(f"__OUTPUT_SAMPLE_RATE__ {sample_rate}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref-audio", required=True)
    p.add_argument("--target-text", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cfg-scale", type=float, default=1.3)
    p.add_argument("--dry-run", action="store_true",
                   help="Skip model loading; write a test tone to --output instead.")
    args = p.parse_args()

    if args.dry_run:
        _dry_run(args.output, sample_rate=24000)
        return

    print(f"[Model 3] Reference audio : {args.ref_audio}")
    print(f"[Model 3] Target text     : {args.target_text[:80]}...")
    print("[Model 3] NOTE: best results with Hindi (Devanagari) text.")

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
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    except ModuleNotFoundError:
        print(
            "ERROR: vibevoice package not installed.\n"
            "Install with: pip install git+https://github.com/vibevoice-community/VibeVoice.git",
            file=sys.stderr,
        )
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[Model 3] WARNING: No CUDA GPU detected. Inference will be very slow.")
    else:
        print(f"[Model 3] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Model 3] Device: {device}")

    # ── Load processor ────────────────────────────────────────────────────────
    MODEL_ID = "tarun7r/vibevoice-hindi-1.5B"
    BASE_MODEL_ID = "vibevoice/VibeVoice-1.5B"

    print("[Model 3] Loading processor...")
    try:
        processor = VibeVoiceProcessor.from_pretrained(MODEL_ID)
        print(f"[Model 3] Processor loaded from {MODEL_ID}.")
    except Exception:
        print(f"[Model 3] Falling back to base processor {BASE_MODEL_ID}...")
        try:
            processor = VibeVoiceProcessor.from_pretrained(BASE_MODEL_ID)
            print(f"[Model 3] Processor loaded from {BASE_MODEL_ID}.")
        except Exception as e:
            print(f"ERROR: Could not load processor: {e}", file=sys.stderr)
            sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[Model 3] Loading {MODEL_ID} (this may take a few minutes on first run)...")
    try:
        if device == "cuda":
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                attn_implementation="sdpa",
            )
        else:
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                attn_implementation="sdpa",
            )
            model.to(device)
    except Exception as e:
        print(f"ERROR: Could not load model: {e}", file=sys.stderr)
        sys.exit(1)
    model.eval()
    print("[Model 3] Model loaded.")

    # ── Preprocess reference audio ────────────────────────────────────────────
    print("[Model 3] Preprocessing reference audio...")
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
    print(f"[Model 3] Reference duration: {len(ref_wav) / 16000:.1f}s @ 16 kHz")

    # Write preprocessed reference to a temp WAV file (processor expects a path)
    tmp_ref_path = os.path.join(tempfile.gettempdir(), "_vibevoice_ref.wav")
    sf.write(tmp_ref_path, ref_wav, 16000)

    # ── Build script and run processor ───────────────────────────────────────
    full_script = f"Speaker 1: {args.target_text}"
    print(f"[Model 3] Script: {full_script[:80]}...")

    try:
        inputs = processor(
            text=[full_script],
            voice_samples=[[tmp_ref_path]],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
    except Exception as e:
        print(f"ERROR: Processor failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Move inputs to device
    try:
        inputs = inputs.to(device)
    except Exception:
        inputs = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()
        }

    # ── Generate ──────────────────────────────────────────────────────────────
    print(f"[Model 3] Generating (cfg_scale={args.cfg_scale})...")
    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=args.cfg_scale,
                tokenizer=processor.tokenizer,
            )
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        print("ERROR: CUDA OOM during generation.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Generation failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("[Model 3] Generation complete.")

    # ── Save output ───────────────────────────────────────────────────────────
    speech = outputs.speech_outputs[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Try processor's native save method first
    try:
        processor.save_audio(speech, output_path=args.output)
        saved_sr = 24000
        print(f"[Model 3] Saved via processor: {args.output}")
    except Exception as save_err:
        print(f"[Model 3] Processor save failed ({save_err}), falling back to soundfile...")
        try:
            audio_np = speech.detach().float().cpu().numpy().squeeze()
            sf.write(args.output, audio_np, 24000)
            saved_sr = 24000
            print(f"[Model 3] Saved via soundfile: {args.output}")
        except Exception as e2:
            print(f"ERROR: Could not save output: {e2}", file=sys.stderr)
            sys.exit(1)

    # Verify saved file to get actual duration
    try:
        check_wav, check_sr = sf.read(args.output)
        duration = len(check_wav) / check_sr
        saved_sr = check_sr
        print(f"[Model 3] Output: {duration:.1f}s @ {saved_sr} Hz")
    except Exception:
        pass

    print(f"__OUTPUT_SAMPLE_RATE__ {saved_sr}")

    # Clean up temp file
    try:
        os.remove(tmp_ref_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
