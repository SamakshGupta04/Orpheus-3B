#!/usr/bin/env python3
"""
Model 3 (VibeVoice Hindi 1.5B) voice cloning inference.

NOTE: This model is specialised for Hindi (Devanagari). English text will
produce low-quality audio. Use Hindi text for best results.

Two modes:

1. One-shot CLI (batch / smoke tests) — loads the model, runs one inference,
   exits. VRAM is freed on exit.

       python infer_model3.py \
           --ref-audio  reference.wav \
           --target-text "यह एक हिन्दी वाक्य है।" \
           --output  outputs/live/out.wav \
           [--cfg-scale 1.3]

   Installation (vibevoice is not on PyPI):
       pip install git+https://github.com/vibevoice-community/VibeVoice.git

   Exit 0 on success, 1 on failure. Emits "__OUTPUT_SAMPLE_RATE__ 24000"
   before exit.

2. Persistent server (used by app.py so the model stays warm across
   requests) — loads once, then serves inference requests over HTTP until
   told to shut down.

       python infer_model3.py --serve --port 8003 [--dry-run]

   See model_server.py for the wire protocol.
"""

import argparse
import sys
import os
import gc
import tempfile
import warnings

warnings.filterwarnings("ignore")

MODEL_ID = "tarun7r/vibevoice-hindi-1.5B"
BASE_MODEL_ID = "vibevoice/VibeVoice-1.5B"
DEFAULT_SAMPLE_RATE = 24000


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
    """Load the VibeVoice Hindi processor + model once. Returns an objs dict."""
    if dry_run:
        log("[Model 3] DRY RUN mode — skipping model loading, will generate test tones.")
        return {"dry_run": True}

    import torch

    try:
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "vibevoice package not installed. Install with: "
            "pip install git+https://github.com/vibevoice-community/VibeVoice.git"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log("[Model 3] WARNING: No CUDA GPU detected. Inference will be very slow.")
    else:
        log(f"[Model 3] GPU: {torch.cuda.get_device_name(0)}")
    log(f"[Model 3] Device: {device}")

    log("[Model 3] Loading processor...")
    try:
        processor = VibeVoiceProcessor.from_pretrained(MODEL_ID)
        log(f"[Model 3] Processor loaded from {MODEL_ID}.")
    except Exception:
        log(f"[Model 3] Falling back to base processor {BASE_MODEL_ID}...")
        try:
            processor = VibeVoiceProcessor.from_pretrained(BASE_MODEL_ID)
            log(f"[Model 3] Processor loaded from {BASE_MODEL_ID}.")
        except Exception as e:
            raise RuntimeError(f"Could not load processor: {e}") from e

    log(f"[Model 3] Loading {MODEL_ID} (this may take a few minutes on first run)...")
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
        raise RuntimeError(f"Could not load model: {e}") from e
    model.eval()
    log("[Model 3] Model loaded.")

    return {"dry_run": False, "model": model, "processor": processor, "device": device}


def run_inference(objs: dict, payload: dict, log) -> dict:
    """Run one cloning request.

    payload keys: ref_audio, target_text, output, cfg_scale (optional).
    ref_transcript is accepted but ignored — this model doesn't use it.
    Returns {"output_path", "sample_rate", "duration"}.
    """
    output_path = payload["output"]

    if objs.get("dry_run"):
        sr, duration = _dry_run_audio(output_path, sample_rate=DEFAULT_SAMPLE_RATE)
        log(f"[Model 3] DRY RUN saved: {output_path}  ({duration:.1f}s @ {sr} Hz)")
        return {"output_path": output_path, "sample_rate": sr, "duration": duration}

    ref_audio = payload["ref_audio"]
    target_text = payload["target_text"]
    cfg_scale = float(payload.get("cfg_scale") or 1.3)

    log(f"[Model 3] Reference audio : {ref_audio}")
    log(f"[Model 3] Target text     : {target_text[:80]}...")
    log("[Model 3] NOTE: best results with Hindi (Devanagari) text.")

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
    processor = objs["processor"]
    device = objs["device"]

    # ── Preprocess reference audio ─────────────────────────────────────────
    log("[Model 3] Preprocessing reference audio...")
    try:
        ref_wav, _ = librosa.load(ref_audio, sr=16000, mono=True)
    except Exception as e:
        raise RuntimeError(f"Could not load reference audio: {e}") from e

    ref_wav, _ = librosa.effects.trim(ref_wav, top_db=30)
    peak = float(np.max(np.abs(ref_wav))) if ref_wav.size else 0.0
    if peak > 0:
        ref_wav = (ref_wav / peak) * 0.95
    ref_wav = ref_wav.astype(np.float32)
    log(f"[Model 3] Reference duration: {len(ref_wav) / 16000:.1f}s @ 16 kHz")

    # Write preprocessed reference to a temp WAV file (processor expects a
    # path). Include the PID so two servers/jobs never race on the same path.
    tmp_ref_path = os.path.join(tempfile.gettempdir(), f"_vibevoice_ref_{os.getpid()}.wav")
    sf.write(tmp_ref_path, ref_wav, 16000)

    try:
        # ── Build script and run processor ─────────────────────────────────
        full_script = f"Speaker 1: {target_text}"
        log(f"[Model 3] Script: {full_script[:80]}...")

        try:
            inputs = processor(
                text=[full_script],
                voice_samples=[[tmp_ref_path]],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
        except Exception as e:
            raise RuntimeError(f"Processor failed: {e}") from e

        try:
            inputs = inputs.to(device)
        except Exception:
            inputs = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in inputs.items()
            }

        # ── Generate ──────────────────────────────────────────────────────
        log(f"[Model 3] Generating (cfg_scale={cfg_scale})...")
        try:
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=processor.tokenizer,
                )
        except torch.cuda.OutOfMemoryError as e:
            gc.collect()
            torch.cuda.empty_cache()
            raise RuntimeError("CUDA OOM during generation.") from e
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}") from e

        log("[Model 3] Generation complete.")

        # ── Save output ─────────────────────────────────────────────────────
        speech = outputs.speech_outputs[0]
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            processor.save_audio(speech, output_path=output_path)
            saved_sr = DEFAULT_SAMPLE_RATE
            log(f"[Model 3] Saved via processor: {output_path}")
        except Exception as save_err:
            log(f"[Model 3] Processor save failed ({save_err}), falling back to soundfile...")
            try:
                audio_np = speech.detach().float().cpu().numpy().squeeze()
                sf.write(output_path, audio_np, DEFAULT_SAMPLE_RATE)
                saved_sr = DEFAULT_SAMPLE_RATE
                log(f"[Model 3] Saved via soundfile: {output_path}")
            except Exception as e2:
                raise RuntimeError(f"Could not save output: {e2}") from e2

        # Verify saved file to get actual duration
        duration = None
        try:
            check_wav, check_sr = sf.read(output_path)
            duration = len(check_wav) / check_sr
            saved_sr = check_sr
            log(f"[Model 3] Output: {duration:.1f}s @ {saved_sr} Hz")
        except Exception:
            duration = 0.0
    finally:
        try:
            os.remove(tmp_ref_path)
        except OSError:
            pass

    return {"output_path": output_path, "sample_rate": saved_sr, "duration": duration}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref-audio")
    p.add_argument("--target-text")
    p.add_argument("--output")
    p.add_argument("--cfg-scale", type=float, default=1.3)
    p.add_argument("--dry-run", action="store_true",
                   help="Skip model loading; write a test tone to --output instead.")
    p.add_argument("--serve", action="store_true",
                   help="Run as a persistent HTTP inference server instead of one-shot CLI.")
    p.add_argument("--port", type=int, default=int(os.environ.get("MODEL3_PORT", 8003)))
    args = p.parse_args()

    if args.serve:
        from model_server import ModelServer, serve
        server = ModelServer(
            name="Model 3",
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
            "target_text": args.target_text,
            "output": args.output,
            "cfg_scale": args.cfg_scale,
        }, print)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"__OUTPUT_SAMPLE_RATE__ {result['sample_rate']}")


if __name__ == "__main__":
    main()
