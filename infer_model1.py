#!/usr/bin/env python3
"""
Model 1 (Orpheus 3B) voice cloning inference.

Two modes:

1. One-shot CLI (batch / smoke tests) — loads the model, runs one inference,
   exits. VRAM is freed on exit.

       python infer_model1.py \
           --ref-audio  reference.wav \
           --target-text "Say this sentence." \
           --output  outputs/live/out.wav \
           [--ref-transcript "Words spoken in the reference clip."] \
           [--max-new-tokens 1200]

   Exit 0 on success, 1 on failure. Emits "__OUTPUT_SAMPLE_RATE__ 24000"
   before exit.

2. Persistent server (used by app.py so the model stays warm across
   requests) — loads once, then serves inference requests over HTTP until
   told to shut down.

       python infer_model1.py --serve --port 8001 [--dry-run]

   See model_server.py for the wire protocol.
"""

import argparse
import sys
import os
import gc
import warnings

warnings.filterwarnings("ignore")

MODEL_NAME = "unsloth/orpheus-3b-0.1-pretrained"
SAMPLE_RATE = 24000

# Orpheus special tokens (verbatim from the official Unsloth notebook).
END_OF_TEXT     = 128009
START_OF_SPEECH = 128257
END_OF_SPEECH   = 128258
START_OF_HUMAN  = 128259
END_OF_HUMAN    = 128260
START_OF_AI     = 128261
END_OF_AI       = 128262


def _dry_run_audio(output_path: str, sample_rate: int = SAMPLE_RATE, duration: float = 2.0):
    """Write a 440 Hz test tone to output_path without loading any model."""
    import numpy as np
    import soundfile as sf
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sf.write(output_path, audio, sample_rate)
    return sample_rate, duration


def load_model(log, dry_run: bool = False) -> dict:
    """Load Orpheus + SNAC once. Returns an objs dict consumed by run_inference."""
    if dry_run:
        log("[Model 1] DRY RUN mode — skipping model loading, will generate test tones.")
        return {"dry_run": True}

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log("[Model 1] WARNING: No CUDA GPU detected. Inference will be very slow.")
    else:
        log(f"[Model 1] GPU: {torch.cuda.get_device_name(0)}")
    log(f"[Model 1] Device: {device}")

    log("[Model 1] Loading Orpheus 3B (this may take a few minutes on first run)...")
    try:
        import unsloth  # noqa: F401
    except ImportError as e:
        raise RuntimeError("unsloth not installed. Run: pip install unsloth") from e

    model = tokenizer = None
    last_err = None
    for loader_name in ("FastLanguageModel", "FastModel"):
        try:
            import unsloth as _uns
            loader = getattr(_uns, loader_name, None)
            if loader is None:
                continue
            model, tokenizer = loader.from_pretrained(
                model_name=MODEL_NAME,
                max_seq_length=8192,
                dtype=None,
                load_in_4bit=(device == "cuda"),
            )
            break
        except Exception as e:
            last_err = e
            continue

    if model is None:
        raise RuntimeError(f"Could not load model: {last_err}")

    try:
        from unsloth import FastLanguageModel as _FLM
        _FLM.for_inference(model)
    except Exception:
        pass
    model.eval()
    log("[Model 1] Orpheus loaded.")

    try:
        from snac import SNAC
    except ImportError as e:
        raise RuntimeError("snac not installed. Run: pip install snac") from e

    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
    log("[Model 1] SNAC codec loaded.")

    return {
        "dry_run": False,
        "model": model,
        "tokenizer": tokenizer,
        "snac_model": snac_model,
        "device": device,
    }


def run_inference(objs: dict, payload: dict, log) -> dict:
    """Run one cloning request.

    payload keys: ref_audio, ref_transcript (optional), target_text, output,
    max_new_tokens (optional).
    Returns {"output_path", "sample_rate", "duration"}.
    """
    output_path = payload["output"]

    if objs.get("dry_run"):
        sr, duration = _dry_run_audio(output_path, sample_rate=SAMPLE_RATE)
        log(f"[Model 1] DRY RUN saved: {output_path}  ({duration:.1f}s @ {sr} Hz)")
        return {"output_path": output_path, "sample_rate": sr, "duration": duration}

    ref_audio = payload["ref_audio"]
    ref_transcript = payload.get("ref_transcript") or ""
    target_text = payload["target_text"]
    max_new_tokens = int(payload.get("max_new_tokens") or 1200)

    log(f"[Model 1] Reference audio : {ref_audio}")
    log(f"[Model 1] Target text     : {target_text[:80]}...")
    if ref_transcript:
        log(f"[Model 1] Transcript      : {ref_transcript[:80]}...")
    else:
        log("[Model 1] No reference transcript provided (quality may vary).")

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
    tokenizer = objs["tokenizer"]
    snac_model = objs["snac_model"]
    device = objs["device"]

    # ── Preprocess reference audio ────────────────────────────────────────
    log("[Model 1] Preprocessing reference audio...")
    try:
        ref_wav, _ = librosa.load(ref_audio, sr=24000, mono=True)
    except Exception as e:
        raise RuntimeError(f"Could not load reference audio: {e}") from e

    ref_wav, _ = librosa.effects.trim(ref_wav, top_db=30)
    peak = float(np.max(np.abs(ref_wav))) if ref_wav.size else 0.0
    if peak > 0:
        ref_wav = (ref_wav / peak) * 0.95
    ref_wav = ref_wav.astype(np.float32)
    log(f"[Model 1] Reference duration: {len(ref_wav) / 24000:.1f}s @ 24 kHz")

    # ── Encode reference with SNAC ────────────────────────────────────────
    log("[Model 1] Encoding reference audio into SNAC tokens...")
    wav_t = torch.from_numpy(ref_wav).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        codes = snac_model.encode(wav_t)

    ref_codes: list = []
    for i in range(codes[0].shape[1]):
        ref_codes += [
            codes[0][0][i].item()           + 128266,
            codes[1][0][2 * i].item()       + 128266 + 4096,
            codes[2][0][4 * i].item()       + 128266 + 8192,
            codes[2][0][(4 * i) + 1].item() + 128266 + 12288,
            codes[1][0][(2 * i) + 1].item() + 128266 + 16384,
            codes[2][0][(4 * i) + 2].item() + 128266 + 20480,
            codes[2][0][(4 * i) + 3].item() + 128266 + 24576,
        ]
    log(f"[Model 1] Encoded {len(ref_codes)} tokens ({len(ref_codes) // 7} frames).")

    # ── Build cloning prompt ──────────────────────────────────────────────
    ref_transcript_or_fallback = ref_transcript or target_text
    ref_ids = tokenizer.encode(ref_transcript_or_fallback, add_special_tokens=True)
    ref_ids.append(END_OF_TEXT)
    tgt_ids = tokenizer.encode(target_text, add_special_tokens=True)
    tgt_ids.append(END_OF_TEXT)

    prompt_ids = (
        [START_OF_HUMAN] + ref_ids + [END_OF_HUMAN]
        + [START_OF_AI, START_OF_SPEECH] + ref_codes + [END_OF_SPEECH, END_OF_AI]
        + [START_OF_HUMAN] + tgt_ids + [END_OF_HUMAN]
        + [START_OF_AI, START_OF_SPEECH]
    )

    input_ids = torch.tensor([prompt_ids], dtype=torch.int64).to(device)
    am = torch.ones_like(input_ids)
    log(f"[Model 1] Prompt length: {input_ids.shape[1]} tokens. Generating...")

    # ── Generate ──────────────────────────────────────────────────────────
    try:
        with torch.inference_mode():
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=am,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.1,
                eos_token_id=END_OF_SPEECH,
                use_cache=True,
            )
    except torch.cuda.OutOfMemoryError as e:
        gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError("CUDA OOM during generation. Use a shorter reference or text.") from e

    log("[Model 1] Generation complete. Parsing tokens...")

    # ── Parse generated tokens ─────────────────────────────────────────────
    idx = (gen_ids == START_OF_SPEECH).nonzero(as_tuple=True)
    if len(idx[1]) > 0:
        cropped = gen_ids[:, idx[1][-1].item() + 1:]
    else:
        cropped = gen_ids

    row = cropped[0]
    row = row[row != END_OF_SPEECH]
    row = row[:(row.size(0) // 7) * 7]
    code_list = [t.item() - 128266 for t in row]

    if not code_list:
        raise RuntimeError("No audio tokens generated. Try a different reference or text.")

    log(f"[Model 1] Decoded {len(code_list)} audio tokens ({len(code_list) // 7} frames).")

    # ── Decode SNAC tokens to waveform ─────────────────────────────────────
    l1, l2, l3 = [], [], []
    for i in range(len(code_list) // 7):
        l1.append(code_list[7 * i])
        l2.append(code_list[7 * i + 1] - 4096)
        l3.append(code_list[7 * i + 2] - 8192)
        l3.append(code_list[7 * i + 3] - 12288)
        l2.append(code_list[7 * i + 4] - 16384)
        l3.append(code_list[7 * i + 5] - 20480)
        l3.append(code_list[7 * i + 6] - 24576)

    c = [
        torch.tensor(l1).unsqueeze(0),
        torch.tensor(l2).unsqueeze(0),
        torch.tensor(l3).unsqueeze(0),
    ]
    with torch.inference_mode():
        audio_hat = snac_model.decode(c)

    audio_np = audio_hat.detach().squeeze().cpu().numpy().astype(np.float32)
    duration = len(audio_np) / SAMPLE_RATE

    if device == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    # ── Save output ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sf.write(output_path, audio_np, SAMPLE_RATE)
    log(f"[Model 1] Saved: {output_path}  ({duration:.1f}s @ {SAMPLE_RATE} Hz)")

    return {"output_path": output_path, "sample_rate": SAMPLE_RATE, "duration": duration}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref-audio")
    p.add_argument("--ref-transcript", default="")
    p.add_argument("--target-text")
    p.add_argument("--output")
    p.add_argument("--max-new-tokens", type=int, default=1200)
    p.add_argument("--dry-run", action="store_true",
                   help="Skip model loading; write a test tone to --output instead.")
    p.add_argument("--serve", action="store_true",
                   help="Run as a persistent HTTP inference server instead of one-shot CLI.")
    p.add_argument("--port", type=int, default=int(os.environ.get("MODEL1_PORT", 8001)))
    args = p.parse_args()

    if args.serve:
        from model_server import ModelServer, serve
        server = ModelServer(
            name="Model 1",
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
            "max_new_tokens": args.max_new_tokens,
        }, print)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"__OUTPUT_SAMPLE_RATE__ {result['sample_rate']}")


if __name__ == "__main__":
    main()
