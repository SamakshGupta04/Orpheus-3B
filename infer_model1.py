#!/usr/bin/env python3
"""
Model 1 (Orpheus 3B) voice cloning inference script.

Called as a subprocess by app.py so that VRAM is freed on exit and deps are
isolated from the other models.

Usage:
    python infer_model1.py \
        --ref-audio  reference.wav \
        --target-text "Say this sentence." \
        --output  outputs/live/out.wav \
        [--ref-transcript "Words spoken in the reference clip."] \
        [--max-new-tokens 1200]

Exit 0 on success, 1 on failure.
Prints progress to stdout. Emits "__OUTPUT_SAMPLE_RATE__ 24000" before exit.
"""

import argparse
import sys
import os
import gc
import warnings

warnings.filterwarnings("ignore")


def _dry_run(output_path: str, sample_rate: int = 24000, duration: float = 2.0) -> None:
    """Write a 440 Hz test tone to output_path without loading any model."""
    import numpy as np
    import soundfile as sf
    print("[Model 1] DRY RUN — skipping model loading, generating test tone.")
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sf.write(output_path, audio, sample_rate)
    print(f"[Model 1] DRY RUN saved: {output_path}  ({duration:.1f}s @ {sample_rate} Hz)")
    print(f"__OUTPUT_SAMPLE_RATE__ {sample_rate}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref-audio", required=True)
    p.add_argument("--ref-transcript", default="")
    p.add_argument("--target-text", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-new-tokens", type=int, default=1200)
    p.add_argument("--dry-run", action="store_true",
                   help="Skip model loading; write a test tone to --output instead.")
    args = p.parse_args()

    if args.dry_run:
        _dry_run(args.output, sample_rate=24000)
        return

    print(f"[Model 1] Reference audio : {args.ref_audio}")
    print(f"[Model 1] Target text     : {args.target_text[:80]}...")
    if args.ref_transcript:
        print(f"[Model 1] Transcript      : {args.ref_transcript[:80]}...")
    else:
        print("[Model 1] No reference transcript provided (quality may vary).")

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[Model 1] WARNING: No CUDA GPU detected. Inference will be very slow.")
    else:
        print(f"[Model 1] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Model 1] Device: {device}")

    # ── Load Orpheus model ────────────────────────────────────────────────────
    print("[Model 1] Loading Orpheus 3B (this may take a few minutes on first run)...")
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: unsloth not installed. Run: pip install unsloth", file=sys.stderr)
        sys.exit(1)

    MODEL_NAME = "unsloth/orpheus-3b-0.1-pretrained"
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
        print(f"ERROR: Could not load model: {last_err}", file=sys.stderr)
        sys.exit(1)

    try:
        from unsloth import FastLanguageModel as _FLM
        _FLM.for_inference(model)
    except Exception:
        pass
    model.eval()
    print("[Model 1] Orpheus loaded.")

    # ── Load SNAC codec ───────────────────────────────────────────────────────
    try:
        from snac import SNAC
    except ImportError:
        print("ERROR: snac not installed. Run: pip install snac", file=sys.stderr)
        sys.exit(1)

    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
    print("[Model 1] SNAC codec loaded.")

    # ── Preprocess reference audio ────────────────────────────────────────────
    print("[Model 1] Preprocessing reference audio...")
    try:
        ref_wav, _ = librosa.load(args.ref_audio, sr=24000, mono=True)
    except Exception as e:
        print(f"ERROR: Could not load reference audio: {e}", file=sys.stderr)
        sys.exit(1)

    ref_wav, _ = librosa.effects.trim(ref_wav, top_db=30)
    peak = float(np.max(np.abs(ref_wav))) if ref_wav.size else 0.0
    if peak > 0:
        ref_wav = (ref_wav / peak) * 0.95
    ref_wav = ref_wav.astype(np.float32)
    print(f"[Model 1] Reference duration: {len(ref_wav) / 24000:.1f}s @ 24 kHz")

    # ── Encode reference with SNAC ────────────────────────────────────────────
    print("[Model 1] Encoding reference audio into SNAC tokens...")
    wav_t = torch.from_numpy(ref_wav).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        codes = snac_model.encode(wav_t)

    ref_codes: list[int] = []
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
    print(f"[Model 1] Encoded {len(ref_codes)} tokens ({len(ref_codes) // 7} frames).")

    # ── Build cloning prompt ──────────────────────────────────────────────────
    # Orpheus special tokens (verbatim from the official Unsloth notebook).
    END_OF_TEXT     = 128009
    START_OF_SPEECH = 128257
    END_OF_SPEECH   = 128258
    START_OF_HUMAN  = 128259
    END_OF_HUMAN    = 128260
    START_OF_AI     = 128261
    END_OF_AI       = 128262

    ref_transcript = args.ref_transcript or args.target_text  # fallback
    ref_ids = tokenizer.encode(ref_transcript, add_special_tokens=True)
    ref_ids.append(END_OF_TEXT)
    tgt_ids = tokenizer.encode(args.target_text, add_special_tokens=True)
    tgt_ids.append(END_OF_TEXT)

    prompt_ids = (
        [START_OF_HUMAN] + ref_ids + [END_OF_HUMAN]
        + [START_OF_AI, START_OF_SPEECH] + ref_codes + [END_OF_SPEECH, END_OF_AI]
        + [START_OF_HUMAN] + tgt_ids + [END_OF_HUMAN]
        + [START_OF_AI, START_OF_SPEECH]
    )

    input_ids = torch.tensor([prompt_ids], dtype=torch.int64).to(device)
    am = torch.ones_like(input_ids)
    print(f"[Model 1] Prompt length: {input_ids.shape[1]} tokens. Generating...")

    # ── Generate ──────────────────────────────────────────────────────────────
    try:
        with torch.inference_mode():
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=am,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.1,
                eos_token_id=END_OF_SPEECH,
                use_cache=True,
            )
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        print("ERROR: CUDA OOM during generation. Use a shorter reference or text.", file=sys.stderr)
        sys.exit(1)

    print("[Model 1] Generation complete. Parsing tokens...")

    # ── Parse generated tokens ────────────────────────────────────────────────
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
        print("ERROR: No audio tokens generated. Try a different reference or text.", file=sys.stderr)
        sys.exit(1)

    print(f"[Model 1] Decoded {len(code_list)} audio tokens ({len(code_list) // 7} frames).")

    # ── Decode SNAC tokens to waveform ────────────────────────────────────────
    l1, l2, l3 = [], [], []
    for i in range(len(code_list) // 7):
        l1.append(code_list[7 * i])
        l2.append(code_list[7 * i + 1] - 4096)
        l3.append(code_list[7 * i + 2] - 8192)
        l3.append(code_list[7 * i + 3] - 12288)
        l2.append(code_list[7 * i + 4] - 16384)
        l3.append(code_list[7 * i + 5] - 20480)
        l3.append(code_list[7 * i + 6] - 24576)

    # Move SNAC to CPU for decoding to free VRAM
    snac_model.to("cpu")
    if device == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    c = [
        torch.tensor(l1).unsqueeze(0),
        torch.tensor(l2).unsqueeze(0),
        torch.tensor(l3).unsqueeze(0),
    ]
    with torch.inference_mode():
        audio_hat = snac_model.decode(c)

    audio_np = audio_hat.detach().squeeze().cpu().numpy().astype(np.float32)
    duration = len(audio_np) / 24000

    # ── Save output ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sf.write(args.output, audio_np, 24000)
    print(f"[Model 1] Saved: {args.output}  ({duration:.1f}s @ 24000 Hz)")
    print("__OUTPUT_SAMPLE_RATE__ 24000")


if __name__ == "__main__":
    main()
