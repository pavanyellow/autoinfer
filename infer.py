"""
infer.py - Inference code for Qwen3-ASR (AGENT MODIFIES THIS FILE)

CONTRACT:
- transcribe(audio_paths: list[str]) -> list[str]
- Must work on a single NVIDIA GPU.
"""

import os
import re
import sys
import torch
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from prepare import MODEL_DIR, MODEL_NAME, SAMPLE_RATE

# Global references - loaded once
_model = None
_processor = None
_prompt_template = None


def load_model():
    """Load model and processor directly, bypassing Qwen3ASRModel wrapper."""
    global _model, _processor, _prompt_template

    if _model is not None:
        return

    from transformers import AutoModel, AutoProcessor
    # Register qwen3_asr model type with transformers
    import qwen_asr  # noqa: F401

    _model = AutoModel.from_pretrained(
        MODEL_DIR,
        dtype=torch.float16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    )
    _model.eval()

    _processor = AutoProcessor.from_pretrained(MODEL_DIR, fix_mistral_regex=True)

    # Pre-build the prompt template once (force English, skip language detection)
    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    base = _processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    _prompt_template = base + "language English<asr_text>"

    print(f"Model loaded: {MODEL_NAME} (direct)")


def transcribe(audio_paths: list[str]) -> list[str]:
    load_model()

    # Load audio files as numpy arrays
    wavs = []
    for p in audio_paths:
        audio, sr = sf.read(p, dtype="float32")
        if sr != SAMPLE_RATE:
            import torchaudio
            audio = torch.from_numpy(audio).unsqueeze(0)
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE).squeeze(0).numpy()
        wavs.append(audio)

    # Build prompts (same template for all)
    texts = [_prompt_template] * len(wavs)

    # Process and run inference
    inputs = _processor(text=texts, audio=wavs, return_tensors="pt", padding=True)
    inputs = inputs.to(_model.device).to(_model.dtype)

    with torch.no_grad():
        output_ids = _model.generate(**inputs, max_new_tokens=512)

    decoded = _processor.batch_decode(
        output_ids.sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Normalize text
    transcriptions = []
    for text in decoded:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        transcriptions.append(text)

    return transcriptions


if __name__ == "__main__":
    import json
    from prepare import MANIFEST_PATH
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        paths = [manifest[0]["audio_path"]]
        print(f"Reference: {manifest[0]['reference']}")

    results = transcribe(paths)
    for p, t in zip(paths, results):
        print(f"  {os.path.basename(p)}: {t}")
