"""
infer.py - Inference code for Qwen3-ASR (AGENT MODIFIES THIS FILE)

CONTRACT:
- transcribe(audio_paths: list[str]) -> list[str]
- Must work on a single NVIDIA GPU.
"""

import os
import re
import sys
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from prepare import MODEL_DIR, MODEL_NAME, SAMPLE_RATE

# Global references - loaded once
_model = None
_processor = None
_sampling_params = None
_prompt_template = None


def load_model():
    """Load model via vLLM backend for optimized inference."""
    global _model, _processor, _sampling_params, _prompt_template

    if _model is not None:
        return

    from vllm import LLM, SamplingParams
    from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

    _model = LLM(
        model=MODEL_DIR,
        dtype="float16",
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        scheduling_policy="priority",
    )

    _processor = Qwen3ASRProcessor.from_pretrained(MODEL_DIR, fix_mistral_regex=True)
    _sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    # Pre-build the prompt template (force English)
    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    base = _processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    _prompt_template = base + "language English<asr_text>"

    print(f"Model loaded: {MODEL_NAME} (vLLM)")


def transcribe(audio_paths: list[str]) -> list[str]:
    load_model()

    # Load audio in parallel
    from concurrent.futures import ThreadPoolExecutor
    def _load(p):
        audio, sr = sf.read(p, dtype="float32")
        if sr != SAMPLE_RATE:
            import torchaudio, torch
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio).unsqueeze(0), sr, SAMPLE_RATE
            ).squeeze(0).numpy()
        return audio
    with ThreadPoolExecutor(max_workers=8) as pool:
        audios = list(pool.map(_load, audio_paths))

    # Sort by audio length (shorter first) for better vLLM scheduling
    order = sorted(range(len(audios)), key=lambda i: len(audios[i]))
    inputs = [
        {"prompt": _prompt_template, "multi_modal_data": {"audio": [audios[i]]}}
        for i in order
    ]

    # Run vLLM inference
    outputs = _model.generate(inputs, sampling_params=_sampling_params, use_tqdm=False)

    # Unsort and normalize text
    results = [""] * len(audios)
    for idx, o in zip(order, outputs):
        text = o.outputs[0].text.lower().strip()
        text = re.sub(r"[^\w\s']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        results[idx] = text

    return results


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
