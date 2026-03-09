"""
infer.py - Inference code for Qwen3-ASR (AGENT MODIFIES THIS FILE)

This is the ONLY file the AI agent is allowed to modify.
Everything is fair game: model loading, preprocessing, decoding strategy,
quantization, batching, caching, attention implementation, etc.

The agent's goal is to make transcribe() as fast as possible
while keeping WER within the allowed degradation threshold.

CONTRACT:
- transcribe(audio_paths: list[str]) -> list[str]
  Takes a list of audio file paths, returns a list of transcription strings.
- The function signature MUST NOT change.
- Must work on a single NVIDIA GPU.
- No new pip dependencies without justification in commit message.
"""

import os
import re
import sys
import torch
import numpy as np
from typing import Optional

# ---- Import prepare.py constants ----
sys.path.insert(0, os.path.dirname(__file__))
from prepare import MODEL_DIR, MODEL_NAME, SAMPLE_RATE

# ============================================================
# MODEL LOADING (agent can modify everything below)
# ============================================================

# Global model reference - loaded once, reused across calls
_model = None
_processor = None


def load_model():
    """Load the Qwen3-ASR model. Called once at startup."""
    global _model, _processor

    if _model is not None:
        return

    from qwen_asr import Qwen3ASRModel

    _model = Qwen3ASRModel.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=32,
        attn_implementation="flash_attention_2",
    )

    print(f"Model loaded: {MODEL_NAME}")


def preprocess_audio(audio_path: str) -> str:
    """
    Preprocess a single audio file before inference.
    Agent can add resampling, normalization, chunking, etc.
    Returns the path (or modified path) to feed into the model.
    """
    # Baseline: no preprocessing, pass path directly
    return audio_path


def transcribe(audio_paths: list[str]) -> list[str]:
    """
    Transcribe a batch of audio files.

    Args:
        audio_paths: List of paths to .wav files (16kHz mono)

    Returns:
        List of transcription strings (lowercase, no punctuation)

    RULES:
    - This function signature MUST NOT change
    - Must return one transcription per input audio
    - Must work on a single GPU
    - Output should be lowercase for WER comparison
    """
    load_model()

    # Preprocess all audio
    processed_paths = [preprocess_audio(p) for p in audio_paths]

    # Run inference — force English to skip language detection
    results = _model.transcribe(processed_paths, language="English")

    # Post-process: normalize text for WER comparison
    transcriptions = []
    for result in results:
        text = result.text if hasattr(result, "text") else str(result)
        text = text.lower().strip()
        # Remove punctuation (model outputs commas, periods, dashes, etc.)
        text = re.sub(r"[^\w\s']", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        transcriptions.append(text)

    return transcriptions


# ============================================================
# OPTIMIZATION IDEAS (for the agent to explore):
# ============================================================
# 1. Quantization: INT8, INT4, GPTQ, AWQ, bitsandbytes
# 2. Attention: flash_attention_2, sdpa, custom kernels
# 3. Batching: dynamic batching by audio length, chunked prefill
# 4. Caching: KV-cache optimization, prefix caching
# 5. Model compilation: torch.compile(), ONNX export, TensorRT
# 6. Audio preprocessing: downsampling, VAD trimming, chunking
# 7. Decoding: greedy vs beam search tradeoffs, speculative decoding
# 8. Mixed precision: fp16 vs bf16 vs fp8
# 9. Memory: gradient checkpointing off, memory-efficient attention
# 10. vLLM backend: switch from transformers to vLLM for serving
# ============================================================


if __name__ == "__main__":
    # Quick smoke test
    import sys
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        # Use first sample from manifest
        import json
        from prepare import MANIFEST_PATH
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        paths = [manifest[0]["audio_path"]]
        print(f"Reference: {manifest[0]['reference']}")

    results = transcribe(paths)
    for p, t in zip(paths, results):
        print(f"  {os.path.basename(p)}: {t}")
