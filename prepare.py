"""
prepare.py - One-time setup for autoinfer (DO NOT MODIFY)

Downloads the Qwen3-ASR model, test audio dataset, and verifies dependencies.
This file is READ-ONLY for the agent. Only the human should modify this.
"""

import os
import sys
import json
import subprocess
import hashlib
from pathlib import Path

# ---- Configuration ----
CACHE_DIR = os.path.expanduser("~/.cache/autoinfer")
MODEL_NAME = "Qwen/Qwen3-ASR-0.6B"  # smaller model for faster iteration; swap to 1.7B for final runs
DATASET_NAME = "librispeech_test_clean"  # standard ASR benchmark
NUM_EVAL_SAMPLES = 100  # number of audio samples for benchmarking
SAMPLE_RATE = 16000
MAX_AUDIO_DURATION_SEC = 30  # cap audio length for consistent benchmarking
WER_DEGRADATION_THRESHOLD = 0.02  # max allowed WER increase vs baseline (absolute)

# ---- Published Targets (from Qwen3-ASR paper, arXiv:2601.21337) ----
# These are the official numbers using vLLM v0.14.0 + CUDA Graph + bf16
# on ~2min audio, single GPU. Our goal is to match or beat these.
PAPER_TARGETS = {
    "0.6B": {
        # Table 2: Inference efficiency at concurrency=1 (single request)
        "rtf": 0.00923,               # real-time factor (lower = faster)
        "throughput": 108.34,          # audio seconds processed per wall second
        "ttft_avg_ms": 92,             # time to first token, average
        "ttft_p95_ms": 105,            # time to first token, 95th percentile
        # Table 3: WER on LibriSpeech test (our eval set)
        "wer_librispeech_clean": 0.0211,
        "wer_librispeech_other": 0.0455,
        "wer_fleurs_en": 0.0439,
        # At higher concurrency (for reference)
        "throughput_conc8": 500.0,     # 8 concurrent requests
        "throughput_conc128": 2000.0,  # 128 concurrent → 2000 sec audio/sec
    },
    "1.7B": {
        "rtf": 0.01482,
        "throughput": 67.48,
        "ttft_avg_ms": 102,
        "ttft_p95_ms": 113,
        "wer_librispeech_clean": 0.0163,
        "wer_librispeech_other": 0.0338,
        "wer_fleurs_en": 0.0335,
    },
}

# ---- Paths ----
MODEL_DIR = os.path.join(CACHE_DIR, "model")
AUDIO_DIR = os.path.join(CACHE_DIR, "audio")
MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.json")
BASELINE_PATH = os.path.join(CACHE_DIR, "baseline.json")
RESULTS_PATH = os.path.join(CACHE_DIR, "results.tsv")


def install_dependencies():
    """Install required packages."""
    deps = [
        "torch",
        "torchaudio",
        "transformers>=4.40",
        "datasets",
        "jiwer",        # WER computation
        "soundfile",
        "librosa",
        "numpy",
    ]
    print("Installing dependencies...")
    for dep in deps:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", dep])

    # Install qwen-asr separately (may need special handling)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "qwen-asr"])
    print("Dependencies installed.")


def download_model():
    """Download and cache the Qwen3-ASR model."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Downloading model: {MODEL_NAME}...")
    from transformers import AutoModel, AutoTokenizer

    # The qwen_asr package handles model loading, but we pre-download weights
    from huggingface_hub import snapshot_download
    snapshot_download(
        MODEL_NAME,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"Model cached at: {MODEL_DIR}")


def download_audio():
    """Download LibriSpeech test-clean samples and create a manifest."""
    os.makedirs(AUDIO_DIR, exist_ok=True)

    print(f"Downloading {DATASET_NAME} ({NUM_EVAL_SAMPLES} samples)...")
    from datasets import load_dataset
    import soundfile as sf

    ds = load_dataset(
        "librispeech_asr",
        "clean",
        split="test",
        trust_remote_code=True,
    )

    manifest = []
    count = 0
    for item in ds:
        if count >= NUM_EVAL_SAMPLES:
            break

        audio = item["audio"]
        duration = len(audio["array"]) / audio["sampling_rate"]

        if duration > MAX_AUDIO_DURATION_SEC:
            continue

        # Save audio file
        audio_path = os.path.join(AUDIO_DIR, f"sample_{count:04d}.wav")
        sf.write(audio_path, audio["array"], audio["sampling_rate"])

        manifest.append({
            "id": f"sample_{count:04d}",
            "audio_path": audio_path,
            "reference": item["text"].lower(),  # normalize to lowercase
            "duration_sec": round(duration, 2),
        })
        count += 1

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {len(manifest)} audio samples to {AUDIO_DIR}")
    print(f"Manifest written to {MANIFEST_PATH}")
    return manifest


def init_results_log():
    """Initialize the results TSV file."""
    if not os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "w") as f:
            f.write("experiment_id\ttimestamp\twer\tlatency_avg_ms\tlatency_p95_ms\tthroughput_rtf\tmemory_peak_mb\tstatus\tnotes\n")
        print(f"Results log initialized at {RESULTS_PATH}")
    else:
        print(f"Results log already exists at {RESULTS_PATH}")


def print_config():
    """Print configuration for the agent to read."""
    print("\n" + "=" * 60)
    print("AUTOINFER CONFIGURATION")
    print("=" * 60)
    print(f"  Model:              {MODEL_NAME}")
    print(f"  Dataset:            {DATASET_NAME}")
    print(f"  Eval samples:       {NUM_EVAL_SAMPLES}")
    print(f"  Sample rate:        {SAMPLE_RATE} Hz")
    print(f"  Max audio duration: {MAX_AUDIO_DURATION_SEC}s")
    print(f"  WER threshold:      +{WER_DEGRADATION_THRESHOLD} (absolute)")
    print(f"  Cache dir:          {CACHE_DIR}")
    print(f"  Model dir:          {MODEL_DIR}")
    print(f"  Audio dir:          {AUDIO_DIR}")
    print(f"  Manifest:           {MANIFEST_PATH}")
    print(f"  Results log:        {RESULTS_PATH}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("=" * 60)
    print("AUTOINFER - Preparing environment")
    print("=" * 60)

    install_dependencies()
    download_model()
    download_audio()
    init_results_log()
    print_config()

    print("\nSetup complete! You can now run experiments with:")
    print("  python bench.py          # run a single benchmark")
    print("  # or let the agent iterate on infer.py autonomously")
