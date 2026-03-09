"""
Simple data prep: download LibriSpeech test-clean directly via torchaudio.
Avoids the HuggingFace datasets library entirely.
Creates the same manifest.json format that bench.py expects.
"""
import os
import json
import torchaudio
from prepare import AUDIO_DIR, MANIFEST_PATH, NUM_EVAL_SAMPLES, SAMPLE_RATE, MAX_AUDIO_DURATION_SEC

os.makedirs(AUDIO_DIR, exist_ok=True)

print("Downloading LibriSpeech test-clean via torchaudio...")
ds = torchaudio.datasets.LIBRISPEECH(
    root="/workspace/.cache/autoinfer",
    url="test-clean",
    download=True,
)

manifest = []
count = 0
for i in range(len(ds)):
    if count >= NUM_EVAL_SAMPLES:
        break
    waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = ds[i]
    duration = waveform.shape[1] / sample_rate
    if duration > MAX_AUDIO_DURATION_SEC:
        continue

    # Resample if needed
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)

    # Save as wav
    audio_path = os.path.join(AUDIO_DIR, f"sample_{count:04d}.wav")
    torchaudio.save(audio_path, waveform, SAMPLE_RATE)

    manifest.append({
        "id": f"sample_{count:04d}",
        "audio_path": audio_path,
        "reference": transcript.lower(),
        "duration_sec": round(duration, 2),
    })
    count += 1

with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Saved {len(manifest)} samples to {AUDIO_DIR}")
print(f"Manifest: {MANIFEST_PATH}")
