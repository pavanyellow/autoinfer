# AutoInfer

**Autonomous inference optimization for ASR models, inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).**

Give an AI agent a speech recognition model and a benchmark suite. Let it experiment overnight. Wake up to a faster model.

## How It Works

The same loop as autoresearch, but for **inference** instead of training:

```
┌─────────────────────────────────────────────────┐
│  Agent modifies infer.py                        │
│       ↓                                         │
│  bench.py measures: WER, latency, throughput    │
│       ↓                                         │
│  Quality guard: WER degradation < 0.02?         │
│       ↓              ↓                          │
│     YES → KEEP     NO → REVERT                  │
│       ↓              ↓                          │
│  Commit + log    git checkout infer.py          │
│       ↓              ↓                          │
│  ←──── next experiment ────→                    │
└─────────────────────────────────────────────────┘
```

## Files

| File | Description | Agent modifies? |
|---|---|---|
| `prepare.py` | Downloads model & test audio, installs deps | No |
| `infer.py` | Inference code — the agent's playground | **Yes** |
| `bench.py` | Benchmark harness (WER, latency, throughput, memory) | No |
| `program.md` | Research directions & agent instructions | No |

## Quick Start

```bash
# 1. Setup (one-time)
python prepare.py

# 2. Establish baseline
python bench.py --baseline --experiment baseline

# 3. Initialize git tracking
git init && git add -A && git commit -m "baseline: initial Qwen3-ASR-0.6B inference"

# 4. Let the agent loose
# Point your favorite AI coding agent at program.md and let it run
```

## Running with Different Agents

### Claude Code
```bash
claude "Read program.md and follow the experiment loop. Start from the baseline and run experiments autonomously."
```

### Cursor / Aider / etc.
Point the agent at `program.md` as its system instruction, and let it iterate on `infer.py`.

## Metrics

| Metric | What it measures | Goal |
|---|---|---|
| **WER** | Word Error Rate on LibriSpeech test-clean | Stay within +0.02 of baseline |
| **Latency** | Average ms per utterance | ↓ Lower is better |
| **Throughput** | Real-time factor (audio_sec / wall_sec) | ↑ Higher is better |
| **Memory** | Peak GPU VRAM usage | ↓ Lower enables more concurrency |

## Configuration

Edit constants in `prepare.py` before running:

- `MODEL_NAME`: Change to `Qwen/Qwen3-ASR-1.7B` for the larger model
- `NUM_EVAL_SAMPLES`: Increase for more thorough benchmarks (slower)
- `WER_DEGRADATION_THRESHOLD`: Tighten (0.01) or loosen (0.05) the quality guard

## Results

All experiments are logged to `~/.cache/autoinfer/results.tsv`:

```
experiment_id  timestamp              wer     latency_avg_ms  throughput_rtfx  memory_peak_mb  status  notes
baseline       2026-03-08T10:00:00    0.0312  45.2            12.5             1842.3          OK      BASELINE
exp-001        2026-03-08T10:05:00    0.0315  38.7            15.1             1844.1          OK      KEEP: flash_attention_2
exp-002        2026-03-08T10:10:00    0.0489  32.1            18.3             1201.5          OK      REVERT: INT4 broke WER
```

## Adapting to Other Models

Swap out `infer.py` and update `prepare.py` constants. The bench/program framework is model-agnostic. Works with:
- Whisper (any size)
- NVIDIA Parakeet / Canary
- Wav2Vec2 / HuBERT
- Any model with a `transcribe(paths) -> texts` interface

## License

MIT
