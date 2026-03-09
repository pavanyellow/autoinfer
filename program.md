# AutoInfer: Autonomous Inference Optimization for Qwen3-ASR

## Overview

You are an AI research agent. Your task is to **optimize the inference speed** of a Qwen3-ASR speech recognition model while **maintaining transcription quality**. You will iteratively modify `infer.py`, benchmark each change, and keep or revert based on measured results.

This is modeled after [karpathy/autoresearch](https://github.com/karpathy/autoresearch), but applied to **inference optimization** rather than training.

## The Setup

There are four files. Read and understand all of them before starting:

| File | Role | Can you modify it? |
|---|---|---|
| `prepare.py` | Downloads model, audio data, installs deps. Contains constants. | **NO** - read only |
| `infer.py` | Inference code: model loading, preprocessing, `transcribe()` function | **YES** - this is YOUR file |
| `bench.py` | Benchmarking harness: measures WER, latency, throughput, memory | **NO** - read only |
| `program.md` | This file. Your instructions. | **NO** - read only |

## The Rules

1. **Only modify `infer.py`**. The function signature `transcribe(audio_paths: list[str]) -> list[str]` must not change.

2. **Quality guard**: Your changes must not increase WER by more than 0.02 (absolute) vs the baseline. If they do, the experiment is REVERTED. A tiny WER increase is acceptable if the speed gain is substantial.

3. **One change at a time**. Make a single, focused modification per experiment. Don't combine multiple ideas — we need to isolate what works.

4. **Commit before and after**. Use git to track every experiment:
   ```
   git add infer.py && git commit -m "exp-NNN: <description of change>"
   ```

5. **Never stop**. Once you start the experiment loop, do NOT pause to ask the human. They might be asleep. Keep running experiments until manually stopped.

6. **Log everything**. The bench.py harness logs to results.tsv automatically, but also keep notes in your commits about what you tried and why.

## The Experiment Loop

Repeat this loop indefinitely:

```
1. Read the current infer.py
2. Choose an optimization to try (see Research Directions below)
3. Modify infer.py with the change
4. Run: python bench.py --compare --experiment exp-NNN
5. Check the verdict:
   - KEEP → commit the change, update your mental baseline
   - REVERT → git checkout infer.py, try something else
6. Increment experiment counter, go to step 1
```

### First Run (Establishing Baseline)

Before starting the loop, establish the baseline:
```bash
python prepare.py              # one-time setup
python bench.py --baseline     # measure and save baseline metrics
git add -A && git commit -m "baseline: initial Qwen3-ASR inference"
```

## Research Directions

Explore these optimization categories **roughly in this order** (low-hanging fruit first):

### Phase 1: Quick Wins (experiments 1-10)
- **Attention implementation**: Try `flash_attention_2` or `sdpa` in model loading
- **Precision**: Test fp16 vs bf16 vs fp8 (if supported)
- **Batch size tuning**: Find the optimal batch size for the GPU
- **torch.compile()**: Apply compilation with different modes (reduce-overhead, max-autotune)
- **Greedy decoding**: Ensure we're using greedy (not beam search) if quality permits

### Phase 2: Quantization (experiments 10-25)
- **INT8 quantization**: bitsandbytes, GPTQ, or native PyTorch quantization
- **INT4 quantization**: More aggressive, watch WER carefully
- **Mixed quantization**: Different precision for encoder vs decoder
- **AWQ**: Activation-aware weight quantization if available for this model

### Phase 3: Pipeline Optimization (experiments 25-40)
- **Audio preprocessing**: VAD (Voice Activity Detection) to trim silence
- **Dynamic batching**: Sort by audio length, batch similar lengths together
- **Chunked audio processing**: Process long audio in overlapping chunks
- **Prefix caching**: Cache encoder outputs for repeated prefixes
- **Memory layout**: Optimize tensor memory layout (channels_last, contiguous)

### Phase 4: Advanced (experiments 40+)
- **vLLM backend**: Switch from transformers to vLLM for the decoder
- **ONNX export + TensorRT**: Export and optimize the compute graph
- **Speculative decoding**: Use a smaller draft model for the decoder
- **Custom CUDA kernels**: If PyTorch primitives are the bottleneck
- **KV-cache optimization**: Quantized KV-cache, paged attention
- **Model pruning**: Remove attention heads or layers with minimal WER impact

## Decision Criteria

When deciding whether to KEEP an experiment:

| Metric | KEEP if... | REVERT if... |
|---|---|---|
| WER | Δ ≤ +0.02 | Δ > +0.02 |
| Latency | Any decrease | Increase with no throughput gain |
| Throughput | Any increase | Decrease with no latency gain |
| Memory | Decrease is a bonus | Large increase is a concern |

**Speed vs Quality tradeoff**: A 20% speed improvement with 0.01 WER degradation is a GOOD trade. A 2% speed improvement with 0.015 WER degradation is a BAD trade.

## Style Guide

- Keep `infer.py` clean and readable. Remove dead code.
- Add brief comments explaining non-obvious optimizations.
- If an optimization requires a new dependency, note it clearly in the commit.
- Prefer PyTorch-native solutions over third-party libraries when performance is similar.
- A 0.5ms latency improvement that adds 50 lines of hacky code? Probably not worth it.

## Important Notes

- The model is Qwen3-ASR-0.6B (the smaller variant for fast iteration). Findings should generally transfer to the 1.7B model.
- All audio is 16kHz mono WAV, max 30 seconds.
- Benchmarks use 100 LibriSpeech test-clean utterances.
- Peak GPU memory matters — lower memory means we could eventually serve more concurrent requests.
- The evaluation set is fixed. Don't overfit to it — optimizations should be general.

## Getting Started

```bash
# Setup
python prepare.py

# Establish baseline
python bench.py --baseline --experiment baseline
git init && git add -A && git commit -m "initial: baseline Qwen3-ASR inference"

# Start experimenting!
# (the agent takes over from here)
```

Now go optimize. Make it fast. Don't break it. Don't stop.
