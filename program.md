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

### What we know so far (experiments 1-4)
- flash_attention_2 gives marginal gains — encoder sequences are short (12.5Hz), attention is NOT the bottleneck
- torch.compile HURTS — model too small, sequences too short, compilation overhead dominates
- Forcing language="English" was the biggest win — fewer autoregressive decode steps = direct speedup
- The decoder is MEMORY-BANDWIDTH BOUND at batch=1 (0.84GB weights, read every token step)
- Theoretical analysis: paper achieves ~10% of hardware ceiling. Massive headroom exists.

### Phase 2: The Bandwidth Wall (experiments 5-15)
The decoder reads 0.84GB of weights every token step. This is THE bottleneck. Attack it directly:

- **INT8 quantization**: Halves weight reads → should give ~1.5-1.8x throughput. Try `bitsandbytes` load_in_8bit or `torch.ao.quantization`. This is the single highest-ROI experiment right now.
- **INT4 quantization**: Quarters weight reads. Watch WER — the 0.6B model has less redundancy. Try GPTQ or AWQ if available for this architecture.
- **Mixed precision quantization**: INT4 the decoder (bandwidth-bound), keep encoder at bf16 (compute is fine). The encoder is only 0.36GB — quantizing it barely helps.
- **Static KV cache**: Pre-allocate KV cache tensors instead of dynamic allocation. Eliminates memory allocation overhead every decode step. Use `model.generation_config.cache_implementation = "static"` or manually pre-allocate.
- **Batch size sweep**: You have ~48GB on A40, model uses ~7GB. Try batch_size=[16, 32, 64] — larger batches amortize weight reads across more sequences. Find the sweet spot before you OOM.

### Phase 3: Decode Step Reduction (experiments 15-25)
Every token you DON'T generate is free speed. The forcing-English trick proved this.

- **Minimal output format**: Strip any unnecessary tokens from the output template. Does the model emit `<|im_start|>assistant\nlanguage English<asr_text>...`? Can you truncate earlier or skip the preamble?
- **Greedy decoding with temperature=0**: Ensure there's zero sampling overhead. No beam search, no top-k, no top-p.
- **Max new tokens cap**: Set a tight `max_new_tokens` based on expected output length. For 30s audio at ~2.5 words/sec = ~100 tokens max. Don't let the model ramble.
- **Early stopping on EOS**: Verify the model stops immediately on `<|im_end|>`, not padding to max length.
- **Speculative decoding**: Use a tiny draft model (if one exists for Qwen3) to propose multiple tokens at once. Even a simple n-gram predictor for common words could help.

### Phase 4: Pipeline & Memory (experiments 25-35)
- **Sort by audio length before batching**: Groups similar-length sequences → less padding waste in the encoder, more uniform decode lengths.
- **Encoder output caching**: If processing the same audio repeatedly (e.g., retries), cache encoder hidden states.
- **Overlap encoder and decoder**: While the decoder is generating tokens for batch N, start encoding batch N+1 on a separate CUDA stream.
- **Memory-efficient attention for cross-attention**: The decoder cross-attends to 12.5Hz × audio_sec encoder tokens every step. For 30s audio = 375 tokens. At batch=64, that's a lot of KV cache reads.
- **torch.cuda.graphs**: Capture the decode step as a CUDA graph to eliminate kernel launch overhead (this is what vLLM does). Note: torch.compile failed, but raw CUDA graphs are different — they skip Python overhead entirely.

### Phase 5: Nuclear Options (experiments 35+)
- **torch.cuda.CUDAGraph**: Capture the entire decode step as a CUDA graph. This eliminates Python overhead and kernel launch latency — it's the single biggest trick vLLM uses. Different from torch.compile (which failed). Manually capture the forward pass.
- **CTranslate2 conversion**: Convert decoder to CTranslate2 format (like faster-whisper did for Whisper). Optimized C++ runtime with INT8 built-in. This is how faster-whisper gets 150-200x.
- **TensorRT-LLM**: Export and compile the decoder. Maximum performance but painful setup. This is the path to 500x+.
- **Decoder layer pruning**: The 0.6B model has 28 decoder layers. Try dropping the last 2-4 layers and see WER impact. If WER holds, you just got a permanent 7-14% speedup.
- **FP8 inference**: If the GPU supports it (A40 does not, H100 does). Save for when someone runs this on H100.
- **Continuous batching**: Process a stream of requests, inserting new sequences into decode slots as others finish. This is how you get to the paper's 2000x at conc=128.
- **vLLM backend**: Switch to vLLM for the decoder only. This is what the paper uses. Less interesting as a research finding — we want to beat vLLM, not use it.

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

## The Real Competition

Forget the Qwen3-ASR paper target (108x with vLLM on A100). The real question is: **can Qwen3-ASR beat Whisper's optimized ecosystem on an A40?**

### Competitive Landscape (A40, single GPU)
| Model | Size | Throughput | WER (test-clean) | Stack |
|---|---|---|---|---|
| **Qwen3-ASR-0.6B (us, now)** | 0.6B | **99x** | **1.45%** | PyTorch transformers |
| Whisper large-v3 (HF) | 1.5B | ~50-80x | 2.0-2.5% | PyTorch transformers |
| faster-whisper large-v3 | 1.5B | ~150-200x | 2.0-2.5% | CTranslate2 + INT8 |
| Whisper large-v3-turbo | 0.8B | ~300-500x | 2.5-3.0% | Distilled + optimized |

**Qwen3-ASR already wins on WER.** The gap is throughput. faster-whisper turbo gets 300-500x through CTranslate2 (optimized C++ runtime) + INT8 + CUDA graphs. We need to close that gap using PyTorch-native tools.

### A40 Hardware Limits
| Spec | Value | Implication |
|---|---|---|
| Memory BW | 696 GB/s | Decoder bottleneck: 696 / 0.42 GB (INT8 weights) ≈ 1,657 decode steps/sec at batch=1 |
| INT8 TOPS | 299 | 2x effective throughput vs FP16 — INT8 quantization is the single biggest lever |
| FP16 TFLOPS | 150 | Encoder is fine, not the bottleneck |
| VRAM | 48 GB | Plenty of room for large batches — batch=32 amortizes weight reads 32x |

**Theoretical ceiling:** At INT8 batch=1, decoder can do ~1,657 weight reads/sec. Average ASR output is ~30 tokens for 10s audio → ~550 utterances/sec → **~400-600x RTF**. With batching, the ceiling is in the thousands. We are currently at 99x — there is a **4-6x gap** to close.

### Milestone Goals (updated after 8 experiments)
Current: 99x throughput, 1.45% WER.

1. ~~**Bronze**: 108x — match Qwen paper target (vLLM + A100) on A40 with PyTorch~~ ALMOST (99x)
2. **Silver**: 200x — beat faster-whisper large-v3. Requires INT8 + CUDA graphs.
3. **Gold**: 400x — match faster-whisper turbo throughput while keeping WER < 2%. This is the real target.
4. **Platinum**: 600x+ — beat the entire Whisper ecosystem on A40. INT8 + CUDA graphs + batching. This is the theoretical ceiling.

## Important Notes

- The model is Qwen3-ASR-0.6B (the smaller variant for fast iteration). Findings should generally transfer to the 1.7B model.
- All audio is 16kHz mono WAV, max 30 seconds.
- Benchmarks use 100 LibriSpeech test-clean utterances.
- Peak GPU memory matters — lower memory means we could eventually serve more concurrent requests.
- The evaluation set is fixed. Don't overfit to it — optimizations should be general.
- The published numbers use vLLM — reaching them without vLLM would be a notable finding. Reaching them with a simpler setup is also valuable.

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
