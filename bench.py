"""
bench.py - Benchmarking harness for autoinfer (DO NOT MODIFY)

Measures inference quality and speed, enforces the WER quality guard,
and logs results. This is the equivalent of autoresearch's eval harness.

Metrics:
  - WER (Word Error Rate) via jiwer
  - Latency: avg and p95 per-utterance time (ms)
  - Throughput: Real-Time Factor (RTFx) = audio_duration / processing_time
  - Peak GPU memory (MB)

Usage:
  python bench.py                    # run benchmark, print results
  python bench.py --baseline         # run and save as baseline
  python bench.py --compare          # run and compare to baseline
  python bench.py --experiment EXP1  # run with experiment tag
"""

import os
import sys
import json
import time
import argparse
import datetime
import importlib
import traceback
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from prepare import (
    MANIFEST_PATH,
    BASELINE_PATH,
    RESULTS_PATH,
    WER_DEGRADATION_THRESHOLD,
    SAMPLE_RATE,
)


def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    """Compute Word Error Rate using jiwer."""
    from jiwer import wer
    # Handle empty strings
    refs = [r if r.strip() else "<empty>" for r in references]
    hyps = [h if h.strip() else "<empty>" for h in hypotheses]
    return wer(refs, hyps)


def measure_memory_peak() -> float:
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def load_manifest() -> list[dict]:
    """Load the evaluation manifest."""
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def run_benchmark(batch_size: int = 8) -> dict:
    """
    Run a full benchmark of infer.py's transcribe() function.

    Returns a dict with all metrics.
    """
    # Reset GPU memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Import infer.py fresh (to pick up any agent modifications)
    if "infer" in sys.modules:
        del sys.modules["infer"]
    import infer

    manifest = load_manifest()
    audio_paths = [item["audio_path"] for item in manifest]
    references = [item["reference"] for item in manifest]
    total_audio_duration = sum(item["duration_sec"] for item in manifest)

    # ---- Warmup run (not timed) ----
    print("Warmup run...")
    try:
        _ = infer.transcribe(audio_paths[:2])
    except Exception as e:
        print(f"Warmup failed: {e}")
        traceback.print_exc()
        return {"status": "WARMUP_FAILED", "error": str(e)}

    # ---- Timed benchmark: per-utterance latency ----
    print(f"Benchmarking {len(audio_paths)} utterances...")
    latencies = []
    all_hypotheses = []

    # Process in batches
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i : i + batch_size]

        start = time.perf_counter()
        try:
            batch_results = infer.transcribe(batch_paths)
        except Exception as e:
            print(f"Inference failed on batch {i}: {e}")
            traceback.print_exc()
            return {"status": "INFERENCE_FAILED", "error": str(e)}
        elapsed = time.perf_counter() - start

        # Per-utterance latency for this batch
        per_utt = (elapsed / len(batch_paths)) * 1000  # ms
        latencies.extend([per_utt] * len(batch_paths))
        all_hypotheses.extend(batch_results)

    # ---- Total wall-clock time (full pipeline) ----
    print("Full pipeline timing...")
    start_total = time.perf_counter()
    try:
        _ = infer.transcribe(audio_paths)
    except Exception as e:
        print(f"Full pipeline failed: {e}")
        return {"status": "PIPELINE_FAILED", "error": str(e)}
    total_time = time.perf_counter() - start_total

    # ---- Compute metrics ----
    wer_score = compute_wer(references, all_hypotheses)
    latency_avg = np.mean(latencies)
    latency_p95 = np.percentile(latencies, 95)
    throughput_rtfx = total_audio_duration / total_time  # >1 means faster than real-time
    memory_peak = measure_memory_peak()

    results = {
        "wer": round(wer_score, 4),
        "latency_avg_ms": round(latency_avg, 1),
        "latency_p95_ms": round(latency_p95, 1),
        "throughput_rtfx": round(throughput_rtfx, 2),
        "memory_peak_mb": round(memory_peak, 1),
        "total_time_sec": round(total_time, 2),
        "num_samples": len(audio_paths),
        "total_audio_sec": round(total_audio_duration, 1),
        "status": "OK",
    }

    return results


def save_baseline(results: dict):
    """Save current results as baseline."""
    with open(BASELINE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline saved to {BASELINE_PATH}")


def load_baseline() -> dict | None:
    """Load the saved baseline."""
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            return json.load(f)
    return None


def compare_to_baseline(results: dict, baseline: dict) -> dict:
    """
    Compare current results to baseline.
    Returns a verdict: KEEP, REVERT, or QUALITY_FAIL.
    """
    verdict = {
        "wer_delta": results["wer"] - baseline["wer"],
        "latency_delta_ms": results["latency_avg_ms"] - baseline["latency_avg_ms"],
        "throughput_delta_rtfx": results["throughput_rtfx"] - baseline["throughput_rtfx"],
        "memory_delta_mb": results["memory_peak_mb"] - baseline["memory_peak_mb"],
    }

    # Quality guard: WER must not degrade beyond threshold
    if verdict["wer_delta"] > WER_DEGRADATION_THRESHOLD:
        verdict["decision"] = "REVERT"
        verdict["reason"] = (
            f"WER degraded by {verdict['wer_delta']:.4f} "
            f"(threshold: {WER_DEGRADATION_THRESHOLD})"
        )
    # Speed improvement: either latency decreased OR throughput increased
    elif (verdict["latency_delta_ms"] < 0) or (verdict["throughput_delta_rtfx"] > 0):
        verdict["decision"] = "KEEP"
        verdict["reason"] = "Speed improved without quality degradation"
    # No speed improvement but no quality loss either
    elif abs(verdict["latency_delta_ms"]) < 1.0 and abs(verdict["throughput_delta_rtfx"]) < 0.1:
        verdict["decision"] = "REVERT"
        verdict["reason"] = "No meaningful speed improvement"
    else:
        verdict["decision"] = "REVERT"
        verdict["reason"] = "Speed regressed"

    return verdict


def log_result(experiment_id: str, results: dict, notes: str = ""):
    """Append a result to the TSV log."""
    timestamp = datetime.datetime.now().isoformat()
    with open(RESULTS_PATH, "a") as f:
        f.write(
            f"{experiment_id}\t"
            f"{timestamp}\t"
            f"{results.get('wer', 'N/A')}\t"
            f"{results.get('latency_avg_ms', 'N/A')}\t"
            f"{results.get('latency_p95_ms', 'N/A')}\t"
            f"{results.get('throughput_rtfx', 'N/A')}\t"
            f"{results.get('memory_peak_mb', 'N/A')}\t"
            f"{results.get('status', 'N/A')}\t"
            f"{notes}\n"
        )


def print_results(results: dict, label: str = "Results"):
    """Pretty-print benchmark results."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    if results["status"] != "OK":
        print(f"  STATUS: {results['status']}")
        print(f"  ERROR:  {results.get('error', 'unknown')}")
        return

    print(f"  WER:              {results['wer']:.4f} ({results['wer']*100:.2f}%)")
    print(f"  Latency (avg):    {results['latency_avg_ms']:.1f} ms/utterance")
    print(f"  Latency (p95):    {results['latency_p95_ms']:.1f} ms/utterance")
    print(f"  Throughput:       {results['throughput_rtfx']:.2f}x real-time")
    print(f"  Peak GPU Memory:  {results['memory_peak_mb']:.1f} MB")
    print(f"  Total Time:       {results['total_time_sec']:.2f}s for {results['total_audio_sec']:.1f}s audio")
    print(f"  Samples:          {results['num_samples']}")
    print(f"{'=' * 60}\n")


def print_comparison(verdict: dict):
    """Pretty-print comparison results."""
    print(f"\n{'=' * 60}")
    print(f"  COMPARISON vs BASELINE")
    print(f"{'=' * 60}")
    print(f"  WER delta:        {verdict['wer_delta']:+.4f}")
    print(f"  Latency delta:    {verdict['latency_delta_ms']:+.1f} ms")
    print(f"  Throughput delta:  {verdict['throughput_delta_rtfx']:+.2f}x")
    print(f"  Memory delta:     {verdict['memory_delta_mb']:+.1f} MB")
    print(f"  ---")
    print(f"  DECISION: {verdict['decision']}")
    print(f"  REASON:   {verdict['reason']}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="AutoInfer Benchmark")
    parser.add_argument("--baseline", action="store_true", help="Save results as baseline")
    parser.add_argument("--compare", action="store_true", help="Compare to saved baseline")
    parser.add_argument("--experiment", type=str, default="manual", help="Experiment ID tag")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    args = parser.parse_args()

    print("Running benchmark...")
    results = run_benchmark(batch_size=args.batch_size)
    print_results(results, label=f"Experiment: {args.experiment}")

    if results["status"] != "OK":
        log_result(args.experiment, results, notes=f"FAILED: {results.get('error', '')}")
        sys.exit(1)

    if args.baseline:
        save_baseline(results)
        log_result(args.experiment, results, notes="BASELINE")

    if args.compare:
        baseline = load_baseline()
        if baseline is None:
            print("No baseline found! Run with --baseline first.")
            sys.exit(1)
        verdict = compare_to_baseline(results, baseline)
        print_comparison(verdict)
        log_result(args.experiment, results, notes=f"{verdict['decision']}: {verdict['reason']}")

        # Exit code: 0 = KEEP, 1 = REVERT
        if verdict["decision"] == "REVERT":
            sys.exit(1)
    else:
        log_result(args.experiment, results)


if __name__ == "__main__":
    main()
