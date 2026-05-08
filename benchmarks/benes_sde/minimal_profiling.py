"""
Minimal profiling pass for benes_marginalised_gsf_em.py.

Profiles:
- run_experiment (end-to-end)
- _compute_ensemble_path_results
- _compute_mc_error_results
- estimate_errors_for_delta at smallest delta

Note
----
This script reports simple wall-clock timings but does not call
`jax.block_until_ready` on JAX results. As a consequence, timings can be noisier
or underestimate compute time depending on asynchronous dispatch behavior.
For tighter JAX timing, prefer:
- `benchmarks/benes_sde/section_timing_matrix.py` (section-level timings), and
- `benchmarks/benes_sde/marginalised_scaling_bench.py` (inner vs facade scaling).
"""

from __future__ import annotations

import statistics
import time

import jax

from benchmarks.benes_sde.benes_marginalised_gsf_em import (
    ExperimentConfig,
    _compute_ensemble_path_results,
    _compute_mc_error_results,
    estimate_errors_for_delta,
    run_experiment,
)


def timed_call(name, fn, repeats=3):
    """Return timing stats and last result."""
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    print(
        f"{name:35s} "
        f"mean={statistics.mean(times):.4f}s  "
        f"min={min(times):.4f}s  "
        f"max={max(times):.4f}s  "
        f"(n={repeats})"
    )
    return result, times


def main():
    """Run a compact profiling pass over key benchmark stages."""
    cfg = ExperimentConfig()

    # 1) End-to-end benchmark timing
    timed_call("run_experiment (full)", lambda: run_experiment(cfg), repeats=3)

    # Prepare keys once for section-level timings
    run_seed = 0 if cfg.mc.seed is None else int(cfg.mc.seed)
    key_paths, key_mc = jax.random.split(jax.random.PRNGKey(run_seed), 2)

    # 2) Ensemble-panel computation
    timed_call(
        "_compute_ensemble_path_results",
        lambda: _compute_ensemble_path_results(key_paths, cfg),
        repeats=3,
    )

    # 3) MC error curves computation
    timed_call(
        "_compute_mc_error_results",
        lambda: _compute_mc_error_results(key_mc, cfg),
        repeats=3,
    )

    # 4) Single-delta cost at hardest delta (smallest step)
    smallest_delta = min(cfg.time.deltas)
    timed_call(
        "estimate_errors_for_delta (min delta)",
        lambda: estimate_errors_for_delta(key_mc, smallest_delta, cfg, progress_bar=None),
        repeats=3,
    )


if __name__ == "__main__":
    main()
