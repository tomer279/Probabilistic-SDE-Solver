"""Section-level runtime matrix for Benes SDE benchmark scripts.

Profiles:
- gsf_vs_em:
    - run_experiment (full)
    - _compute_path_results
    - _compute_mc_error_results
    - estimate_errors_for_delta (smallest delta)
- mgsf_gsf_em:
    - run_experiment (full)
    - _compute_path_results
    - _compute_mc_error_results
    - estimate_errors_for_delta (smallest delta)
- marginalised_gsf_em:
    - run_experiment (full)
    - _compute_ensemble_path_results
    - _compute_mc_error_results
    - estimate_errors_for_delta (smallest delta)

Uses default `ExperimentConfig()` from each benchmark.
Default Monte Carlo sizes × deltas can make runs take many minutes on CPU.
Reproducibility: fixes `mc.seed=0` via `_with_fixed_seed`.
"""

from __future__ import annotations

import statistics
import time

import jax

from benchmarks.benes_sde import benes_gsf_vs_em as gsf
from benchmarks.benes_sde import benes_mgsf_gsf_em as mgsf
from benchmarks.benes_sde import benes_marginalised_gsf_em as marg


def _with_fixed_seed(cfg, seed: int = 0):
    """Return config with mc.seed fixed for reproducible timing.
    
    Note
    ----
    This helper is intentionally duplicated in the companion timing script
    to keep each file self-contained.
    """
    if getattr(cfg, "mc", None) is None:
        return cfg
    return cfg.__class__(
        **{
            **cfg.__dict__,
            "mc": cfg.mc.__class__(**{**cfg.mc.__dict__, "seed": int(seed)}),
        }
    )


def _timed_call(fn, repeats: int = 3):
    """Return cold runtime and repeated warm runtimes."""
    t0 = time.perf_counter()
    fn()
    cold = time.perf_counter() - t0

    warm = []
    for _ in range(repeats):
        t1 = time.perf_counter()
        fn()
        warm.append(time.perf_counter() - t1)

    return cold, warm


def _print_line(name: str, cold: float, warm: list[float]) -> None:
    """Print compact stats line."""
    print(
        f"{name:40s} "
        f"cold={cold:8.3f}s  "
        f"mean={statistics.mean(warm):8.3f}s  "
        f"min={min(warm):8.3f}s  "
        f"max={max(warm):8.3f}s  "
        f"(n={len(warm)})"
    )


def _run_gsf_sections(repeats: int) -> None:
    cfg = _with_fixed_seed(gsf.ExperimentConfig(), seed=0)

    run_seed = 0 if cfg.mc.seed is None else int(cfg.mc.seed)
    key_paths, key_mc = jax.random.split(jax.random.PRNGKey(run_seed), 2)
    smallest_delta = min(cfg.time.deltas)

    tasks = [
        ("gsf::run_experiment(full)", lambda: gsf.run_experiment(cfg)),
        ("gsf::_compute_path_results", lambda: gsf._compute_path_results(key_paths, cfg)),
        ("gsf::_compute_mc_error_results", lambda: gsf._compute_mc_error_results(key_mc, cfg)),
        (
            "gsf::estimate_errors_for_delta(min)",
            lambda: gsf.estimate_errors_for_delta(
                key_mc, smallest_delta, cfg, progress_bar=None
            ),
        ),
    ]

    print("\n[GSF benchmark sections]")
    for name, fn in tasks:
        cold, warm = _timed_call(fn, repeats=repeats)
        _print_line(name, cold, warm)


def _run_mgsf_sections(repeats: int) -> None:
    cfg = _with_fixed_seed(mgsf.ExperimentConfig(), seed=0)

    run_seed = 0 if cfg.mc.seed is None else int(cfg.mc.seed)
    key_paths, key_mc = jax.random.split(jax.random.PRNGKey(run_seed), 2)
    smallest_delta = min(cfg.time.deltas)

    tasks = [
        ("mgsf::run_experiment(full)", lambda: mgsf.run_experiment(cfg)),
        ("mgsf::_compute_path_results", lambda: mgsf._compute_path_results(key_paths, cfg)),
        ("mgsf::_compute_mc_error_results", lambda: mgsf._compute_mc_error_results(key_mc, cfg)),
        (
            "mgsf::estimate_errors_for_delta(min)",
            lambda: mgsf.estimate_errors_for_delta(
                key_mc, smallest_delta, cfg, progress_bar=None
            ),
        ),
    ]

    print("\n[MGSF benchmark sections]")
    for name, fn in tasks:
        cold, warm = _timed_call(fn, repeats=repeats)
        _print_line(name, cold, warm)


def _run_marg_sections(repeats: int) -> None:
    cfg = _with_fixed_seed(marg.ExperimentConfig(), seed=0)

    run_seed = 0 if cfg.mc.seed is None else int(cfg.mc.seed)
    key_paths, key_mc = jax.random.split(jax.random.PRNGKey(run_seed), 2)
    smallest_delta = min(cfg.time.deltas)

    tasks = [
        ("marg::run_experiment(full)", lambda: marg.run_experiment(cfg)),
        (
            "marg::_compute_ensemble_path_results",
            lambda: marg._compute_ensemble_path_results(key_paths, cfg),
        ),
        ("marg::_compute_mc_error_results", lambda: marg._compute_mc_error_results(key_mc, cfg)),
        (
            "marg::estimate_errors_for_delta(min)",
            lambda: marg.estimate_errors_for_delta(
                key_mc, smallest_delta, cfg, progress_bar=None
            ),
        ),
    ]

    print("\n[Marginalised benchmark sections]")
    for name, fn in tasks:
        cold, warm = _timed_call(fn, repeats=repeats)
        _print_line(name, cold, warm)


def main() -> None:
    repeats = 3
    print(f"Section timing matrix (warm repeats={repeats}, seed=0)")
    print("-" * 100)

    _run_gsf_sections(repeats)
    _run_mgsf_sections(repeats)
    _run_marg_sections(repeats)


if __name__ == "__main__":
    main()