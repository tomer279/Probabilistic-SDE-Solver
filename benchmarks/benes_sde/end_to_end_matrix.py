"""End-to-end runtime matrix for Benes SDE benchmark scripts.

Profiles:
- benes_gsf_vs_em.run_experiment
- benes_mgsf_gsf_em.run_experiment
- benes_marginalised_gsf_em.run_experiment

Uses default `ExperimentConfig()` from each benchmark.
Default Monte Carlo sizes × deltas can make runs take many minutes on CPU.
Reproducibility: fixes `mc.seed=0` via `_with_fixed_seed`.
"""

from __future__ import annotations

import statistics
import time

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


def _time_run(fn, cfg, repeats: int = 3) -> tuple[float, list[float]]:
    """Return cold-run time and repeated warm-run times."""
    t0 = time.perf_counter()
    fn(cfg)
    cold = time.perf_counter() - t0

    warm = []
    for _ in range(repeats):
        t1 = time.perf_counter()
        fn(cfg)
        warm.append(time.perf_counter() - t1)
    return cold, warm


def main() -> None:
    """Run end-to-end matrix and print compact timing summary."""
    runs = [
        ("gsf_vs_em", gsf.run_experiment, gsf.ExperimentConfig()),
        ("mgsf_gsf_em", mgsf.run_experiment, mgsf.ExperimentConfig()),
        ("marginalised_gsf_em", marg.run_experiment, marg.ExperimentConfig()),
    ]

    repeats = 3
    print(f"End-to-end runtime matrix (warm repeats={repeats})")
    print("-" * 72)

    for name, fn, cfg in runs:
        cfg = _with_fixed_seed(cfg, seed=0)
        cold, warm = _time_run(fn, cfg, repeats=repeats)
        print(
            f"{name:24s} "
            f"cold={cold:8.3f}s  "
            f"mean={statistics.mean(warm):8.3f}s  "
            f"min={min(warm):8.3f}s  "
            f"max={max(warm):8.3f}s  "
            f"(n={repeats})"
        )


if __name__ == "__main__":
    main()