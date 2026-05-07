"""
Marginalised scaling benchmark: inner Algorithm-4 vs facade replication.

This script measures and compares three execution modes on a fixed grid:

(A) One call to `solve_sde_marginalised` (inner Algorithm-4 implementation).
(B) One call to the unified facade `solve_sde(..., method="marginalised")` with
    `num_samples = N` (which currently performs a Python loop of N inner solves
    and aggregates mean/variance).
(C) An explicit Python loop that calls `solve_sde_marginalised` N times and stacks
    the sampled trajectories (to isolate the overhead of the facade vs a manual loop).

The goal is to identify whether marginalised runtime is dominated by:
- per-trajectory Algorithm-4 cost, or
- Python-level repetition across samples (`num_samples`).

Timing notes
------------
- The first execution includes JAX compilation and is not representative.
- Timings use `jax.block_until_ready` to measure actual compute time.
- For reproducibility, the script fixes the PRNG seed and uses deterministic
  configuration defaults unless overridden in the code.

Run from the repository root (after `pip install -e .`):

    python benchmarks/benes_sde/marginalised_scaling_bench.py
"""

import time
import jax
import jax.numpy as jnp

from prob_sde import SDESpec
from prob_sde.filtering.sde.marginalised import MarginalisedConfig, solve_sde_marginalised
from prob_sde.solvers.sde_solver import (
    TimeGridConfig,
    SDESolverConfig,
    MarginalisedRunConfig,
    solve_sde,
)


def benes_drift(x, _t):
    return jnp.tanh(x)


def benes_diffusion(_x, _t):
    return jnp.array(1.0)


def time_call(name, fn, warmup=2, repeats=5):
    for _ in range(warmup):
        jax.block_until_ready(fn())

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)

    mean = sum(times) / len(times)
    print(f"{name}: mean={mean:.4f}s  min={min(times):.4f}s  max={max(times):.4f}s  (n={repeats})")


def main():
    key = jax.random.PRNGKey(0)

    delta = 2**-4
    t_final = 1.0
    num_steps = int(round(t_final / delta))
    n_samples = 100

    grid = TimeGridConfig(delta=float(delta), num_steps=num_steps, t0=0.0)

    # SDE without bm_factory — marginalised ignores it internally
    sde = SDESpec.from_args(benes_drift, benes_diffusion, jnp.array(0.0), bm_factory=None)

    marg_cfg = MarginalisedConfig(
        delta=float(delta),
        num_steps=num_steps,
        sample_posterior_position=True,
        use_ekf1=True,
        variance_floor=1e-12,
        prior_diffusion=1.0,
        return_uncertainty=False,
    )

    key1 = jax.random.fold_in(key, 1)

    # (A) Single Algorithm-4 path
    time_call(
        "marg: single solve_sde_marginalised",
        lambda: solve_sde_marginalised(key1, sde, marg_cfg)[1],  # trajectory only
    )

    # (B) Facade with N samples (includes Python loop + mean/var)
    solver_cfg = SDESolverConfig(
        method="marginalised",
        grid=grid,
        marginalised=MarginalisedRunConfig(num_samples=n_samples, prior_scale=1.0),
    )
    time_call(
        f"marg: facade solve_sde (N={n_samples})",
        lambda: solve_sde(key, sde, solver_cfg).mean_trajectory,
    )

    # (C) Explicit N-loop (mirrors facade loop; isolates Python overhead)
    keys = jax.random.split(jax.random.fold_in(key, 2), n_samples)

    def run_n_looped():
        samples = []
        for k in keys:
            samples.append(solve_sde_marginalised(k, sde, marg_cfg)[1])
        return jnp.stack(samples)

    time_call(
        f"marg: N× solve_sde_marginalised (Python loop, N={n_samples})",
        run_n_looped,
    )


if __name__ == "__main__":
    main()