"""
Marginalised scaling benchmark: single-path, batched, facade, and loop modes.

This script measures and compares execution modes on a fixed grid:

(A) One call to `solve_sde_marginalised` for a single Algorithm-4 trajectory.
(B) One direct call to `solve_sde_marginalised_batch` with N independent keys,
    returning stacked trajectories.
(C) One direct batched call followed by pointwise mean/variance aggregation,
    with precomputed keys.
(D) Same as (C), including key splitting inside the timed function.
(E) One call to `solve_sde(..., method="marginalised")` with `num_samples = N`
    (batched marginalised sampling plus facade aggregation).
(F) Explicit Python loop calling `solve_sde_marginalised` N times and stacking
    trajectories (old baseline behavior).

The goal is to separate:
- per-trajectory Algorithm-4 cost,
- batched Algorithm-4 cost,
- facade aggregation overhead,
- and the cost of Python-level repetition across samples.

Timing notes
------------
- The first execution includes JAX tracing/compilation and is not representative.
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
from prob_sde.filtering.sde.marginalised import (
    MarginalisedConfig,
    solve_sde_marginalised,
    solve_sde_marginalised_batch
)
from prob_sde.solvers.sde_solver import (
    TimeGridConfig,
    SDESolverConfig,
    MarginalisedRunConfig,
    solve_sde,
)

from benchmarks.benes_sde.benes_dynamics import drift, diffusion


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
    sde = SDESpec.from_args(drift, diffusion, jnp.array(0.0), bm_factory=None)

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
        lambda: solve_sde_marginalised(key1, sde, marg_cfg)[1],
    )

    # Shared keys for rows that should isolate solver cost from key splitting.
    sample_keys = jax.random.split(jax.random.fold_in(key, 2), n_samples)

    # (B) Direct batched Algorithm-4 paths with precomputed keys.
    time_call(
        f"marg: direct batch trajectories precomputed keys (N={n_samples})",
        lambda: solve_sde_marginalised_batch(sample_keys, sde, marg_cfg)[1],
    )

    def run_direct_batch_aggregated():
        _, samples = solve_sde_marginalised_batch(sample_keys, sde, marg_cfg)
        return jnp.mean(samples, axis=0), jnp.var(samples, axis=0)

    # (C) Direct batch plus mean/variance with precomputed keys.
    time_call(
        f"marg: direct batch mean/var precomputed keys (N={n_samples})",
        run_direct_batch_aggregated,
    )

    def run_direct_batch_aggregated_with_split():
        sample_keys_i = jax.random.split(key, n_samples)
        _, samples = solve_sde_marginalised_batch(sample_keys_i, sde, marg_cfg)
        return jnp.mean(samples, axis=0), jnp.var(samples, axis=0)

    # (D) Direct batch plus mean/variance, including key splitting cost.
    time_call(
        f"marg: direct batch mean/var with split (N={n_samples})",
        run_direct_batch_aggregated_with_split,
    )

    # (E) Facade with N samples (batched paths + mean/var).
    solver_cfg = SDESolverConfig(
        method="marginalised",
        grid=grid,
        marginalised=MarginalisedRunConfig(num_samples=n_samples, prior_scale=1.0),
    )

    def run_facade():
        out = solve_sde(key, sde, solver_cfg)
        return out.mean_trajectory, out.var_trajectory

    time_call(
        f"marg: facade solve_sde mean/var (N={n_samples})",
        run_facade,
    )

    def run_n_looped():
        samples = []
        for key_i in sample_keys:
            samples.append(solve_sde_marginalised(key_i, sde, marg_cfg)[1])
        return jnp.stack(samples)

    # (F) Explicit N-loop (old baseline behavior).
    time_call(
        f"marg: old Nx solve_sde_marginalised loop (N={n_samples})",
        run_n_looped,
    )

if __name__ == "__main__":
    main()
