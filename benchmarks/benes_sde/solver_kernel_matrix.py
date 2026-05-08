"""
Solver kernel timing matrix for `solve_sde` methods (GSF / MGSF / marginalised).

This script isolates solver call cost on a fixed grid, outside benchmark-level
Monte Carlo loops. It times:

- `solve_sde(..., method="gsf")`
- `solve_sde(..., method="mgsf")`
- `solve_sde(..., method="marginalised")`
  (facade-level Monte Carlo aggregation over `num_samples`, with batched
  marginalised trajectory sampling in the underlying implementation)

Timing notes
------------
- The first run includes JAX compilation; compare warm-run timings.
- Use `jax.block_until_ready` on returned arrays to measure compute time.

Run from the repository root (after `pip install -e .`):

    python benchmarks/benes_sde/solver_kernel_matrix.py
"""

import jax
import jax.numpy as jnp
import time

from prob_sde.solvers.sde_solver import (
    TimeGridConfig, SDESolverConfig, GSFRunConfig, MGSFRunConfig,
    MarginalisedRunConfig, solve_sde,
)
from prob_sde import SDESpec, piecewise_parabolic_brownian

# define drift/diffusion/x0 like benchmarks
def drift(x, t):
    return jax.numpy.tanh(x)

def diffusion(_x, _t):
    return jnp.array(1.0)

def time_call(name, fn, warmup=1, repeats=5):
    for _ in range(warmup):
        jax.block_until_ready(fn())
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)  # IMPORTANT
        ts.append(time.perf_counter() - t0)
    print(name, "mean", sum(ts)/len(ts), "min", min(ts), "max", max(ts))


def main() -> None:
    """Run the solver kernel timing matrix"""

    key = jax.random.PRNGKey(0)
    sde_mgsf = SDESpec.from_args(
        drift, diffusion, jnp.array(0.0), piecewise_parabolic_brownian)
    sde_gsf = SDESpec.from_args(
        drift, diffusion, jnp.array(0.0), bm_factory=None)

    delta = 2**-5
    t_final = 1.0
    grid = TimeGridConfig(
        delta=delta,
        num_steps=int(round(t_final / delta)),
        t0=0.0)

    # GSF
    cfg_gsf = SDESolverConfig(
        method="gsf",
        grid=grid,
        gsf=GSFRunConfig(return_uncertainty=False)
    )
    time_call(
        "gsf",
        lambda: solve_sde(key, sde_gsf, cfg_gsf).trajectory
    )
    # MGSF (note: your benchmark uses parabolic factory on SDE)
    cfg_mgsf = SDESolverConfig(
        method="mgsf", grid=grid, mgsf=MGSFRunConfig(return_uncertainty=False))
    time_call(
        "mgsf",
        lambda: solve_sde(key, sde_mgsf, cfg_mgsf).trajectory
    )
    # Marginalised facade (batched marginalised sampling + aggregation in solve_marginalised)
    cfg_m = SDESolverConfig(
        method="marginalised",
        grid=grid,
        marginalised=MarginalisedRunConfig(num_samples=100, prior_scale=1.0),
    )
    time_call(
        "marginalised_facade",
        lambda: solve_sde(key, sde_mgsf, cfg_m).mean_trajectory
    )

if __name__ == "__main__":
    main()