"""
Time and compare single-trajectory runs of ``solve_sde_marginalised``.

This script isolates the inner marginalised SDE filter (Algorithm 4) from
Monte Carlo facades and benchmark loops. It warms JAX compilation, measures
wall time with ``jax.block_until_ready``, and compares ``use_ekf1=True`` vs
``False`` to see how much time is spent on drift Jacobian work inside the
measurement update.

Run from the repository root (with the package installed, e.g. ``pip install -e .``):

    python benchmarks/benes_sde/profile_marginalised_inner_solve.py

Functions
---------
benes_drift, benes_diffusion
    Scalar Benes SDE fields for the demo problem.
build_sde_spec
    Construct ``SDESpec`` for the timed runs.
time_single_marginalised_solve
    One warmup plus repeated timed calls for one ``MarginalisedConfig``.
print_comparison_table
    Print timing statistics and simple per-step breakdown.
main
    CLI entry: run EKF1 on/off comparison for a fixed grid.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prob_sde.core.sde import SDESpec
from prob_sde.filtering.sde.marginalised import MarginalisedConfig, solve_sde_marginalised


def benes_drift(x: jnp.ndarray, _t: float) -> jnp.ndarray:
    """Benes drift tanh(x)."""
    return jnp.tanh(x)


def benes_diffusion(_x: jnp.ndarray, _t: float) -> jnp.ndarray:
    """Constant diffusion equal to one."""
    return jnp.array(1.0)


def build_sde_spec(x0: float) -> SDESpec:
    """Return ``SDESpec`` for the timed solves (no Brownian factory required)."""
    return SDESpec.from_args(
        benes_drift,
        benes_diffusion,
        jnp.asarray(x0),
        bm_factory=None,
    )


@dataclass(frozen=True)
class TimedRunStats:
    """Aggregated timing results for one configuration."""

    label: str
    seconds: tuple[float, ...]
    num_steps: int

    @property
    def mean_s(self) -> float:
        """Return the mean wall time in seconds."""
        return float(statistics.mean(self.seconds))

    @property
    def min_s(self) -> float:
        """Return the minimum wall time in seconds."""
        return float(min(self.seconds))

    @property
    def max_s(self) -> float:
        """Return the maximum wall time in seconds."""
        return float(max(self.seconds))

    def seconds_per_step(self) -> float:
        """Return mean time divided by the number of integration steps."""
        return self.mean_s / max(self.num_steps, 1)


def time_single_marginalised_solve(
    key: jax.Array,
    sde: SDESpec,
    cfg: MarginalisedConfig,
    label: str,
    warmup: int,
    repeats: int,
) -> TimedRunStats:
    """Warm JAX once, then time ``solve_sde_marginalised`` with ``block_until_ready``."""
    for _ in range(warmup):
        out = solve_sde_marginalised(key, sde, cfg)
        if isinstance(out, tuple):
            trajectory = out[1]
        else:
            trajectory = out
        jax.block_until_ready(trajectory)

    times: list[float] = []
    for i in range(repeats):
        k = jax.random.fold_in(key, i)
        t0 = time.perf_counter()
        out = solve_sde_marginalised(k, sde, cfg)
        if isinstance(out, tuple):
            trajectory = out[1]
        else:
            trajectory = out
        jax.block_until_ready(trajectory)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return TimedRunStats(label=label, seconds=tuple(times), num_steps=cfg.num_steps)


def print_comparison_table(stats_list: list[TimedRunStats]) -> None:
    """Print a small table of mean / min / max and per-step means."""
    print("\nSingle-trajectory marginalised solve (inner API)")
    print("label".ljust(22), "mean_s", "min_s", "max_s", "s/step", sep="\t")
    for s in stats_list:
        print(
            s.label.ljust(22),
            f"{s.mean_s:.6f}",
            f"{s.min_s:.6f}",
            f"{s.max_s:.6f}",
            f"{s.seconds_per_step():.8f}",
            sep="\t",
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the timing harness."""
    parser = argparse.ArgumentParser(
        description=(
            "Time one solve_sde_marginalised call and compare EKF1 Jacobian on/off."
        )
    )
    parser.add_argument("--delta", type=float, default=1.0 / 16.0, help="Step size.")
    parser.add_argument(
        "--t-final",
        type=float,
        default=1.0,
        help="End time; num_steps = round(t_final / delta).",
    )
    parser.add_argument("--x0", type=float, default=0.0, help="Initial state.")
    parser.add_argument("--seed", type=int, default=0, help="Base PRNG seed.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs before timing.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Timed repetitions after warmup.",
    )
    parser.add_argument(
        "--deterministic-position",
        action="store_true",
        help="Use posterior mean for positions; default is posterior sampling.",
    )
    return parser.parse_args()


def main() -> None:
    """Run timed comparisons for ``use_ekf1`` True vs False."""
    args = parse_args()
    num_steps = int(round(args.t_final / args.delta))
    if num_steps < 1:
        raise SystemExit("num_steps must be at least 1; check --delta and --t-final.")

    key = jax.random.PRNGKey(int(args.seed))
    sde = build_sde_spec(args.x0)

    sample_posterior = not bool(args.deterministic_position)

    base = dict(
        delta=float(args.delta),
        num_steps=num_steps,
        sample_posterior_position=sample_posterior,
        variance_floor=1e-12,
        prior_diffusion=1.0,
        return_uncertainty=False,
    )

    stats_ekf1 = time_single_marginalised_solve(
        key=key,
        sde=sde,
        cfg=MarginalisedConfig(**base, use_ekf1=True),
        label="use_ekf1=True",
        warmup=int(args.warmup),
        repeats=int(args.repeats),
    )
    stats_ekf0 = time_single_marginalised_solve(
        key=jax.random.fold_in(key, 1),
        sde=sde,
        cfg=MarginalisedConfig(**base, use_ekf1=False),
        label="use_ekf1=False",
        warmup=int(args.warmup),
        repeats=int(args.repeats),
    )

    print(
        "delta=",
        args.delta,
        " t_final=",
        args.t_final,
        " num_steps=",
        num_steps,
        " warmup=",
        args.warmup,
        " repeats=",
        args.repeats,
        " sample_posterior_position=",
        sample_posterior,
        sep="",
    )
    print_comparison_table([stats_ekf1, stats_ekf0])

    ratio = stats_ekf1.mean_s / max(stats_ekf0.mean_s, 1e-12)
    print("\nMean time ratio (EKF1 / EKF0 measurement Jacobian):", f"{ratio:.3f}")


if __name__ == "__main__":
    main()