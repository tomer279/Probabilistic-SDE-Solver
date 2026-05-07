"""
Time the Benes marginalised benchmark ensemble path construction.

This script measures wall time for the same work as
``_compute_ensemble_path_results`` in ``benes_marginalised_gsf_em``:
for each seed, build coupled coarse EM increments and one marginalised
trajectory, then aggregate means and quantiles.

It reports total time for the instrumented loop, per-component sums
(``prepare_coupled_discretization``, coarse Euler--Maruyama path,
inner ``solve_sde_marginalised`` with ``block_until_ready``), and mean
time per path. With ``--verify-full-api``, also times one call to the
uninstrumented ``_compute_ensemble_path_results`` (second full pass).

Run from the repository root::

    python benchmarks/benes_sde/time_marginalised_ensemble_paths.py
    python benchmarks/benes_sde/time_marginalised_ensemble_paths.py --num-sample-paths 50
    python benchmarks/benes_sde/time_marginalised_ensemble_paths.py --verify-full-api

Functions
---------
parse_args
    CLI: seed, num_sample_paths override, optional full-API reference timing.
build_experiment_config
    Build ``ExperimentConfig`` with optional overrides.
split_ensemble_key
    Match ``run_experiment`` key splitting for the ensemble ``key_paths``.
time_ensemble_path_construction
    One instrumented pass mirroring ``_collect_ensemble_paths``.
time_full_api_reference
    Wall time for ``_compute_ensemble_path_results`` only.
print_report
    Print totals, fractions, per-path marginalised min/mean/max.
main
    Entry point.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.benchmark_utils import (
    euler_maruyama_from_increments,
    prepare_coupled_discretization,
    resolve_mc_run_seed,
)
from benchmarks.benes_sde.benes_marginalised_gsf_em import (
    ExperimentConfig,
    _compute_ensemble_path_results,
    _ensemble_summary_arrays,
    _marginalised_path,
    _resolve_delta_path,
    drift,
    diffusion,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Time ensemble EM + marginalised path construction for the "
            "marginalised Benes benchmark (path panel)."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed passed via MonteCarloConfig.seed (default: fresh seed).",
    )
    parser.add_argument(
        "--num-sample-paths",
        type=int,
        default=None,
        help="Override mc.num_sample_paths for scaling experiments.",
    )
    parser.add_argument(
        "--verify-full-api",
        action="store_true",
        help=(
            "Time one uninstrumented _compute_ensemble_path_results call "
            "before the instrumented pass (runs the full ensemble twice)."
        ),
    )
    return parser.parse_args()


def build_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    """Return ``ExperimentConfig`` with optional MC seed and path count."""
    cfg = ExperimentConfig()
    mc = cfg.mc
    if args.num_sample_paths is not None:
        mc = replace(mc, num_sample_paths=int(args.num_sample_paths))
    if args.seed is not None:
        mc = replace(mc, seed=int(args.seed))
    if mc is cfg.mc:
        return cfg
    return replace(cfg, mc=mc)


def split_ensemble_key(cfg: ExperimentConfig) -> jax.Array:
    """Replicate ``key_paths`` from ``run_experiment`` for the ensemble block."""
    run_seed = resolve_mc_run_seed(cfg.mc.seed)
    key_paths, _key_mc = jax.random.split(jax.random.PRNGKey(run_seed), 2)
    return key_paths


def time_ensemble_path_construction(
    key_paths_root: jax.Array,
    cfg: ExperimentConfig,
):
    """
    Run one instrumented ensemble pass mirroring ``_collect_ensemble_paths``.

    Returns
    -------
    summary : dict
        Output of ``_ensemble_summary_arrays``.
    stats : dict
        Timing aggregates and per-path lists.
    """
    delta_path = _resolve_delta_path(cfg)
    path_keys = jax.random.split(key_paths_root, cfg.mc.num_sample_paths)

    em_paths: list = []
    marg_paths: list = []
    t_disc_list: list[float] = []
    t_em_list: list[float] = []
    t_marg_list: list[float] = []

    t_loop0 = time.perf_counter()
    for key_i in path_keys:
        t0 = time.perf_counter()
        disc = prepare_coupled_discretization(key_i, delta_path, cfg.time.t_final)
        t1 = time.perf_counter()

        em_paths.append(
            euler_maruyama_from_increments(
                drift,
                diffusion,
                disc.dw_coarse,
                delta_path,
                cfg.x0,
            )
        )
        t2 = time.perf_counter()

        marg_key = jax.random.fold_in(key_i, 2)
        traj = _marginalised_path(marg_key, delta_path, cfg)
        jax.block_until_ready(traj)
        t3 = time.perf_counter()

        marg_paths.append(traj)
        t_disc_list.append(t1 - t0)
        t_em_list.append(t2 - t1)
        t_marg_list.append(t3 - t2)

    t_loop1 = time.perf_counter()
    loop_wall_s = t_loop1 - t_loop0

    summary = _ensemble_summary_arrays(em_paths, marg_paths, cfg.time.t_final)

    n = len(path_keys)
    sum_disc = float(sum(t_disc_list))
    sum_em = float(sum(t_em_list))
    sum_marg = float(sum(t_marg_list))

    stats = {
        "n_paths": n,
        "delta_path": delta_path,
        "loop_wall_s": loop_wall_s,
        "sum_disc_s": sum_disc,
        "sum_em_s": sum_em,
        "sum_marg_s": sum_marg,
        "mean_disc_s": sum_disc / max(n, 1),
        "mean_em_s": sum_em / max(n, 1),
        "mean_marg_s": sum_marg / max(n, 1),
        "t_disc_list": t_disc_list,
        "t_em_list": t_em_list,
        "t_marg_list": t_marg_list,
    }
    return summary, stats


def time_full_api_reference(key_paths_root: jax.Array, cfg: ExperimentConfig) -> float:
    """Return wall seconds for one call to ``_compute_ensemble_path_results``."""
    t0 = time.perf_counter()
    _ = _compute_ensemble_path_results(key_paths_root, cfg)
    t1 = time.perf_counter()
    return t1 - t0


def print_report(
    cfg: ExperimentConfig,
    stats: dict,
    full_api_s: float | None,
) -> None:
    """Print timing report and component fractions."""
    n = stats["n_paths"]
    wall = stats["loop_wall_s"]
    sd = stats["sum_disc_s"]
    se = stats["sum_em_s"]
    sm = stats["sum_marg_s"]

    def pct(x: float) -> float:
        return 100.0 * x / wall if wall > 0.0 else 0.0

    print("\nInstrumented ensemble construction")
    print(
        "num_sample_paths=",
        n,
        " delta_for_path=",
        stats["delta_path"],
        " t_final=",
        cfg.time.t_final,
        sep="",
    )
    print("loop_wall_s (instrumented loop):".ljust(40), f"{wall:.6f}")
    print("sum prepare_coupled_discretization:".ljust(40), f"{sd:.6f}", f"({pct(sd):.1f}% of wall)")
    print("sum euler_maruyama (coarse EM):".ljust(40), f"{se:.6f}", f"({pct(se):.1f}% of wall)")
    print("sum marginalised inner solve:".ljust(40), f"{sm:.6f}", f"({pct(sm):.1f}% of wall)")
    print("sum of three parts:".ljust(40), f"{sd + se + sm:.6f}")
    print("mean per path — disc / em / marg (s):".ljust(40), end=" ")
    print(
        f"{stats['mean_disc_s']:.6f}",
        "/",
        f"{stats['mean_em_s']:.6f}",
        "/",
        f"{stats['mean_marg_s']:.6f}",
    )
    denom = sd + se + sm
    print(
        "marg fraction of (disc+em+marg):".ljust(40),
        f"{100.0 * sm / max(denom, 1e-12):.1f}%",
    )

    print("\nPer-path marginalised times: min / mean / max (s)")
    print(
        min(stats["t_marg_list"]),
        statistics.mean(stats["t_marg_list"]),
        max(stats["t_marg_list"]),
    )

    if full_api_s is not None:
        print(
            "\nReference _compute_ensemble_path_results (one call):".ljust(40),
            f"{full_api_s:.6f}s",
        )
        print(
            "ratio reference_wall / instrumented_loop_wall:".ljust(40),
            f"{full_api_s / max(wall, 1e-12):.3f}",
        )


def main() -> None:
    """Run ensemble timing and print a report."""
    args = parse_args()
    cfg = build_experiment_config(args)
    key_root = split_ensemble_key(cfg)

    full_api_s: float | None = None
    if args.verify_full_api:
        full_api_s = time_full_api_reference(key_root, cfg)

    _summary, stats = time_ensemble_path_construction(key_root, cfg)
    print_report(cfg, stats, full_api_s)

    if args.verify_full_api:
        print(
            "\nNote: with --verify-full-api, the ensemble was computed twice "
            "(reference API, then instrumented loop). Use a separate run without "
            "the flag for a single-pass timing only."
        )


if __name__ == "__main__":
    main()