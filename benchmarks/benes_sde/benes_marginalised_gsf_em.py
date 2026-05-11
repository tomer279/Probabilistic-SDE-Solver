"""
Benes SDE benchmark: terminal weak-g comparison for EM, GSF, and Marginalised.

This module runs a coupled Monte Carlo benchmark on the scalar Benes SDE and
reports article-style weak error at terminal time T using
    g(x) = x^2  (scalar case of g(X) = X X^T).

For each step size delta, the benchmark computes paired paths (fine EM reference,
coarse EM, GSF, and marginalised) and estimates:

    eps_wg(delta) = || (1/N) * sum_k [ g(X^scheme_{T,k}) - g(X^ref_{T,k}) ] ||.

Reported weak series
--------------------
- weak_em_g
    Terminal weak-g error for coarse Euler-Maruyama vs fine EM reference.
- weak_gsf_g
    Terminal weak-g error for GSF vs fine EM reference.
- weak_marg_g
    Terminal weak-g error for marginalised solver vs fine EM reference.

Exports
-------
TimeConfig
    Time-grid settings for path panel and convergence sweeps.
MonteCarloConfig
    Monte Carlo sampling settings.
GSFSolverConfig
    Algorithm-2 Gaussian SDE Filter settings.
MarginalisedSolverConfig
    Algorithm-4 marginalised solver settings.
ExperimentConfig
    Top-level benchmark configuration.
run_experiment
    Compute ensemble summaries and weak-g convergence series.
plot_results
    Plot ensemble trajectories and weak-g log-log curves.
print_weak_g_error_comparison_table
    Print terminal weak-g table for EM/GSF/Marginalised.
main
    CLI entry point.

Presets
-------
- publish: full benchmark defaults.
- smoke: reduced settings for faster local iteration.

Notes
-----
SDE fields ``drift`` and ``diffusion`` are imported from ``benchmarks.benes_sde.benes_dynamics``.
Chunked Monte Carlo uses ``resolve_mc_chunk_size``, ``chunked_accumulate_keys``, and
``chunked_map_concat_keys`` from ``benchmarks.benchmark_utils``.
"""
# pylint: disable=wrong-import-position
from pathlib import Path
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_STR = str(_REPO_ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)


import argparse
from dataclasses import dataclass, replace
from typing import Optional

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from benchmarks.benchmark_utils import (
    EM_GSF_MARG_WEAK_ERROR_SERIES_SPECS,
    POWER_LAW_FIT_HELP_TEXT,
    chunked_accumulate_keys,
    chunked_map_concat_keys,
    coeffs_array_to_list,
    euler_maruyama_from_increments,
    fit_power_laws_for_error_series,
    format_power_law_text,
    plot_error_data_series,
    plot_fitted_error_series,
    prepare_coupled_discretization,
    resolve_mc_chunk_size,
    resolve_mc_run_seed,
)

from benchmarks.benes_sde.benes_dynamics import drift, diffusion

from prob_sde import (
    SDESpec,
)
from prob_sde.filtering.sde.gaussian_sde_filter import (
    GaussianSDEFilterConfig
)
from prob_sde.filtering.sde.marginalised import (
    MarginalisedConfig,
    solve_sde_marginalised_batch
)

from prob_sde.solvers.sde_solver import (
    TimeGridConfig,
    GSFRunConfig,
    SDESolverConfig,
    solve_sde,
)

@dataclass(frozen=True)
class TimeConfig:
    """
    Time-discretization settings for path generation and convergence tests.

    Attributes
    ----------
    t_final : float
        Final simulation time.
    deltas : tuple[float, ...]
        Coarse step sizes used for Monte Carlo strong-error estimation.
    delta_for_path : float | None
        Step size used for single-path visualization. If ``None``, the first
        value in ``deltas`` is used.
    """

    t_final: float = 1.0
    deltas: tuple[float, ...] = (2.0**-3, 2.0**-4, 2.0 ** -5, 2.0**-6, 2.0**-7)
    delta_for_path: Optional[float] = 0.01


@dataclass(frozen=True)
class MonteCarloConfig:
    """
    Monte Carlo settings for weak-g error statistics and ensemble paths.

    Attributes
    ----------
    num_sample_paths : int
        Number of independent PRNG seeds per coarse step size (Monte Carlo) and
        number of ensemble trajectories for the path panel when that panel uses
        ``cfg.mc.num_sample_paths``.
    seed : int | None
        Run-level random seed. If ``None``, a fresh seed is generated.
    chunk_size : int | None
        Optional number of seeds per batch for chunked ``vmap``/``jit`` Monte Carlo
        and for chunked ensemble EM construction. If ``None``,
        :func:`benchmarks.benchmark_utils.resolve_mc_chunk_size` picks a default
        from ``num_sample_paths``.
    """

    num_sample_paths: int = 5000
    seed: Optional[int] = None
    chunk_size: Optional[int] = None


@dataclass(frozen=True)
class GSFSolverConfig:
    """
    Algorithm-2 Gaussian SDE Filter settings for coupled GSF trajectories.

    Attributes
    ----------
    measurement_noise : float
        Measurement noise scale used in the GSF update model.
    sample_posterior_position : bool
        If ``True``, sample posterior position; otherwise use posterior mean.
    variance_floor : float
        Lower bound applied to covariance terms for numerical stability.
    initial_cov_scale : float
        Initial covariance scaling used at filter initialization.
    """

    measurement_noise: float = 1e-6
    sample_posterior_position: bool = True
    variance_floor: float = 1e-12
    initial_cov_scale: float = 1e-8


@dataclass(frozen=True)
class MarginalisedSolverConfig:
    """
    Algorithm-4 marginalised solver settings.

    Attributes
    ----------
    sample_posterior_position : bool
        If ``True``, sample posterior position; otherwise use posterior mean.
    use_ekf1 : bool
        If ``True``, use EKF1 linearization mode; otherwise use the EKF0-style mode
        implemented by the marginalised solver.
    variance_floor : float
        Lower bound applied to covariance terms for numerical stability.
    prior_diffusion : float
        Diffusion scale parameter for the marginalised prior model.
    """

    sample_posterior_position: bool = True
    use_ekf1: bool = True
    variance_floor: float = 1e-12
    prior_diffusion: float = 1.0


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Top-level benchmark configuration.

    Attributes
    ----------
    x0 : float
        Initial scalar SDE state.
    time : TimeConfig
        Time-grid controls for ensemble paths and convergence sweeps.
    mc : MonteCarloConfig
        Monte Carlo sampling controls (including chunked batching settings).
    gsf : GSFSolverConfig
        Coupled GSF solver settings.
    marginalised : MarginalisedSolverConfig
        Marginalised solver settings.
    """

    x0: float = 0.0
    time: TimeConfig = TimeConfig()
    mc: MonteCarloConfig = MonteCarloConfig()
    gsf: GSFSolverConfig = GSFSolverConfig()
    marginalised: MarginalisedSolverConfig = MarginalisedSolverConfig()


def g_weak_observable(x):
    """Return the weak observable ``g(X)=XX^T`` (scalar case: ``g(x)=x^2``).

    For scalar states this returns a scalar square. For vector states it returns the
    outer product ``x x^T``.
    """
    x = jnp.asarray(x)
    if x.shape == () or x.ndim == 0:
        return x * x
    col = jnp.reshape(x, (-1, 1))
    return col @ col.T


def estimate_errors_for_delta(base_key, delta, cfg, progress_bar=None):
    """Compute article weak-g error estimates (EM/GSF/Marginalised) for one delta.

    For each seed, forms terminal ``g(x)=x^2`` differences vs the fine EM reference
    for EM, GSF, and marginalised paths. Scalar weak-g reports use the absolute
    mean of stacked differences across seeds.

    Seeds are processed in contiguous chunks. Chunk size comes from
    ``resolve_mc_chunk_size`` using ``cfg.mc.num_sample_paths`` and optional
    ``cfg.mc.chunk_size``. Each chunk batches marginalised trajectories, then sums
    per-seed weak-difference vectors with ``jax.vmap`` and ``jax.jit`` inside
    :func:`_chunked_weak_sum` via :func:`benchmarks.benchmark_utils.chunked_accumulate_keys`.

    Parameters
    ----------
    base_key : jax.Array
        Root PRNG key for the Monte Carlo batch at this step size.
    delta : float
        Coarse step size.
    cfg : ExperimentConfig
        Full benchmark configuration.
    progress_bar : object, optional
        If given, must expose ``update(int)``; updated by the number of seeds
        completed per chunk.

    Returns
    -------
    dict[str, float]
        Keys ``weak_em_g``, ``weak_gsf_g``, ``weak_marg_g``.
    """
    keys = jax.random.split(base_key, cfg.mc.num_sample_paths)

    chunk_size = resolve_mc_chunk_size(
        cfg.mc.num_sample_paths, cfg.mc.chunk_size)
    total = _chunked_weak_sum(
        keys=keys,
        delta=delta,
        cfg=cfg,
        chunk_size=chunk_size,
        progress_bar=progress_bar,
    )

    n = float(keys.shape[0])
    means = jax.device_get(total / n)
    weak_em_g = abs(float(means[0]))
    weak_gsf_g = abs(float(means[1]))
    weak_marg_g = abs(float(means[2]))
    return {
        "weak_em_g": weak_em_g,
        "weak_gsf_g": weak_gsf_g,
        "weak_marg_g": weak_marg_g,
    }


def _chunked_weak_sum(keys, delta, cfg, chunk_size, progress_bar=None):
    """Return summed terminal weak-g difference vector ``(3,)`` over all seeds.

    Delegates chunk slicing and progress updates to
    :func:`benchmarks.benchmark_utils.chunked_accumulate_keys`. Each chunk builds
    marginalised paths for folded keys, then ``jit``-sums ``vmap``'d weak
    differences for EM, GSF, and marginalised vs the coupled reference.
    """
    batch_sum_jit = jax.jit(
        lambda ks, mp: _batch_seed_weak_sum(ks, mp, delta, cfg)
    )

    def weak_chunk_vector(chunk_keys):
        marg_keys_chunk = jax.vmap(
            lambda key_i: jax.random.fold_in(key_i, 2)
        )(chunk_keys)
        marg_paths_chunk = _marginalised_paths_batch(
            marg_keys_chunk, delta, cfg)
        return batch_sum_jit(chunk_keys, marg_paths_chunk)

    return chunked_accumulate_keys(
        keys, chunk_size, weak_chunk_vector, progress_bar=progress_bar)


def _batch_seed_weak_sum(keys, marg_paths, delta, cfg):
    """Sum vectorized one-seed weak-difference stacks over a key batch; shape ``(3,)``."""
    vals = jax.vmap(
        lambda k, xm: _one_seed_weak_diffs_stacked(k, xm, delta, cfg),
        in_axes=(0, 0),
    )(keys, marg_paths)
    return jnp.sum(vals, axis=0)


def _one_seed_weak_diffs_stacked(root_key, x_marg, delta, cfg):
    """Stack one-seed terminal weak differences into shape ``(3,)``."""
    x_ref, x_em, x_gsf = _simulate_coupled_non_marginalised_paths(
        root_key,
        delta,
        cfg,
    )
    weak_em = _weak_diffs_against_ref_terminal(x_em, x_ref)
    weak_gsf = _weak_diffs_against_ref_terminal(x_gsf, x_ref)
    weak_marg = _weak_diffs_against_ref_terminal(x_marg, x_ref)
    return jnp.stack([weak_em, weak_gsf, weak_marg])


def _simulate_coupled_non_marginalised_paths(root_key, delta, cfg):
    """Return discretization, reference, EM, and GSF paths for one seed."""
    disc = prepare_coupled_discretization(root_key, delta, cfg.time.t_final)
    x_ref = euler_maruyama_from_increments(
        drift, diffusion, disc.dw_ref, disc.delta_ref, cfg.x0
    )
    x_em = euler_maruyama_from_increments(
        drift, diffusion, disc.dw_coarse, delta, cfg.x0
    )
    x_gsf = gsf_path_coupled_via_sde_solver(
        jax.random.fold_in(root_key, 1),
        cfg,
        delta,
        disc.num_steps,
        disc.coeffs,
    )
    return x_ref, x_em, x_gsf


def _weak_diffs_against_ref_terminal(x_scheme, x_ref):
    """Return terminal-time weak-observable difference for one scheme."""
    return g_weak_observable(x_scheme[-1]) - g_weak_observable(x_ref[-1])


def gsf_path_coupled_via_sde_solver(
        root_key,
        cfg,
        delta,
        num_steps,
        coeffs):
    """One GSF trajectory via ``solve_sde`` with caller-supplied parabolic coefficients.

    Parameters
    ----------
    root_key : jax.Array
        PRNG key for the GSF rollout (split into per-step keys inside ``solve_sde``).
    cfg : ExperimentConfig
        Benchmark configuration (``x0`` and ``cfg.gsf`` filter options).
    delta : float
        Coarse step size.
    num_steps : int
        Number of coarse steps; must equal ``len(coeffs_list)``.
    coeffs_list : Sequence
        Per-interval coefficients from ``prepare_coupled_discretization``.

    Returns
    -------
    jax.Array
        Trajectory of shape ``(num_steps + 1,)``.
    """
    sde = SDESpec.from_args(drift, diffusion, jnp.asarray(cfg.x0), bm_factory=None)
    run_cfg = _build_gsf_run_config_for_facade(cfg, coeffs=coeffs)
    solver_cfg = SDESolverConfig(
        method="gsf",
        grid=TimeGridConfig(delta=float(delta), num_steps=int(num_steps), t0=0.0),
        gsf=run_cfg,
    )
    result = solve_sde(root_key, sde, solver_cfg)
    return result.trajectory


def _build_gsf_run_config_for_facade(cfg, coeffs):
    """Map benchmark ``GSFSolverConfig`` to facade ``GSFRunConfig``."""
    return GSFRunConfig(
        prior_scale=1.0,
        filter_config=GaussianSDEFilterConfig(
            measurement_noise=cfg.gsf.measurement_noise,
            sample_posterior_position=cfg.gsf.sample_posterior_position,
            variance_floor=cfg.gsf.variance_floor,
            initial_cov_scale=cfg.gsf.initial_cov_scale,
            return_beta_coeffs=False,
            ekf_mode="ekf1",
        ),
        coeffs_list=coeffs_array_to_list(coeffs),
        return_uncertainty=False,
    )


def run_experiment(cfg):
    """Run the full benchmark pipeline and return plot/report inputs.

    Resolves the run seed, computes ensemble path summaries (EM and marginalised),
    computes Monte Carlo weak-g convergence curves over configured ``deltas``, and
    merges both result blocks into a single dictionary.

    Parameters
    ----------
    cfg : ExperimentConfig
        Full benchmark configuration.

    Returns
    -------
    dict[str, np.ndarray]
        Combined results for plotting and reporting, including ensemble summaries
        (means/quantiles/time grid) and weak-g error series over ``deltas``.
    """
    run_seed = resolve_mc_run_seed(cfg.mc.seed)
    print("Using seed =", run_seed)

    key_paths, key_mc = jax.random.split(jax.random.PRNGKey(run_seed), 2)

    ensemble_results = _compute_ensemble_path_results(key_paths, cfg)
    mc_results = _compute_mc_error_results(key_mc, cfg)
    return {**ensemble_results, **mc_results}


def _compute_ensemble_path_results(key_paths, cfg):
    """Compute ensemble mean and quantile bands for EM and Marginalised paths."""
    delta_path = _resolve_delta_path(cfg)
    path_keys = jax.random.split(key_paths, cfg.mc.num_sample_paths)
    em_paths, marg_paths = _collect_ensemble_paths(path_keys, delta_path, cfg)
    return _ensemble_summary_arrays(em_paths, marg_paths, cfg.time.t_final)


def _resolve_delta_path(cfg):
    """Return delta used for ensemble paths."""
    if cfg.time.delta_for_path is not None:
        return float(cfg.time.delta_for_path)
    return float(cfg.time.deltas[0])


def _one_ensemble_em_path(key, delta_path, cfg):
    """One coupled coarse-grid EM path for ensemble panel (same math as the old loop)."""
    disc = prepare_coupled_discretization(key, delta_path, cfg.time.t_final)
    return euler_maruyama_from_increments(
        drift, diffusion, disc.dw_coarse, delta_path, cfg.x0
    )


def _ensemble_em_paths_chunked(path_keys, delta_path, cfg):
    """Return all ensemble coarse EM paths, shape ``(N, num_steps + 1)``.

    Builds chunked ``jax.jit`` + ``jax.vmap`` EM batches and concatenates chunk
    outputs with :func:`benchmarks.benchmark_utils.chunked_map_concat_keys`.
    Chunk size follows ``resolve_mc_chunk_size`` from ``cfg.mc``.
    """
    chunk_size = resolve_mc_chunk_size(
        cfg.mc.num_sample_paths, cfg.mc.chunk_size
    )

    def one_path_for_chunk(key):
        return _one_ensemble_em_path(key, delta_path, cfg)

    def run_chunk_impl(keys_chunk):
        return jax.vmap(one_path_for_chunk)(keys_chunk)

    run_chunk = jax.jit(run_chunk_impl)

    with tqdm(
        total=int(path_keys.shape[0]),
        desc="Ensemble EM paths (chunked)",
        leave=True,
    ) as progress_bar:
        return chunked_map_concat_keys(
            path_keys,
            chunk_size,
            run_chunk,
            progress_bar=progress_bar,
        )


def _collect_ensemble_paths(path_keys, delta_path, cfg):
    """Return EM and marginalised ensemble path arrays for summary statistics.
    
    EM paths come from ``_ensemble_em_paths_chunked``. Marginalised paths are sampled
    in batch via ``_marginalised_paths_batch``. Both outputs are converted to NumPy
    arrays for downstream mean/quantile aggregation.
    """
    em_batch = _ensemble_em_paths_chunked(path_keys, delta_path, cfg)

    marg_keys = jax.vmap(lambda key_i: jax.random.fold_in(key_i, 2))(path_keys)
    marg_paths = _marginalised_paths_batch(marg_keys, delta_path, cfg)

    em_np = np.asarray(jax.device_get(em_batch), dtype=float)
    marg_np = np.asarray(jax.device_get(marg_paths), dtype=float)
    return em_np, marg_np


def _marginalised_paths_batch(keys, delta, cfg):
    """Simulate many independent Algorithm-4 Marginalised-GSF paths."""
    num_steps = int(round(cfg.time.t_final / delta))
    sde = SDESpec.from_args(drift, diffusion, jnp.asarray(cfg.x0), bm_factory=None)
    marg_cfg = MarginalisedConfig(
        delta=float(delta),
        num_steps=num_steps,
        sample_posterior_position=cfg.marginalised.sample_posterior_position,
        use_ekf1=cfg.marginalised.use_ekf1,
        variance_floor=cfg.marginalised.variance_floor,
        prior_diffusion=cfg.marginalised.prior_diffusion,
        return_uncertainty=False,
    )
    _, trajectories = solve_sde_marginalised_batch(keys, sde, marg_cfg)
    return trajectories


def _ensemble_summary_arrays(em_paths, marg_paths, t_final):
    """Convert paths to arrays and compute means/quantiles/time grid."""
    em_arr = np.asarray(em_paths, dtype=float)
    marg_arr = np.asarray(marg_paths, dtype=float)

    em_mean, em_q05, em_q95 = _mean_and_quantiles(em_arr, 0.05, 0.95)
    marg_mean, marg_q05, marg_q95 = _mean_and_quantiles(marg_arr, 0.05, 0.95)
    ts = np.asarray(jnp.linspace(0.0, t_final, em_arr.shape[1]))

    return {
        "ts_ens": ts,
        "em_mean": em_mean,
        "em_q05": em_q05,
        "em_q95": em_q95,
        "marg_mean": marg_mean,
        "marg_q05": marg_q05,
        "marg_q95": marg_q95,
    }


def _mean_and_quantiles(paths_arr, q_low, q_high):
    """Return mean and pointwise quantiles across an ensemble axis."""
    mean = np.mean(paths_arr, axis=0)
    low = np.quantile(paths_arr, q_low, axis=0)
    high = np.quantile(paths_arr, q_high, axis=0)
    return mean, low, high


def _compute_mc_error_results(key_mc, cfg):
    """Compute weak-g error curves over deltas for EM, GSF, and Marginalised."""
    series = {
        "weak_em_g": [],
        "weak_gsf_g": [],
        "weak_marg_g": []
    }

    total_mc = len(cfg.time.deltas) * cfg.mc.num_sample_paths
    mc_bar = tqdm(total=total_mc, desc="MC samples", leave=True)
    delta_iter = tqdm(cfg.time.deltas, desc="Step sizes", leave=True)

    for idx, delta in enumerate(delta_iter):
        key_i = jax.random.fold_in(key_mc, idx)
        estimates = estimate_errors_for_delta(
            key_i, delta, cfg, progress_bar=mc_bar)
        for name, value in estimates.items():
            series[name].append(value)
        delta_iter.set_postfix(
            em_wg=round(estimates['weak_em_g'], 5),
            gsf_wg=round(estimates['weak_gsf_g'], 5),
            marg_wg=round(estimates['weak_marg_g'], 5),
        )

    mc_bar.close()
    return {
        "deltas": np.asarray(cfg.time.deltas),
        **{name: np.asarray(values) for name, values in series.items()},
    }



def plot_results(results):
    """Render and save the benchmark figure with ensemble and weak-g panels.

    Creates a two-panel figure:
    1. ensemble means with 5-95% quantile bands for EM and marginalised paths,
    2. log-log weak-g convergence curves with optional fitted power-law overlays.

    Parameters
    ----------
    results : dict[str, np.ndarray | dict[str, float]]
        Output dictionary from :func:`run_experiment`.

    Returns
    -------
    None
        Saves ``benes_marginalised_gsf_em.png`` and displays the figure. If
        ``matplotlib`` is unavailable, prints a message and returns.
    """
    if plt is None:
        print("matplotlib is not installed. Install matplotlib to plot figures.")
        return

    fig, axes = plt.subplots(2,1, figsize=(10, 10))
    _plot_ensemble_panel(axes[0], results)
    _plot_error_panel(axes[1], results)

    fig.tight_layout()
    out = "benes_marginalised_gsf_em.png"
    fig.savefig(out, dpi=150)
    plt.show()
    plt.close(fig)
    print("Saved " + out)


def _plot_ensemble_panel(ax, results):
    """Plot EM and Marginalised ensemble means with quantile bands."""
    ts = results["ts_ens"]

    ax.fill_between(ts, results["em_q05"], results["em_q95"],
                    color="C0", alpha=0.18, label="EM 5-95%")
    ax.plot(ts, results["em_mean"], color="C0", linewidth=1.8, label="EM mean")

    ax.fill_between(
        ts,
        results["marg_q05"],
        results["marg_q95"],
        color="C4",
        alpha=0.18,
        label="Marginalised 5-95%",
    )
    ax.plot(ts, results["marg_mean"],
            color="C4", linewidth=1.8, linestyle="-.", label="Marginalised mean")

    ax.set_xlabel("t")
    ax.set_ylabel("X(t)")
    ax.set_title("Weak comparison: ensemble means and quantile bands")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def _plot_error_panel(ax, results):
    """Plot log-log terminal weak-g errors and fitted convergence rates."""
    deltas = np.asarray(results["deltas"])
    plot_error_data_series(
        ax, deltas, results, EM_GSF_MARG_WEAK_ERROR_SERIES_SPECS)

    if deltas.shape[0] >= 2:
        fits = fit_power_laws_for_error_series(
            deltas, results, EM_GSF_MARG_WEAK_ERROR_SERIES_SPECS )
        plot_fitted_error_series(
            ax, deltas, fits, EM_GSF_MARG_WEAK_ERROR_SERIES_SPECS)
        ax.text(
            0.02,
            0.02,
            format_power_law_text(fits, EM_GSF_MARG_WEAK_ERROR_SERIES_SPECS),
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
        )
    else:
        ax.text(
            0.02,
            0.02,
            POWER_LAW_FIT_HELP_TEXT,
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
        )

    ax.set_xlabel("delta")
    ax.set_ylabel("MC mean error")
    ax.set_title("Terminal weak-g errors (EM, GSF, Marginalised)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)


def print_weak_g_error_comparison_table(results):
    """Print terminal weak-g error comparison for EM, GSF, and Marginalised."""
    deltas = np.asarray(results["deltas"], dtype=float)
    w_em = np.asarray(results["weak_em_g"], dtype=float)
    w_gsf = np.asarray(results["weak_gsf_g"], dtype=float)
    w_marg = np.asarray(results["weak_marg_g"], dtype=float)

    headers = (
        "delta",
        "W_EM",
        "W_GSF",
        "W_Marg",
        "Best",
    )
    widths = (10, 12, 12, 12, 10)

    def fmt_num(x):
        return f"{x:.6e}"

    def winner_name(a, b, c):
        min_val = min(a, b, c)
        winners = []
        if a == min_val:
            winners.append("EM")
        if b == min_val:
            winners.append("GSF")
        if c == min_val:
            winners.append("Marg")
        return "/".join(winners)

    def row(values):
        cells = []
        for value, width in zip(values, widths):
            cells.append(str(value).rjust(width))
        return " | ".join(cells)

    separator = "-+-".join("-" * width for width in widths)

    print("\nWeak error g(X)=X X^T vs fine EM at terminal time T")
    print(row(headers))
    print(separator)

    for i in range(deltas.shape[0]):
        print(
            row(
                (
                    f"{deltas[i]:.6f}",
                    fmt_num(w_em[i]),
                    fmt_num(w_gsf[i]),
                    fmt_num(w_marg[i]),
                    winner_name(w_em[i], w_gsf[i], w_marg[i]),
                )
            )
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for benchmark presets."""
    parser = argparse.ArgumentParser(
        description="Run Benes EM/GSF/Marginalised benchmark."
    )
    parser.add_argument(
        "--preset",
        choices=("publish", "smoke"),
        default="publish",
        help="publish=full benchmark defaults, smoke=faster local iteration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional Monte Carlo seed override.",
    )
    parser.add_argument(
        "--num-sample-paths",
        type=int,
        default=None,
        help="Optional override for MonteCarloConfig.num_sample_paths.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for preset/overrides (useful for IDE runs).",
    )
    args, _unknown = parser.parse_known_args()
    return args


def prompt_runtime_options(args: argparse.Namespace) -> argparse.Namespace:
    """Interactively override runtime options for IDE-friendly runs."""
    print("\nInteractive benchmark options (press Enter to keep current value).")

    preset_in = input(f"Preset [publish/smoke] (current: {args.preset}): ").strip().lower()
    if preset_in in ("publish", "smoke"):
        args.preset = preset_in

    seed_in = input(f"Seed (current: {args.seed}): ").strip()
    if seed_in != "":
        args.seed = int(seed_in)

    n_in = input(
        "num_sample_paths override "
        f"(current: {args.num_sample_paths}, blank=use preset/default): "
    ).strip()
    if n_in != "":
        args.num_sample_paths = int(n_in)

    return args


def build_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    """Build ExperimentConfig from preset plus optional overrides."""
    cfg = ExperimentConfig()  # publish defaults stay unchanged

    if args.preset == "smoke":
        cfg = replace(
            cfg,
            time=replace(
                cfg.time,
                deltas=(2.0**-1, 2.0**-2),   # fewer deltas
                delta_for_path=2.0**-2,      # coarser path panel
            ),
            mc=replace(
                cfg.mc,
                num_sample_paths=40,         # fast local iteration
            ),
        )

    mc_cfg = cfg.mc
    if args.seed is not None:
        mc_cfg = replace(mc_cfg, seed=int(args.seed))
    if args.num_sample_paths is not None:
        mc_cfg = replace(mc_cfg, num_sample_paths=int(args.num_sample_paths))
    if mc_cfg is not cfg.mc:
        cfg = replace(cfg, mc=mc_cfg)

    return cfg


def main():
    """Run Benes SDE benchmark and plots for EM, GSF, and Marginalised."""
    args = parse_args()
    if args.interactive:
        args = prompt_runtime_options(args)

    cfg = build_experiment_config(args)
    results = run_experiment(cfg)
    print_weak_g_error_comparison_table(results)
    plot_results(results)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("--interactive")
    main()
