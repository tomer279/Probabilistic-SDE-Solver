"""
Benes SDE benchmark: Marginalised-GSF vs GSF vs Euler-Maruyama (EM).

Exports
-------
TimeConfig
    Time-discretization settings for path generation and convergence tests.
MonteCarloConfig
    Monte Carlo sampling configuration for error statistics.
GSFSolverConfig
    Algorithm-2 Gaussian SDE Filter numerical parameters.
MarginalisedSolverConfig
    Algorithm-4 Marginalised Gaussian SDE Filter numerical parameters.
ExperimentConfig
    Top-level configuration for the benchmark.
run_experiment
    Compute path samples and error curves over step sizes.
plot_results
    Plot trajectories and log-log convergence curves.
main
    Run the full benchmark and save figures.

Notes
-----
- EM/GSF errors are reported as strong local/global errors using coupled noise.
- Marginalised errors are reported as weak local/global errors:
  absolute difference of sample means versus fine EM reference.
  
Future Work
-----------
- Performance: this benchmark currently has runtime bottlenecks (especially Monte
  Carlo loops and repeated per-seed path construction). Refactor and optimize this
  module in a follow-up pass.
- Brownian backend: migrate Brownian path/coefficient construction to JAX-native
  operations to improve compatibility with JAX transformations (e.g., vmap/jit)
  and reduce host/device conversion overhead.
"""
# pylint: disable=wrong-import-position
from pathlib import Path
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_STR = str(_REPO_ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

from dataclasses import dataclass
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
    euler_maruyama_from_increments,
    fit_power_laws_for_error_series,
    format_power_law_text,
    plot_error_data_series,
    plot_fitted_error_series,
    prepare_coupled_discretization,
    resolve_mc_run_seed,
    strong_errors_from_paths,
)

from prob_sde import (
    SDESpec,
    solve_sde_marginalised
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
    """Time-discretization settings for simulation and convergence tests."""

    t_final: float = 1.0
    deltas: tuple[float, ...] = (2.0**-1, 2.0**-2, 2.0**-3, 2.0**-4)
    delta_for_path: Optional[float] = 0.01


@dataclass(frozen=True)
class MonteCarloConfig:
    """Monte Carlo settings for averaging error statistics."""

    num_sample_paths: int = 500
    seed: Optional[int] = None


@dataclass(frozen=True)
class GSFSolverConfig:
    """Algorithm-2 GSF settings."""

    measurement_noise: float = 1e-6
    sample_posterior_position: bool = False
    variance_floor: float = 1e-12
    initial_cov_scale: float = 1e-8


@dataclass(frozen=True)
class MarginalisedSolverConfig:
    """Algorithm-4 Marginalised-GSF settings."""

    sample_posterior_position: bool = True
    use_ekf1: bool = True
    variance_floor: float = 1e-12
    prior_diffusion: float = 1.0


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level benchmark configuration."""

    x0: float = 0.0
    time: TimeConfig = TimeConfig()
    mc: MonteCarloConfig = MonteCarloConfig()
    gsf: GSFSolverConfig = GSFSolverConfig()
    marginalised: MarginalisedSolverConfig = MarginalisedSolverConfig()


def drift(x, _t):
    """Benes drift."""
    return jnp.tanh(x)


def diffusion(_x, _t):
    """Benes diffusion (constant one)."""
    return jnp.array(1)


def _marginalised_path(key, delta, cfg):
    """Simulate one Algorithm-4 Marginalised-GSF path."""
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
    _, traj = solve_sde_marginalised(key, sde, marg_cfg)
    return traj


def _weak_ingredients_from_paths(x_ref, x_marg, block_size):
    """Return local/global values used for weak mean error aggregation."""
    ref_local = x_ref[block_size]
    ref_global = x_ref[-1]
    marg_local = x_marg[1]
    marg_global = x_marg[-1]
    return ref_local, ref_global, marg_local, marg_global


def g_weak_observable(x):
    """Article weak functional g(X) = X X^T. Scalar SDE -> g(x) = x^2."""
    x = jnp.asarray(x)
    if x.shape == () or x.ndim == 0:
        return x * x
    col = jnp.reshape(x, (-1, 1))
    return col @ col.T


def weak_error_g_hat(diff_per_seed):
    """
    diff_per_seed[k] = g(scheme)_k - g(reference)_k for paired paths.

    Article: (1/N) * || sum_k diff_k || = || mean_k(diff_k) ||
    for Frobenius / absolute value on R^{d x d} / R.
    """
    arr = np.asarray(diff_per_seed, dtype=float)
    if arr.ndim == 1:
        return float(np.abs(np.mean(arr)))
    mean_diff = np.mean(arr, axis=0)
    return float(np.linalg.norm(mean_diff, ord="fro"))


def one_seed_stats(root_key, delta, cfg, x_marg):
    """Per-seed strong errors + weak-g errors with precomputed marginalised path."""
    disc, x_ref, x_em, x_gsf = _simulate_coupled_non_marginalised_paths(
        root_key,
        delta,
        cfg,
    )

    strong = strong_errors_from_paths(x_ref, x_em, x_gsf, disc.block_size)
    ref_local = x_ref[disc.block_size]
    ref_global = x_ref[-1]

    weak_em = _weak_diffs_against_ref_local_global(x_em, ref_local, ref_global)
    weak_gsf = _weak_diffs_against_ref_local_global(x_gsf, ref_local, ref_global)
    weak_marg = _weak_diffs_against_ref_local_global(x_marg, ref_local, ref_global)

    return (*strong, *weak_em, *weak_gsf, *weak_marg)


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
        disc.coeffs_list,
    )
    return disc, x_ref, x_em, x_gsf


def _weak_diffs_against_ref_local_global(x_scheme, ref_local, ref_global):
    """Return weak-observable differences (local, global) for one scheme."""
    return (
        float(g_weak_observable(x_scheme[1]) - g_weak_observable(ref_local)),
        float(g_weak_observable(x_scheme[-1]) - g_weak_observable(ref_global)),
    )


def estimate_errors_for_delta(base_key, delta, cfg, progress_bar=None):
    """Strong errors + article weak-g errors for one delta."""
    keys = jax.random.split(base_key, cfg.mc.num_sample_paths)

    marg_keys = jax.vmap(lambda key_i: jax.random.fold_in(key_i, 2))(keys)
    marg_paths = _marginalised_paths_batch(marg_keys, delta, cfg)
    jax.block_until_ready(marg_paths)

    vals = _collect_seed_stats(keys, marg_paths, delta, cfg, progress_bar)

    arr = np.asarray(vals, dtype=float)

    strong_cols = (0, 1, 2, 3)
    weak_cols = (4, 5, 6, 7, 8, 9)

    strong_vals = _mean_columns(arr, strong_cols)
    weak_vals = _weak_g_columns(arr, weak_cols)
    return (*strong_vals, *weak_vals)


def _mean_columns(arr, columns):
    """Return tuple of column means from a 2D numpy array."""
    return tuple(float(np.mean(arr[:, col])) for col in columns)


def _weak_g_columns(arr, columns):
    """Return tuple of weak-g estimates from selected columns."""
    return tuple(weak_error_g_hat(arr[:, col]) for col in columns)


def _collect_seed_stats(keys, marg_paths, delta, cfg, progress_bar=None):
    """Collect per-seed error statistics using precomputed marginalised paths."""
    vals = []
    for key, x_marg in zip(keys, marg_paths):
        vals.append(one_seed_stats(key, delta, cfg, x_marg))
        if progress_bar is not None:
            progress_bar.update(1)
    return vals


def _build_gsf_run_config_for_facade(cfg, coeffs_list):
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
        coeffs_list=coeffs_list,
        return_uncertainty=False,
    )


def gsf_path_coupled_via_sde_solver(
        root_key,
        cfg,
        delta,
        num_steps,
        coeffs_list):
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
    run_cfg = _build_gsf_run_config_for_facade(cfg, coeffs_list=coeffs_list)
    solver_cfg = SDESolverConfig(
        method="gsf",
        grid=TimeGridConfig(delta=float(delta), num_steps=int(num_steps), t0=0.0),
        gsf=run_cfg,
    )
    result = solve_sde(root_key, sde, solver_cfg)
    return result.trajectory


def run_experiment(cfg):
    """Run ensemble path diagnostics and Monte Carlo error estimation."""
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


def _collect_ensemble_paths(path_keys, delta_path, cfg):
    """Return EM and marginalised path lists for ensemble diagnostics."""
    em_paths = []

    for key_i in tqdm(path_keys, desc="Ensemble EM paths", leave=True):
        disc = prepare_coupled_discretization(key_i, delta_path, cfg.time.t_final)
        em_paths.append(
            euler_maruyama_from_increments(
                drift, diffusion, disc.dw_coarse, delta_path, cfg.x0
            )
        )

    marg_keys = jax.vmap(lambda key_i: jax.random.fold_in(key_i, 2))(path_keys)
    marg_paths = _marginalised_paths_batch(marg_keys, delta_path, cfg)

    return em_paths, list(marg_paths)


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
    """Strong error curves + article weak-g error curves over deltas."""
    series = {
        "em_local": [],
        "em_global": [],
        "gsf_local": [],
        "gsf_global": [],
        "weak_em_local_g": [],
        "weak_em_global_g": [],
        "weak_gsf_local_g": [],
        "weak_gsf_global_g": [],
        "weak_marg_local_g": [],
        "weak_marg_global_g": [],
    }

    total_mc = len(cfg.time.deltas) * cfg.mc.num_sample_paths
    mc_bar = tqdm(total=total_mc, desc="MC samples", leave=True)
    delta_iter = tqdm(cfg.time.deltas, desc="Step sizes", leave=True)

    for idx, delta in enumerate(delta_iter):
        key_i = jax.random.fold_in(key_mc, idx)
        estimates = estimate_errors_for_delta(key_i, delta, cfg, progress_bar=mc_bar)
        for name, value in zip(series.keys(), estimates):
            series[name].append(value)
        delta_iter.set_postfix(
            em_g=round(estimates[1], 5),
            gsf_g=round(estimates[3], 5),
            marg_wg=round(estimates[9], 5),
        )

    mc_bar.close()
    return {
        "deltas": np.asarray(cfg.time.deltas),
        **{name: np.asarray(values) for name, values in series.items()},
    }



def plot_results(results):
    """Create ensemble and convergence figures."""
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
    """Plot log-log errors and fitted rates."""
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
    ax.set_title("Strong (EM/GSF) and weak (Marginalised) errors")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)


def print_weak_g_error_comparison_table(results):
    """Print EM vs GSF vs Marginalised using article weak functional g(X)=X X^T."""
    deltas = np.asarray(results["deltas"], dtype=float)
    w_em_l = np.asarray(results["weak_em_local_g"], dtype=float)
    w_gsf_l = np.asarray(results["weak_gsf_local_g"], dtype=float)
    w_m_l = np.asarray(results["weak_marg_local_g"], dtype=float)
    w_em_g = np.asarray(results["weak_em_global_g"], dtype=float)
    w_gsf_g = np.asarray(results["weak_gsf_global_g"], dtype=float)
    w_m_g = np.asarray(results["weak_marg_global_g"], dtype=float)

    headers = (
        "delta",
        "W_EM loc",
        "W_GSF loc",
        "W_Marg loc",
        "Best loc",
        "W_EM glob",
        "W_GSF glob",
        "W_Marg glob",
        "Best glob",
    )
    widths = (10, 12, 12, 12, 10, 12, 12, 13, 11)

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

    print("\nWeak error g(X)=X X^T vs fine EM (article estimator; scalar g(x)=x^2)")
    print(row(headers))
    print(separator)

    for i in range(deltas.shape[0]):
        print(
            row(
                (
                    f"{deltas[i]:.6f}",
                    fmt_num(w_em_l[i]),
                    fmt_num(w_gsf_l[i]),
                    fmt_num(w_m_l[i]),
                    winner_name(w_em_l[i], w_gsf_l[i], w_m_l[i]),
                    fmt_num(w_em_g[i]),
                    fmt_num(w_gsf_g[i]),
                    fmt_num(w_m_g[i]),
                    winner_name(w_em_g[i], w_gsf_g[i], w_m_g[i]),
                )
            )
        )


def main():
    """Run Benes SDE benchmark and plots for EM, GSF, and Marginalised."""
    cfg = ExperimentConfig()
    results = run_experiment(cfg)
    print_weak_g_error_comparison_table(results)
    plot_results(results)


if __name__ == "__main__":
    main()
