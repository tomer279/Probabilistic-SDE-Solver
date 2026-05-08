"""
Benes SDE benchmark: Mixture-GSF vs GSF vs Euler-Maruyama (EM).

Exports
-------
TimeConfig
    Time-discretization settings for path generation and convergence tests.
MonteCarloConfig
    Monte Carlo sampling configuration for error statistics.
SolverConfig
    Gaussian SDE Filter numerical parameters.
MixtureConfig
    Mixture-GSF numerical and sampling settings.
ExperimentConfig
    Top-level configuration for the benchmark.
run_experiment
    Compute path samples and strong local/global error curves.
plot_results
    Plot trajectories and log-log strong error convergence curves.
main
    Run the full benchmark and save figures.
"""
# pylint: disable=wrong-import-position
from typing import Optional
from dataclasses import dataclass

from pathlib import Path
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_STR = str(_REPO_ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from benchmarks.benchmark_utils import (
    EM_GSF_MGSF_ERROR_SERIES_SPECS,
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
    brownian_and_parabolic_coeffs,
    piecewise_parabolic_brownian,
    IWP2Prior,
    SDESpec,
    solve_sde_pathwise_mixture_with_coeffs,
)
from prob_sde.solvers.sde_solver import (
    TimeGridConfig,
    GSFRunConfig,
    MGSFRunConfig,
    SDESolverConfig,
    solve_sde,
)
from prob_sde.filtering.sde.gaussian_sde_filter import GaussianSDEFilterConfig
from prob_sde.filtering.sde.mixture_sde_filter import (
    EKFConfig,
    SolverConfig as MixtureSolverConfig,
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
    deltas = (2.0**-1, 2.0**-2, 2.0**-3, 2.0**-4, 2.0**-5)
    delta_for_path: Optional[float] = 0.01


@dataclass(frozen=True)
class MonteCarloConfig:
    """
    Monte Carlo sampling configuration for error statistics.

    Attributes
    ----------
    num_sample_paths : int
        Number of independent seeds used per step size.
    seed : int | None
        Run-level random seed. If ``None``, a fresh seed is generated.
    """

    num_sample_paths: int = 500
    seed: Optional[int] = None


@dataclass(frozen=True)
class SolverConfig:
    """
    Gaussian SDE Filter numerical parameters.

    Attributes
    ----------
    measurement_noise : float
        Measurement noise used by the filter update model.
    sample_posterior_position : bool
        Whether to sample from posterior position instead of using the mean.
    variance_floor : float
        Lower bound applied to covariance terms for numerical stability.
    initial_cov_scale : float
        Initial covariance scaling used for filter state initialization.
    """

    measurement_noise: float = 0.0
    sample_posterior_position: bool = True
    variance_floor: float = 1e-12
    initial_cov_scale: float = 1e-8
    posterior_ekf_mode: str = "ekf0"


@dataclass(frozen=True)
class MixtureConfig:
    """
    Mixture-GSF numerical and sampling settings.

    Attributes
    ----------
    num_paths_per_seed : int
        Number of pathwise mixture trajectories per Monte Carlo seed.
    use_ekf1_tk_initialization : bool
        Whether to use EKF1 initialization at t_k in Algorithm 3 stepping.
    """

    num_paths_per_seed: int = 500
    sample_posterior_position: bool = False
    use_ekf1_tk_initialization: bool = False
    posterior_ekf_mode: str = "ekf0"


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Top-level configuration for the Benes SDE benchmark.

    Attributes
    ----------
    x0 : float
        Initial SDE state.
    time : TimeConfig
        Time-grid and step-size controls.
    mc : MonteCarloConfig
        Monte Carlo sampling settings.
    solver : SolverConfig
        Gaussian SDE Filter settings.
    mixture : MixtureConfig
        Mixture-GSF settings.
    """

    x0: float = 0.0
    time: TimeConfig = TimeConfig()
    mc: MonteCarloConfig = MonteCarloConfig()
    solver: SolverConfig = SolverConfig()
    mixture: MixtureConfig = MixtureConfig()

def drift(x, t):
    """Benes drift: tanh(x)."""
    return jnp.tanh(x)

def diffusion(_x, _t):
    """Benes diffusion: constant one."""
    return jnp.array(1)


def gsf_path_facade_coupled(root_key, cfg, delta, num_steps, coeffs_list):
    """One GSF path via `solve_sde` with caller-supplied parabolic coefficients."""
    sde = SDESpec.from_args(drift, diffusion, jnp.asarray(cfg.x0), bm_factory=None)
    run_cfg = _build_gsf_run_config(cfg, coeffs_list=list(coeffs_list))
    solver_cfg = SDESolverConfig(
        method="gsf",
        grid=TimeGridConfig(delta=float(delta), num_steps=int(num_steps), t0=0.0),
        gsf=run_cfg,
    )
    result = solve_sde(root_key, sde, solver_cfg)
    return result.trajectory


def _build_gsf_run_config(cfg, coeffs_list=None):
    """Map experiment config to facade `GSFRunConfig` for `solve_sde(..., method=\"gsf\")`."""
    return GSFRunConfig(
        prior_scale=1.0,
        filter_config=GaussianSDEFilterConfig(
            measurement_noise=cfg.solver.measurement_noise,
            sample_posterior_position=cfg.solver.sample_posterior_position,
            variance_floor=cfg.solver.variance_floor,
            initial_cov_scale=cfg.solver.initial_cov_scale,
            return_beta_coeffs=False,
            ekf_mode=cfg.solver.posterior_ekf_mode,
        ),
        coeffs_list=coeffs_list,
        return_uncertainty=False,
    )


def _build_mixture_solver_inputs(delta, cfg, sampling_key=None):
    """Build Algorithm-3 Mixture-GSF inputs for one step size."""
    sde = SDESpec.from_args(
        drift,
        diffusion,
        jnp.asarray(cfg.x0),
        piecewise_parabolic_brownian,
    )
    prior = IWP2Prior(1.0, measurement_noise=cfg.solver.measurement_noise)
    solver_cfg = MixtureSolverConfig(
        delta=float(delta),
        num_steps=int(round(cfg.time.t_final / delta)),
        return_uncertainty=False,
        ekf=EKFConfig(
            use_ekf1_tk_initialization=cfg.mixture.use_ekf1_tk_initialization,
            posterior_ekf_mode=cfg.mixture.posterior_ekf_mode,
        ),
        sample_posterior_position=cfg.mixture.sample_posterior_position,
        sampling_key=sampling_key
    )
    return sde, prior, solver_cfg


def _simulate_mixture_paths(root_key, delta, cfg, num_paths):
    """Simulate multiple MGSF trajectories via facade (uncoupled noise per path)."""
    sde = SDESpec.from_args(
        drift,
        diffusion,
        jnp.asarray(cfg.x0),
        piecewise_parabolic_brownian,
    )
    num_steps = int(round(cfg.time.t_final / delta))
    grid = TimeGridConfig(delta=float(delta), num_steps=num_steps, t0=0.0)
    mgsf_run = MGSFRunConfig(
        prior_scale=1.0,
        return_uncertainty=False,
        use_ekf1_tk_initialization=cfg.mixture.use_ekf1_tk_initialization,
    )
    solver_cfg_template = SDESolverConfig(method="mgsf", grid=grid, mgsf=mgsf_run)

    keys = jax.random.split(jax.random.fold_in(root_key, 2), num_paths)
    paths = []
    for key_i in keys:
        result = solve_sde(key_i, sde, solver_cfg_template)
        paths.append(result.trajectory)
    return jnp.asarray(paths)


def _simulate_coupled_paths(root_key, delta, cfg, disc):
    """Simulate reference EM, coarse EM, and facade GSF paths for one seed."""
    x_ref = euler_maruyama_from_increments(
        drift, diffusion, disc.dw_ref, disc.delta_ref, cfg.x0)
    x_em = euler_maruyama_from_increments(
        drift, diffusion, disc.dw_coarse, delta, cfg.x0)
    key_gsf = jax.random.fold_in(root_key, 1)
    x_gsf = gsf_path_facade_coupled(
        key_gsf,
        cfg,
        delta,
        disc.num_steps,
        disc.coeffs_list,
    )
    return x_ref, x_em, x_gsf


def _strong_errors_from_paths(x_ref, x_em, x_gsf, block_size):
    """Compute strong local/global absolute errors for EM and GSF."""
    em_local = jnp.abs(x_em[1] - x_ref[block_size])
    gsf_local = jnp.abs(x_gsf[1] - x_ref[block_size])
    em_global = jnp.abs(x_em[-1] - x_ref[-1])
    gsf_global = jnp.abs(x_gsf[-1] - x_ref[-1])
    return em_local, em_global, gsf_local, gsf_global


def _mixture_errors_from_paths(x_ref, x_mixture_paths, block_size):
    """
    Return per-seed ingredients for weak-mean error aggregation.
    We do NOT compute |path - path| here because MGSF paths are independent from x_ref.
    """
    ref_local = x_ref[block_size]
    ref_global = x_ref[-1]
    mgsf_local_mean = jnp.mean(x_mixture_paths[:, 1])
    mgsf_global_mean = jnp.mean(x_mixture_paths[:, -1])
    return ref_local, ref_global, mgsf_local_mean, mgsf_global_mean

# Coupled MGSF still uses the low-level API;
# sde_solver.solve_mgsf does not support coeffs_list yet.
def mgsf_path_coupled_with_coeffs(delta, cfg, coeffs_list, sampling_key=None):
    """Simulate one Mixture-GSF trajectory using supplied interval coeffs."""
    sde, prior, solver_cfg = _build_mixture_solver_inputs(delta, cfg, sampling_key)
    _, traj = solve_sde_pathwise_mixture_with_coeffs(
        sde=sde,
        prior=prior,
        config=solver_cfg,
        coeffs_list=coeffs_list,
    )
    return traj

def one_seed_errors(root_key, delta, cfg):
    """Compute EM/GSF/MGSF strong local/global errors for one seed and step."""
    disc = prepare_coupled_discretization(root_key, delta, cfg.time.t_final)
    x_ref, x_em, x_gsf = _simulate_coupled_paths(root_key, delta, cfg, disc)

    em_local, em_global, gsf_local, gsf_global = strong_errors_from_paths(
        x_ref, x_em, x_gsf, disc.block_size
    )

    key_mgsf = jax.random.fold_in(root_key, 7)
    x_mgsf = mgsf_path_coupled_with_coeffs(
        delta, cfg, disc.coeffs_list, sampling_key=key_mgsf)
    mgsf_local = jnp.abs(x_mgsf[1] - x_ref[disc.block_size])
    mgsf_global = jnp.abs(x_mgsf[-1] - x_ref[-1])

    return em_local, em_global, gsf_local, gsf_global, mgsf_local, mgsf_global

def estimate_errors_for_delta(base_key, delta, cfg, progress_bar=None):
    """Estimate EM/GSF strong errors and MGSF errors for one delta."""
    keys = jax.random.split(base_key, cfg.mc.num_sample_paths)
    vals = []
    for i in range(cfg.mc.num_sample_paths):
        vals.append(one_seed_errors(keys[i], delta, cfg))
        if progress_bar is not None:
            progress_bar.update(1)
    arr = np.asarray(vals, dtype=float)

    em_local = float(np.mean(arr[:, 0]))
    em_global = float(np.mean(arr[:, 1]))
    gsf_local = float(np.mean(arr[:, 2]))
    gsf_global = float(np.mean(arr[:, 3]))
    mgsf_local = float(np.mean(arr[:, 4]))
    mgsf_global = float(np.mean(arr[:, 5]))

    return em_local, em_global, gsf_local, gsf_global, mgsf_local, mgsf_global

def _reconstruct_parabolic_path(times_fine, coeffs_list, eval_fn, delta):
    """Evaluate piecewise-parabolic approximation beta on the fine grid."""
    n_coarse = len(coeffs_list)
    n_fine = len(times_fine) - 1
    block_size = n_fine // n_coarse
    beta_vals = [0.0]
    beta_offset = 0.0
    for k in range(n_coarse):
        coeffs = coeffs_list[k]
        for j in range(1, block_size + 1):
            tau = j * (delta / block_size)
            local = float(eval_fn(tau, delta, *coeffs))
            beta_vals.append(beta_offset + local)
        beta_offset = beta_vals[-1]
    return np.asarray(beta_vals, dtype=float)


def brownian_approx_error(w_fine, beta_fine):
    """Simple diagnostics between sampled Brownian path and beta approximation."""
    diff = np.asarray(beta_fine) - np.asarray(w_fine)
    return {
        "sup_norm": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }

def run_experiment(cfg):
    """Run Monte Carlo convergence and generate one representative path triplet."""
    run_seed = resolve_mc_run_seed(cfg.mc.seed)
    print("Using seed =", run_seed)

    key_paths, key_mc = jax.random.split(jax.random.PRNGKey(run_seed), 2)

    _print_solver_settings(cfg)

    path_results = _compute_path_results(key_paths, cfg)
    mc_results = _compute_mc_error_results(key_mc, cfg)
    return {**path_results, **mc_results}


def _print_solver_settings(cfg):
    """Print EKF-related settings used by GSF (facade) and mixture (direct coupled path)."""
    gsf_run = _build_gsf_run_config(cfg, coeffs_list=None)
    print("EKF settings:")
    print("  GSF posterior mode (filter_config): " + str(gsf_run.filter_config.ekf_mode))
    print(
        "  MGSF t_k init (mixture config; coupled path): "
        + str(cfg.mixture.use_ekf1_tk_initialization)
    )
    print(
        "  MGSF posterior mode (mixture config; coupled path): "
        + str(cfg.mixture.posterior_ekf_mode)
    )
    print(
        "  Note: prob_sde.solvers.sde_solver.solve_mgsf fixes posterior_ekf_mode to "
        "\"ekf1\" and uses prior measurement_noise=1e-6 unless you extend MGSFRunConfig."
    )


def _compute_path_results(key_paths, cfg):
    """Generate representative EM/GSF/Mixture-GSF paths and Brownian diagnostics."""
    delta_path = (
        float(cfg.time.delta_for_path)
        if cfg.time.delta_for_path is not None
        else float(cfg.time.deltas[0])
    )
    parabolic_data = brownian_and_parabolic_coeffs(
        jax.random.fold_in(key_paths, 0),
        cfg.time.t_final,
        delta_path * delta_path,
        delta_path,
    )
    times = parabolic_data["times"]
    w = parabolic_data["w"]
    dw_coarse = parabolic_data["dw_coarse"]
    coeffs_list = parabolic_data["coeffs_list"]
    eval_fn = parabolic_data["eval_parabolic"]

    beta_path = _reconstruct_parabolic_path(times, coeffs_list, eval_fn, delta_path)
    path_em = euler_maruyama_from_increments(
        drift, diffusion, dw_coarse, delta_path, cfg.x0)

    key_gsf = jax.random.fold_in(key_paths, 1)
    num_steps_path = len(coeffs_list)
    key_mgsf_path = jax.random.fold_in(key_paths, 7)
    path_gsf = gsf_path_facade_coupled(
        key_gsf,
        cfg,
        delta_path,
        num_steps_path,
        coeffs_list
    )

    path_mgsf = mgsf_path_coupled_with_coeffs(
        delta_path, cfg, coeffs_list, sampling_key=key_mgsf_path)

    return {
        "ts_path": np.asarray(jnp.linspace(0.0, cfg.time.t_final, len(dw_coarse) + 1)),
        "path_em": np.asarray(path_em),
        "path_gsf": np.asarray(path_gsf),
        "path_mgsf": np.asarray(path_mgsf),
        "times_bm": np.asarray(times),
        "w_fine": np.asarray(w),
        "beta_approx": np.asarray(beta_path),
        "bm_diag": brownian_approx_error(w, beta_path),
    }

def _compute_mc_error_results(key_mc, cfg):
    """Estimate strong local/global error curves across configured step sizes."""
    series = {
        "em_local": [],
        "em_global": [],
        "gsf_local": [],
        "gsf_global": [],
        "mgsf_local": [],
        "mgsf_global": [],
    }

    total_mc = len(cfg.time.deltas) * cfg.mc.num_sample_paths
    with tqdm(total=total_mc, desc="MC", dynamic_ncols=True) as mc_bar:
        for idx, delta in enumerate(cfg.time.deltas):
            key_i = jax.random.fold_in(key_mc, idx)
            estimates = estimate_errors_for_delta(
                key_i, delta, cfg, progress_bar=mc_bar)
            for name, value in zip(series, estimates):
                series[name].append(value)
            mc_bar.set_postfix(
                delta=float(delta),
                em_g=round(estimates[1], 5),
                gsf_g=round(estimates[3], 5),
                mgsf_g=round(estimates[5], 5),
            )

    mc_bar.close()
    return {
        "deltas": np.asarray(cfg.time.deltas),
        **{name: np.asarray(values) for name, values in series.items()},
    }

def plot_results(results):
    """Create path comparison and strong-error log-log plots."""
    if plt is None:
        print("matplotlib is not installed. Install matplotlib to plot figures.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    _plot_path_panel(axes[0], results)
    _plot_error_panel(axes[1], results)

    fig.tight_layout()
    out = "benes_mgsf_gsf_em.png"
    fig.savefig(out, dpi=150)
    plt.show()
    plt.close(fig)
    print("Saved " + out)


def _plot_path_panel(ax, results):
    """Plot representative EM, GSF, and Mixture-GSF sample paths."""
    ax.plot(results["ts_path"], results["path_em"], label="EM path", linewidth=1.6)
    ax.plot(
        results["ts_path"],
        results["path_gsf"],
        label="GSF path",
        linewidth=1.6,
        linestyle="--",
    )
    ax.plot(
        results["ts_path"],
        results["path_mgsf"],
        label="Mixture-GSF path",
        linewidth=1.6,
        linestyle="-.",
    )
    ax.set_xlabel("t")
    ax.set_ylabel("X(t)")
    ax.set_title("Benes SDE: EM vs GSF vs Mixture-GSF")
    ax.grid(True, alpha=0.3)
    ax.legend()


def _plot_error_panel(ax, results):
    """Plot strong errors and fitted power laws on log-log axes."""
    deltas = np.asarray(results["deltas"])
    plot_error_data_series(
        ax, deltas, results, EM_GSF_MGSF_ERROR_SERIES_SPECS)

    if deltas.shape[0] >= 2:
        fits = fit_power_laws_for_error_series(
            deltas, results, EM_GSF_MGSF_ERROR_SERIES_SPECS)
        plot_fitted_error_series(
            ax, deltas, fits, EM_GSF_MGSF_ERROR_SERIES_SPECS)
        ax.text(
            0.02,
            0.02,
            format_power_law_text(fits, EM_GSF_MGSF_ERROR_SERIES_SPECS),
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
    ax.set_ylabel("MC mean |error| vs fine EM")
    ax.set_title("Strong local/global errors vs delta")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)


def print_error_comparison_table_mixture(results):
    """Print EM-vs-GSF-vs-MGSF comparison table for local/global errors.

    Parameters
    ----------
    results : dict[str, np.ndarray]
        Dictionary returned by `run_experiment`, containing:
        `deltas`, `em_local`, `em_global`, `gsf_local`, `gsf_global`,
        `mgsf_local`, `mgsf_global`.
    """
    deltas = np.asarray(results["deltas"], dtype=float)
    em_local = np.asarray(results["em_local"], dtype=float)
    gsf_local = np.asarray(results["gsf_local"], dtype=float)
    mgsf_local = np.asarray(results["mgsf_local"], dtype=float)
    em_global = np.asarray(results["em_global"], dtype=float)
    gsf_global = np.asarray(results["gsf_global"], dtype=float)
    mgsf_global = np.asarray(results["mgsf_global"], dtype=float)

    headers = (
        "delta",
        "EM local",
        "GSF local",
        "MGSF local",
        "Best local",
        "EM global",
        "GSF global",
        "MGSF global",
        "Best global",
    )
    widths = (10, 12, 12, 12, 10, 12, 12, 13, 11)
    separator = "-+-".join("-" * width for width in widths)

    print("\nError comparison: EM vs GSF vs MGSF (MC means)")
    print(_error_comparison_table_row(headers, widths))
    print(separator)

    for i in range(deltas.shape[0]):
        local_best = _error_method_winners(
            em_local[i], gsf_local[i], mgsf_local[i]
        )
        global_best = _error_method_winners(
            em_global[i], gsf_global[i], mgsf_global[i]
        )
        table_row = (
            f"{deltas[i]:.6f}",
            _format_error_comparison_float(em_local[i]),
            _format_error_comparison_float(gsf_local[i]),
            _format_error_comparison_float(mgsf_local[i]),
            local_best,
            _format_error_comparison_float(em_global[i]),
            _format_error_comparison_float(gsf_global[i]),
            _format_error_comparison_float(mgsf_global[i]),
            global_best,
        )
        print(_error_comparison_table_row(table_row, widths,))

def _format_error_comparison_float(value):
    """Format a scalar error for the EM/GSF/MGSF comparison table."""
    return f"{value:.3e}"


def _error_method_winners(em_val, gsf_val, mgsf_val):
    """Return slash-joined method names tied for the smallest error."""
    min_val = min(em_val, gsf_val, mgsf_val)
    winners = []
    if em_val == min_val:
        winners.append("EM")
    if gsf_val == min_val:
        winners.append("GSF")
    if mgsf_val == min_val:
        winners.append("MGSF")
    return "/".join(winners)


def _error_comparison_table_row(values, widths):
    """Right-justify cell strings to ``widths`` and join with ``' | '``."""
    cells = [str(value).rjust(width) for value, width in zip(values, widths)]
    return " | ".join(cells)


def main():
    """Run Benes SDE benchmark and plots for EM, GSF, and Mixture-GSF."""
    cfg = ExperimentConfig()
    results = run_experiment(cfg)
    print_error_comparison_table_mixture(results)
    plot_results(results)

if __name__ == "__main__":
    main()
