"""
Shared helpers for SDE benchmark examples (EM / GSF / extensions).

Exports
-------
CoupledDiscretization
    Container for coupled fine/coarse Brownian data and parabolic coefficients.
ErrorSeriesSpec
    Style and labelling metadata for one error series in log-log plots.
prepare_coupled_discretization
    Build coupled increments and parabolic coefficients for one coarse step size.
euler_maruyama_from_increments
    Scalar Euler-Maruyama path from supplied Brownian increments.
strong_errors_from_paths
    Strong local/global absolute errors vs a fine EM reference path.
fit_power_law
    Fit ``error ~ c * delta^p`` in log-log space.
fit_power_laws_for_error_series
    Fit power laws for every series described by ``ErrorSeriesSpec`` rows.
resolve_mc_run_seed
    Resolve a deterministic MC seed or draw a fresh one.
plot_error_data_series
    Plot measured error curves on ``loglog`` axes.
plot_fitted_error_series
    Overlay fitted power-law curves.
format_power_law_text
    Build multi-line annotation text from fitted ``(c, p)`` pairs.
EM_GSF_ERROR_SERIES_SPECS
    Default four-series style table (EM/GSF local/global).
EM_GSF_MGSF_ERROR_SERIES_SPECS
    Six-series table including Mixture-GSF local/global.
EM_GSF_MARG_WEAK_ERROR_SERIES_SPECS
    Six-series table for strong EM/GSF plus marginalised weak-g series.
POWER_LAW_FIT_HELP_TEXT
    Footnote text when fewer than two deltas are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from prob_sde import brownian_and_parabolic_coeffs


POWER_LAW_FIT_HELP_TEXT = (
    "Add at least two values in results['deltas']\nfor log-log power-law fits."
)


@dataclass(frozen=True)
class CoupledDiscretization:
    """
    Coupled Brownian increments and parabolic coefficients for one coarse grid.

    Public attributes
    -----------------
    delta_ref : float
        Fine EM step size (here ``delta_coarse ** 2``) for the reference path.
    num_steps : int
        Number of coarse steps on ``[0, t_final]``.
    block_size : int
        Number of fine steps per coarse step (fine index after one coarse step).
    dw_ref : jax.Array
        Fine Brownian increments for the reference EM path.
    dw_coarse : jax.Array
        Coarse Brownian increments for EM (and coupled parabolic construction).
    coeffs_list : list[tuple]
        Per-step parabolic coefficients fed to GSF-style solvers.
    """

    delta_ref: float
    num_steps: int
    block_size: int
    dw_ref: jax.Array
    dw_coarse: jax.Array
    coeffs_list: list[tuple]


class ErrorSeriesSpec(NamedTuple):
    """
    Metadata for plotting and annotating one stored error series.

    Public fields
    -------------
    key : str
        Key into the ``results`` mapping (for example ``\"em_local\"``).
    plot_label : str
        Legend label for measured points.
    power_law_label : str
        Short label used in the fitted-rate text box.
    marker_style : str
        Matplotlib line/marker style for measured data.
    color : str
        Matplotlib color for the fitted overlay line.
    fit_style : str
        Matplotlib line style for the fitted overlay line.
    """

    key: str
    plot_label: str
    power_law_label: str
    marker_style: str
    color: str
    fit_style: str

    def plot_pair(self) -> tuple[str, str, str]:
        """Return ``(key, plot_label, marker_style)`` for measured curves."""
        return self.key, self.plot_label, self.marker_style

    def fit_pair(self) -> tuple[str, str, str]:
        """Return ``(key, color, fit_style)`` for fitted overlays."""
        return self.key, self.color, self.fit_style


EM_GSF_ERROR_SERIES_SPECS: tuple[ErrorSeriesSpec, ...] = (
    ErrorSeriesSpec("em_local", "EM strong local", "EM local", "o-", "C0", "-"),
    ErrorSeriesSpec("em_global", "EM strong global", "EM global", "s-", "C1", "-"),
    ErrorSeriesSpec("gsf_local", "GSF strong local", "GSF local", "o--", "C2", "--"),
    ErrorSeriesSpec("gsf_global", "GSF strong global", "GSF global", "s--", "C3", "--"),
)

EM_GSF_MGSF_ERROR_SERIES_SPECS: tuple[ErrorSeriesSpec, ...] = (
    *EM_GSF_ERROR_SERIES_SPECS,
    ErrorSeriesSpec(
        "mgsf_local",
        "Mixture-GSF strong local",
        "Mixture-GSF local",
        "^-.",
        "C4",
        "-.",
    ),
    ErrorSeriesSpec(
        "mgsf_global",
        "Mixture-GSF strong global",
        "Mixture-GSF global",
        "v-.",
        "C5",
        "-.",
    ),
)

EM_GSF_MARG_WEAK_ERROR_SERIES_SPECS: tuple[ErrorSeriesSpec, ...] = (
    ErrorSeriesSpec("weak_em_g", "EM weak g", "EM weak g", "o-", "C0", "-"),
    ErrorSeriesSpec("weak_gsf_g", "GSF weak g", "GSF weak g", "s--", "C1", "--"),
    ErrorSeriesSpec("weak_marg_g", "Marg weak g", "Marg weak g", "^-.", "C2", "-."),
)


def prepare_coupled_discretization(
    root_key: jax.Array,
    delta: float,
    t_final: float,
) -> CoupledDiscretization:
    """
    Build coupled fine/coarse increments and parabolic coefficients.

    Parameters
    ----------
    root_key : jax.Array
        Base PRNG key; a sub-key is folded for the Brownian construction.
    delta : float
        Coarse step size on ``[0, t_final]``.
    t_final : float
        Final simulation time.

    Returns
    -------
    CoupledDiscretization
        Data shared by fine reference EM, coarse EM, and parabolic-coupled GSF.
    """
    delta_ref = delta * delta
    num_steps = int(round(t_final / delta))
    path_key = jax.random.fold_in(root_key, 0)
    parabolic_data = brownian_and_parabolic_coeffs(
        path_key,
        t_final,
        delta_ref,
        delta,
    )
    return CoupledDiscretization(
        delta_ref=delta_ref,
        num_steps=num_steps,
        block_size=int(parabolic_data["block_size"]),
        dw_ref=parabolic_data["dw_fine"],
        dw_coarse=parabolic_data["dw_coarse"],
        coeffs_list=parabolic_data["coeffs_list"],
    )


def euler_maruyama_from_increments(
    drift: Callable[..., jax.Array],
    diffusion: Callable[..., jax.Array],
    dw: jax.Array,
    delta: float,
    x0: float | jax.Array,
) -> jax.Array:
    """
    Scalar Euler-Maruyama path from supplied Brownian increments.

    Parameters
    ----------
    drift : callable
        Drift ``f(x, t)`` compatible with the EM update.
    diffusion : callable
        Diffusion ``g(x, t)`` compatible with the EM update.
    dw : jax.Array
        Increments of shape ``(num_steps,)``.
    delta : float
        Time step per increment.
    x0 : float | jax.Array
        Initial state at ``t = 0``.

    Returns
    -------
    jax.Array
        Path of shape ``(num_steps + 1,)`` including the initial value.

    Notes
    -----
    Uses ``t_k = k * delta`` when evaluating ``drift`` and ``diffusion``.
    """
    x = jnp.asarray(x0)
    path = [x]
    t_k = 0.0
    for k in range(int(dw.shape[0])):
        x = x + drift(x, t_k) * delta + diffusion(x, t_k) * dw[k]
        t_k = t_k + delta
        path.append(x)
    return jnp.asarray(path)


def strong_errors_from_paths(
    x_ref: jax.Array,
    x_em: jax.Array,
    x_gsf: jax.Array,
    block_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Strong local/global absolute errors for EM and GSF vs a fine EM reference.

    Parameters
    ----------
    x_ref : jax.Array
        Fine-step reference EM path.
    x_em : jax.Array
        Coarse-step EM path.
    x_gsf : jax.Array
        Coarse-step GSF path (same coarse grid as ``x_em``).
    block_size : int
        Fine index corresponding to one coarse step.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        ``(em_local, em_global, gsf_local, gsf_global)``.
    """
    em_local = jnp.abs(x_em[1] - x_ref[block_size])
    gsf_local = jnp.abs(x_gsf[1] - x_ref[block_size])
    em_global = jnp.abs(x_em[-1] - x_ref[-1])
    gsf_global = jnp.abs(x_gsf[-1] - x_ref[-1])
    return em_local, em_global, gsf_local, gsf_global


def fit_power_law(
        deltas: jnp.ndarray | np.ndarray,
        errors: jnp.ndarray | np.ndarray) -> tuple[float, float]:
    """
    Fit ``error ~ c * delta^p`` via linear regression in log-log space.

    Parameters
    ----------
    deltas : array-like
        Positive coarse step sizes.
    errors : array-like
        Positive error values aligned with ``deltas``.

    Returns
    -------
    tuple[float, float]
        ``(c, p)`` multiplicative constant and exponent estimate.

    Notes
    -----
    Fits ``log(errors)`` as an affine function of ``log(deltas)``.
    """
    x = jnp.log(jnp.asarray(deltas))
    y = jnp.log(jnp.asarray(errors))
    slope, log_intercept = jnp.polyfit(x, y, deg=1)
    return float(jnp.exp(log_intercept)), float(slope)


def fit_power_laws_for_error_series(
    deltas: jnp.ndarray | np.ndarray,
    results: dict,
    specs: Sequence[ErrorSeriesSpec],
) -> dict[str, tuple[float, float]]:
    """
    Fit a power law for each error series key listed in ``specs``.

    Parameters
    ----------
    deltas : array-like
        Positive coarse step sizes.
    results : dict
        Mapping containing each ``spec.key`` array of errors.
    specs : sequence of ErrorSeriesSpec
        Series to fit, in display order.

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping from series key to ``(c, p)``.
    """
    out: dict[str, tuple[float, float]] = {}
    for spec in specs:
        out[spec.key] = fit_power_law(deltas, results[spec.key])
    return out


def resolve_mc_run_seed(seed: Optional[int]) -> int:
    """
    Return a deterministic seed from config or draw a fresh integer seed.

    Parameters
    ----------
    seed : int | None
        If not ``None``, returned unchanged (after ``int`` coercion).

    Returns
    -------
    int
        Seed for ``jax.random.PRNGKey`` construction.
    """
    if seed is not None:
        return int(seed)
    return int(np.random.SeedSequence().generate_state(1)[0])


def plot_error_data_series(
        ax,
        deltas,
        results,
        specs: Sequence[ErrorSeriesSpec]) -> None:
    """
    Plot measured error curves on logarithmic axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    deltas : array-like
        Step sizes (positive).
    results : dict
        Must contain each ``spec.key`` array.
    specs : sequence of ErrorSeriesSpec
        Series definitions including plot styles.
    """
    for spec in specs:
        key, label, style = spec.plot_pair()
        ax.loglog(deltas, results[key], style, label=label)


def plot_fitted_error_series(
    ax,
    deltas,
    fits: dict[str, tuple[float, float]],
    specs: Sequence[ErrorSeriesSpec],
) -> None:
    """
    Overlay fitted power-law curves for each series.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    deltas : array-like
        Step sizes used to evaluate the fitted curves.
    fits : dict[str, tuple[float, float]]
        ``spec.key -> (c, p)`` from ``fit_power_laws_for_error_series``.
    specs : sequence of ErrorSeriesSpec
        Series definitions including colors and line styles.
    """
    deltas_arr = np.asarray(deltas)
    for spec in specs:
        key, color, style = spec.fit_pair()
        coeff, power = fits[key]
        ax.loglog(deltas_arr, coeff * deltas_arr**power, style, color=color, alpha=0.45)


def format_power_law_text(
    fits: dict[str, tuple[float, float]],
    specs: Sequence[ErrorSeriesSpec],
) -> str:
    """
    Build multi-line text summarizing fitted ``(c, p)`` pairs.

    Parameters
    ----------
    fits : dict[str, tuple[float, float]]
        Fitted coefficients keyed by series key.
    specs : sequence of ErrorSeriesSpec
        Order and ``power_law_label`` for each series.

    Returns
    -------
    str
        Newline-separated annotation text.
    """
    lines = []
    for spec in specs:
        coeff, power = fits[spec.key]
        lines.append(
            spec.power_law_label
            + " ~ "
            + str(round(coeff, 3))
            + " delta^"
            + str(round(power, 2))
        )
    return "\n".join(lines)
