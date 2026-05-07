"""Unified facade for solving stochastic differential equations (SDEs).

This module provides a single entry point (`solve_sde`) that dispatches to one
of three SDE integration methods implemented in this package:

- Gaussian SDE Filter (GSF, Algorithm 2)
- Mixture Gaussian SDE Filter (MGSF, Algorithm 3)
- Marginalised Gaussian SDE Filter (Algorithm 4)

The goal is to standardise:
- how users specify the integration grid (via `TimeGridConfig`),
- how method-specific options are configured (via `GSFRunConfig`, `MGSFRunConfig`,
  `MarginalisedRunConfig`),
- and how results are returned (via `SDESolverResult`).

Important limitations
---------------------
- This facade currently assumes a uniform grid starting at `t0 = 0.0`.
  Some underlying implementations initialize and evaluate interval dynamics
  relative to 0.0. If you pass a nonzero `t0`, `SDESolverResult.ts` will be
  shifted but the underlying dynamics may not be. Prefer enforcing `t0 == 0.0`
  until all algorithms support true nonzero start times end-to-end.
- The marginalised solver implementation (`solve_sde_marginalised`) in this
  codebase is currently scalar-state only and validates that drift/diffusion
  outputs are scalar-compatible.

Return semantics
----------------
`solve_sde` always returns an `SDESolverResult` with a time grid `ts`. The
meaning of `trajectory` and auxiliary fields depends on the method:

- method="gsf"
  - `trajectory`: a single discrete trajectory on the grid
  - `means`, `covs`: per-grid latent Gaussian posterior moments when
    `GSFRunConfig.return_uncertainty=True`

- method="mgsf"
  - `trajectory`: a single discrete trajectory on the grid
  - `means`, `covs`: per-grid latent Gaussian posterior moments when
    `MGSFRunConfig.return_uncertainty=True`

- method="marginalised"
  - Draws `MarginalisedRunConfig.num_samples` independent Algorithm-4 trajectories
    using the batched marginalised path API.
  - `mean_trajectory`, `var_trajectory`: pointwise Monte Carlo mean/variance
  - `trajectory`: currently set to `mean_trajectory` for convenience

Coupled-noise / coefficient-driven runs
---------------------------------------
For fair strong-error comparisons against a reference path, some experiments
precompute Brownian-approximation coefficients per interval (e.g. using
`brownian_and_parabolic_coeffs`). The GSF wrapper supports this via
`GSFRunConfig.coeffs_list`, which forces `solve_gsf` to advance using the
caller-supplied coefficients (one tuple per time step).

Exports
-------
TimeGridConfig
    Uniform-grid settings shared by all methods.
GSFRunConfig
    Options for Gaussian SDE Filter (Algorithm 2).
MGSFRunConfig
    Options for pathwise mixture Gaussian SDE Filter (Algorithm 3).
MarginalisedRunConfig
    Options for marginalised Monte Carlo aggregation.
SDESolverConfig
    Method selector plus per-method configuration bundles.
SDESolverResult
    Standardised return container for trajectories and uncertainty.
solve_sde
    Dispatch to one of the three solver methods by `SDESolverConfig.method`.
solve_gsf
    Run the Gaussian SDE Filter on a uniform grid.
solve_mgsf
    Run the pathwise mixture Gaussian SDE Filter on a uniform grid.
solve_marginalised
    Run the marginalised solver with Monte Carlo aggregation on a uniform grid.
"""

from dataclasses import dataclass, field
from typing import Literal, Sequence

import jax
import jax.numpy as jnp

from prob_sde.filtering.sde.gaussian_sde_filter import (
    GaussianSDEFilter,
    GaussianSDEFilterConfig,
)
from prob_sde.filtering.sde.marginalised import (
    MarginalisedConfig,
    solve_sde_marginalised_batch
)
from prob_sde.filtering.sde.mixture_sde_filter import (
    EKFConfig,
    MixtureSDEFilterConfig,
    solve_sde_pathwise_mixture,
)
from prob_sde.core.prior_models import IWP2Prior
from prob_sde.core.sde import SDESpec


SolverMethod = Literal["gsf", "mgsf", "marginalised"]


@dataclass(frozen=True)
class TimeGridConfig:
    """Uniform integration grid configuration shared by all solver methods.

    Instance variables
    ------------------
    delta : float
        Positive fixed time-step size.
    num_steps : int
        Number of integration steps. The grid has `num_steps + 1` points.
    t0 : float
        Start time of the grid. Current facade-level usage should keep this at 0.0
        unless all underlying methods are extended for nonzero start times.

    Public methods
    --------------
    validate()
        Validate grid hyperparameters and basic invariants.
    time_grid()
        Build the inclusive uniform grid `[t0, ..., t0 + num_steps * delta]`.
    """

    delta: float
    num_steps: int
    t0: float = 0.0

    def validate(self) -> None:
        """Validate grid parameters."""
        if self.delta <= 0.0:
            raise ValueError("delta must be positive.")
        if self.num_steps < 0:
            raise ValueError("num_steps must be non-negative.")
        if self.t0 != 0.0:
            raise ValueError("Nonzero t0 is not supported yet; use t0 = 0.0")

    def time_grid(self) -> jnp.ndarray:
        """Return the inclusive grid [t0, ..., t0 + num_steps*delta]."""
        t1 = self.t0 + self.num_steps * self.delta
        return jnp.linspace(self.t0, t1, self.num_steps + 1)


@dataclass(frozen=True)
class GSFRunConfig:
    """Configuration for Gaussian SDE Filter (Algorithm 2) runs.

    Instance variables
    ------------------
    prior_scale : float
        Diffusion scaling passed to the IWP(2) prior used by the ODE-filter update.
    filter_config : GaussianSDEFilterConfig
        Algorithm-2 filter options (measurement model, EKF mode, sampling policy,
        and initialization covariance scale).
    coeffs_list : Sequence[tuple] | None
        Optional caller-provided Brownian-approximation coefficients, one tuple per
        interval. When provided, length must equal the solver grid `num_steps`.
        Useful for coupled-noise benchmarks and strong-error comparisons.
    return_uncertainty : bool
        If True, include latent posterior means/covariances in `SDESolverResult`.

    Public methods
    --------------
    validate(num_steps)
        Validate coefficient-list length against the integration horizon.
    """

    prior_scale: float = 1.0
    filter_config: GaussianSDEFilterConfig = field(
        default_factory=GaussianSDEFilterConfig
    )
    coeffs_list: Sequence[tuple] | None = None
    return_uncertainty: bool = False

    def validate(self, num_steps: int) -> None:
        """Validate GSF-specific options."""
        if self.coeffs_list is not None and len(self.coeffs_list) != num_steps:
            raise ValueError("coeffs_list length must match num_steps.")


@dataclass(frozen=True)
class MGSFRunConfig:
    """Configuration for pathwise Mixture-GSF (Algorithm 3) runs.

    Instance variables
    ------------------
    prior_scale : float
        Diffusion scaling passed to the IWP(2) prior used in posterior updates.
    return_uncertainty : bool
        If True, include latent posterior means/covariances in `SDESolverResult`.
    use_ekf1_tk_initialization : bool
        If True, use EKF1-style t_k initialization (paper Eq. 14); otherwise use
        EKF0-style initialization (paper Eq. 15).

    Public methods
    --------------
    This dataclass currently provides configuration fields only.
    """

    prior_scale: float = 1.0
    return_uncertainty: bool = False
    use_ekf1_tk_initialization: bool = True


@dataclass(frozen=True)
class MarginalisedRunConfig:
    """Configuration for marginalised solver aggregation (Algorithm 4).

    Instance variables
    ------------------
    prior_scale : float
        Prior diffusion scaling passed to `MarginalisedConfig.prior_diffusion`.
    num_samples : int
        Number of independent marginalised trajectories to draw and aggregate into
        pointwise Monte Carlo statistics.

    Public methods
    --------------
    validate()
        Validate Monte Carlo sampling configuration.
    """

    prior_scale: float = 1.0
    num_samples: int = 100

    def validate(self) -> None:
        """Validate marginalised-method options."""
        if self.num_samples < 1:
            raise ValueError("num_samples must be at least 1.")


@dataclass(frozen=True)
class SDESolverConfig:
    """Top-level unified solver configuration with method dispatch.

    Instance variables
    ------------------
    method : Literal["gsf", "mgsf", "marginalised"]
        Solver backend selected by the facade dispatcher.
    grid : TimeGridConfig
        Shared integration-grid settings used by all methods.
    gsf : GSFRunConfig
        Method-specific configuration used when `method == "gsf"`.
    mgsf : MGSFRunConfig
        Method-specific configuration used when `method == "mgsf"`.
    marginalised : MarginalisedRunConfig
        Method-specific configuration used when `method == "marginalised"`.

    Public methods
    --------------
    This dataclass currently provides configuration fields only.
    """

    method: SolverMethod
    grid: TimeGridConfig
    gsf: GSFRunConfig = field(default_factory=GSFRunConfig)
    mgsf: MGSFRunConfig = field(default_factory=MGSFRunConfig)
    marginalised: MarginalisedRunConfig = field(default_factory=MarginalisedRunConfig)


@dataclass(frozen=True)
class SDESolverResult:
    """Standardized return container for all facade solver methods.

    Instance variables
    ------------------
    ts : jnp.ndarray
        Integration grid of shape `(num_steps + 1,)`.
    trajectory : jnp.ndarray | None
        Primary trajectory output. For GSF/MGSF this is one pathwise trajectory.
        For the marginalised wrapper this is currently the pointwise Monte Carlo
        mean trajectory.
    means : jnp.ndarray | None
        Latent posterior means per grid point when uncertainty is requested for
        GSF/MGSF runs.
    covs : jnp.ndarray | None
        Latent posterior covariances per grid point when uncertainty is requested
        for GSF/MGSF runs.
    mean_trajectory : jnp.ndarray | None
        Pointwise Monte Carlo mean trajectory produced by the marginalised wrapper.
    var_trajectory : jnp.ndarray | None
        Pointwise Monte Carlo variance trajectory produced by the marginalised
        wrapper.

    Public methods
    --------------
    This dataclass currently acts as a structured immutable container.
    """

    ts: jnp.ndarray
    trajectory: jnp.ndarray | None = None
    means: jnp.ndarray | None = None
    covs: jnp.ndarray | None = None
    mean_trajectory: jnp.ndarray | None = None
    var_trajectory: jnp.ndarray | None = None


def solve_sde(key: jax.Array, sde: SDESpec, cfg: SDESolverConfig) -> SDESolverResult:
    """Solve an SDE by dispatching to the configured backend method.

    Parameters
    ----------
    key : jax.Array
        Root PRNG key used by the selected solver backend.
    sde : SDESpec
        SDE specification containing drift, diffusion, initial state, and (when
        relevant) Brownian-approximation factory.
    cfg : SDESolverConfig
        Top-level facade configuration including method selection, grid settings,
        and method-specific options.

    Returns
    -------
    SDESolverResult
        Standardized result container with time grid and method-dependent outputs.

    Raises
    ------
    ValueError
        If `cfg.method` is not one of `"gsf"`, `"mgsf"`, or `"marginalised"`.

    Notes
    -----
    Dispatch mapping:
    - `"gsf"` -> `solve_gsf`
    - `"mgsf"` -> `solve_mgsf`
    - `"marginalised"` -> `solve_marginalised`
    """
    if cfg.method == "gsf":
        return solve_gsf(key, sde, cfg.grid, cfg.gsf)
    if cfg.method == "mgsf":
        return solve_mgsf(key, sde, cfg.grid, cfg.mgsf)
    if cfg.method == "marginalised":
        return solve_marginalised(key, sde, cfg.grid, cfg.marginalised)
    raise ValueError("Unknown method: " + str(cfg.method))


def solve_gsf(
    key: jax.Array,
    sde: SDESpec,
    grid: TimeGridConfig,
    run_cfg: GSFRunConfig | None = None,
) -> SDESolverResult:
    """Run the Gaussian SDE Filter (Algorithm 2) on a uniform grid.

    This wrapper builds an IWP(2) prior and a `GaussianSDEFilter` configured for
    parabolic Brownian approximation, then rolls out one trajectory over
    `grid.num_steps` steps of size `grid.delta`.

    If `run_cfg.coeffs_list` is provided, the rollout uses caller-supplied
    interval coefficients (`step_with_coeffs`) for coupled-noise experiments.
    Otherwise, coefficients are sampled internally each step (`step`).

    Parameters
    ----------
    key : jax.Array
        Root PRNG key split into per-step keys for filter propagation.
    sde : SDESpec
        SDE model (drift, diffusion, initial condition) consumed by the filter.
    grid : TimeGridConfig
        Uniform integration-grid definition.
    run_cfg : GSFRunConfig | None, optional
        Algorithm-2 run options. If `None`, defaults to `GSFRunConfig()`.

    Returns
    -------
    SDESolverResult
        Result with:
        - `ts`: integration grid,
        - `trajectory`: one pathwise trajectory,
        - `means`, `covs`: included when `run_cfg.return_uncertainty=True`.

    Raises
    ------
    ValueError
        If grid parameters are invalid or `coeffs_list` length does not match
        `grid.num_steps`.
    """
    cfg = run_cfg if run_cfg is not None else GSFRunConfig()
    _validate_solver_inputs("gsf", grid, gsf_cfg=cfg)

    prior = IWP2Prior(
        diffusion=cfg.prior_scale,
        measurement_noise=cfg.filter_config.measurement_noise,
    )
    solver = GaussianSDEFilter.from_parabolic(
        prior=prior, sde=sde, config=cfg.filter_config
    )

    ts = grid.time_grid()
    state0 = solver.initialize(x0=jnp.asarray(sde.x0), t0=grid.t0)

    traj, means, covs = _rollout_gsf(
        key=key,
        solver=solver,
        state0=state0,
        grid=grid,
        cfg=cfg)

    if cfg.return_uncertainty:
        return SDESolverResult(
            ts=ts,
            trajectory=jnp.asarray(traj),
            means=jnp.asarray(means),
            covs=jnp.asarray(covs),
        )

    return SDESolverResult(
        ts=ts,
        trajectory=jnp.asarray(traj),
    )


def _rollout_gsf(
    key: jax.Array,
    solver: GaussianSDEFilter,
    state0,
    grid: TimeGridConfig,
    cfg: GSFRunConfig,
) -> tuple[list[jnp.ndarray], list[jnp.ndarray], list[jnp.ndarray]]:
    """Roll out Algorithm-2 state updates across the configured grid."""
    state = state0
    traj = [state.x]
    means = [state.mean]
    covs = [state.cov]

    step_keys = jax.random.split(key, grid.num_steps)
    for k in range(grid.num_steps):
        if cfg.coeffs_list is None:
            step_out = solver.step(step_keys[k], state, grid.delta)
        else:
            step_out = solver.step_with_coeffs(
                key=step_keys[k],
                state_k=state,
                delta=grid.delta,
                coeffs_k=cfg.coeffs_list[k],
            )

        state = step_out[0] if cfg.filter_config.return_beta_coeffs else step_out

        traj.append(state.x)
        means.append(state.mean)
        covs.append(state.cov)

    return traj, means, covs


def solve_mgsf(
    key: jax.Array,
    sde: SDESpec,
    grid: TimeGridConfig,
    run_cfg: MGSFRunConfig | None = None,
) -> SDESolverResult:
    """Run the pathwise Mixture Gaussian SDE Filter (Algorithm 3).

    This wrapper maps facade-level options to `MixtureSDEFilterConfig`, calls
    `solve_sde_pathwise_mixture`, and standardizes the output into
    `SDESolverResult`.

    Parameters
    ----------
    key : jax.Array
        Root PRNG key used by the pathwise mixture rollout.
    sde : SDESpec
        SDE specification consumed by Algorithm 3.
    grid : TimeGridConfig
        Uniform integration-grid definition.
    run_cfg : MGSFRunConfig | None, optional
        Algorithm-3 run options. If `None`, defaults to `MGSFRunConfig()`.

    Returns
    -------
    SDESolverResult
        Result with:
        - `ts`: integration grid,
        - `trajectory`: one pathwise mixture trajectory,
        - `means`, `covs`: included when `run_cfg.return_uncertainty=True`.

    Raises
    ------
    ValueError
        If grid parameters are invalid.

    Notes
    -----
    Current wrapper behavior fixes posterior EKF mode to `"ekf1"` in the mapped
    `EKFConfig`.
    """
    cfg = run_cfg if run_cfg is not None else MGSFRunConfig()
    _validate_solver_inputs("mgsf", grid)

    prior = IWP2Prior(diffusion=cfg.prior_scale, measurement_noise=1e-6)
    mgsf_cfg = MixtureSDEFilterConfig(
        delta=grid.delta,
        num_steps=grid.num_steps,
        return_uncertainty=cfg.return_uncertainty,
        ekf=EKFConfig(
            use_ekf1_tk_initialization=cfg.use_ekf1_tk_initialization,
            posterior_ekf_mode="ekf1",
        ),
        sample_posterior_position=True,
        variance_floor=1e-12,
    )

    raw = solve_sde_pathwise_mixture(
        key=key,
        sde=sde,
        prior=prior,
        config=mgsf_cfg,
    )

    ts = grid.time_grid()
    if cfg.return_uncertainty:
        _, traj, (means, covs) = raw
        return SDESolverResult(
            ts=ts,
            trajectory=traj,
            means=means,
            covs=covs,
        )

    _, traj = raw
    return SDESolverResult(
        ts=ts,
        trajectory=traj,
    )


def solve_marginalised(
    key: jax.Array,
    sde: SDESpec,
    grid: TimeGridConfig,
    run_cfg: MarginalisedRunConfig | None = None,
) -> SDESolverResult:
    """Run the marginalised solver (Algorithm 4) with Monte Carlo aggregation.

    This wrapper draws `run_cfg.num_samples` independent Algorithm-4 trajectories
    via `solve_sde_marginalised_batch`, then returns pointwise Monte Carlo
    mean/variance trajectories in a unified result object.

    Parameters
    ----------
    key : jax.Array
        Root PRNG key split into one key per Monte Carlo trajectory.
    sde : SDESpec
        SDE specification consumed by the marginalised solver.
    grid : TimeGridConfig
        Uniform integration-grid definition.
    run_cfg : MarginalisedRunConfig | None, optional
        Marginalised run options. If `None`, defaults to
        `MarginalisedRunConfig()`.

    Returns
    -------
    SDESolverResult
        Result with:
        - `ts`: integration grid,
        - `mean_trajectory`: pointwise Monte Carlo mean,
        - `var_trajectory`: pointwise Monte Carlo variance,
        - `trajectory`: currently set equal to `mean_trajectory` for convenience.

    Raises
    ------
    ValueError
        If grid parameters are invalid or `run_cfg.num_samples < 1`.

    Notes
    -----
    Independent trajectories are evaluated through the batched marginalised API
    and then aggregated with pointwise mean/variance.
    The underlying `solve_sde_marginalised` implementation in this codebase is
    currently scalar-state oriented.
    """
    cfg = run_cfg if run_cfg is not None else MarginalisedRunConfig()
    _validate_solver_inputs("marginalised", grid, marginalised_cfg=cfg)

    marg_cfg = MarginalisedConfig(
        delta=grid.delta,
        num_steps=grid.num_steps,
        sample_posterior_position=True,
        use_ekf1=True,
        variance_floor=1e-12,
        prior_diffusion=cfg.prior_scale,
        return_uncertainty=False,
    )

    sample_keys = jax.random.split(key, cfg.num_samples)
    _, all_samples = solve_sde_marginalised_batch(
        keys=sample_keys,
        sde=sde,
        config=marg_cfg,
    )

    mean_traj = jnp.mean(all_samples, axis=0)
    var_traj = jnp.var(all_samples, axis=0)

    return SDESolverResult(
        ts=grid.time_grid(),
        trajectory=mean_traj,
        mean_trajectory=mean_traj,
        var_trajectory=var_traj,
    )


def _validate_solver_inputs(
    method: SolverMethod,
    grid: TimeGridConfig,
    gsf_cfg: GSFRunConfig | None = None,
    marginalised_cfg: MarginalisedRunConfig | None = None,
) -> None:
    """Validate shared and method-specific solver inputs."""
    grid.validate()

    if method == "gsf":
        if gsf_cfg is None:
            raise ValueError("gsf_cfg is required when method='gsf'.")
        gsf_cfg.validate(grid.num_steps)
        return

    if method == "marginalised":
        if marginalised_cfg is None:
            raise ValueError("marginalised_cfg is required when method='marginalised'.")
        marginalised_cfg.validate()
        return

    if method == "mgsf":
        return

    raise ValueError("Unknown method: " + str(method))
