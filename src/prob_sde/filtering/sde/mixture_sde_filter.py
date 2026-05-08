"""
Algorithm 3 (Gaussian Mixture SDE Filter): pathwise probabilistic SDE solver.

This module implements the outer loop of Algorithm 3 from Le Fay, Sarkka, and
Corenflos (2025): on each interval it samples Brownian approximation
coefficients, builds a local random ODE, and advances a Gaussian state with
a direct Algorithm-2 ODE filter step.

Exported objects
----------------
EKFConfig
    EKF-mode configuration for t_k initialization and posterior update.
MixtureSDEFilterConfig
    Dataclass containing step size, number of steps, and output options.
MixtureSDEFilterState
    Immutable state at one integration grid point.
MixtureSDEFilter
    Class-based Algorithm 3 implementation.
solve_sde_pathwise_mixture
    Functional wrapper around MixtureSDEFilter.solve().
solve_sde_pathwise_mixture_with_coeffs
    Functional wrapper for rollout with caller-supplied coefficients.

Algorithm mapping
-----------------
- Algorithm 1: consumed per step through bm_factory() -> (get_coeffs, eval_fn),
  which provides interval-wise Brownian approximation coefficients.
- Algorithm 2: applied directly via ode_filter_step with ODEFilterConfig.
- Algorithm 3: implemented by MixtureSDEFilter and ScanContext scan stepping.
"""

from dataclasses import dataclass, field
import jax
import jax.numpy as jnp

from prob_sde.mixture_utils.mixture_scan import (
    make_scan_context,
    make_scan_inputs,
    prepend_uncertainty
    )
from prob_sde.mixture_utils.mixture_tk_init import (
    carry_after_tk_initialization_from_rhs
    )
from prob_sde.filtering.ode.ode_filter import ODEFilterConfig, ode_filter_step
from prob_sde.brownian.pathwise_rhs import make_interval_rhs
from prob_sde.core.prior_models import IWP2Prior
from prob_sde.core.sde import SDESpec
from prob_sde.utils.utils import insert

from .position_sampling import PositionSamplingConfig, select_posterior_position
from .state_helpers import (
    to_carry as _state_to_carry,
    position_marginal as _state_position_marginal,
)


@dataclass(frozen=True)
class EKFConfig:
    """EKF choices for Algorithm 3.

    Instance variables
    ------------------
    use_ekf1_tk_initialization
        If True, use Eq. (14) at t_k; else Eq. (15).
    posterior_ekf_mode
        Posterior update mode on [t_k, t_{k+1}]: "ekf0" or "ekf1".
    """
    use_ekf1_tk_initialization: bool = True
    posterior_ekf_mode: str = "ekf1"

    def __post_init__(self) -> None:
        if self.posterior_ekf_mode not in ("ekf0", "ekf1"):
            raise ValueError("posterior_ekf_mode must be 'ekf0' or 'ekf1'")

@dataclass(frozen=True)
class MixtureSDEFilterConfig:
    """Store numerical options for pathwise SDE integration.
    
    Instance variables
    ------------------
    delta : float
        Fixed integration step size.
    num_steps : int
        Number of integration steps.
    return_uncertainty : bool
        If `True`, return per-step latent means/covariances in addition to
        the trajectory.
    ekf : EKFConfig
        EKF configuration:
        - `use_ekf1_tk_initialization` selects Eq. (14) (`True`) or Eq. (15) (`False`)
          for t_k initialization.
        - `posterior_ekf_mode` selects posterior update linearization mode
          (`"ekf0"` or `"ekf1"`).
    sample_posterior_position : bool
        If `True`, trajectory positions are sampled from the posterior
        position marginal `N(mean[0], cov[0,0])` at each step.
        If `False`, trajectory positions use the posterior mean `mean[0]`.
    variance_floor : float
        Minimum variance used when sampling posterior position to ensure
        numerical stability.
    sampling_key : jax.Array | None
        Root PRNG key used for sampling in coefficient-driven rollout
        (`solve_sde_pathwise_mixture_with_coeffs`). Required when
        `sample_posterior_position=True` in that path.
    """

    delta: float
    num_steps: int
    return_uncertainty: bool = False
    ekf: EKFConfig = field(default_factory=EKFConfig)
    sample_posterior_position: bool = True
    variance_floor: float = 1e-12
    sampling_key: jax.Array | None = None

# Backward-compatible alias (can be removed in a later cleanup pass).
SolverConfig = MixtureSDEFilterConfig

@dataclass(frozen=True)
class MixtureSDEFilterState:
    """Posterior Gaussian state at one integration grid point.

    Instance variables
    ------------------
    t : float
        Current grid time t_k.
    x : jnp.ndarray
        Discrete path value at t_k (position component).
    mean : jnp.ndarray
        Posterior latent mean at t_k.
    cov : jnp.ndarray
        Posterior latent covariance at t_k.

    Public methods
    --------------
    to_carry()
        Return legacy tuple carry (mean, cov).
    position_marginal(variance_floor=1e-12)
        Return first-component posterior mean and variance.
    """

    t: float
    x: jnp.ndarray
    mean: jnp.ndarray
    cov: jnp.ndarray

    def to_carry(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return tuple-form latent Gaussian carry.
        
        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            `(mean, cov)` representation of the current latent state,
            compatible with tuple-based scan/integrator code paths.
        """
        return _state_to_carry(self.mean, self.cov)

    def position_marginal(
        self, variance_floor: float = 1e-12
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return posterior position marginal statistics.

        Parameters
        ----------
        variance_floor : float, optional
            Minimum variance returned for numerical stability.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            `(mean_pos, var_pos)` where `mean_pos = mean[0]` and
            `var_pos = max(cov[0,0], variance_floor)`.
        """
        return _state_position_marginal(self.mean, self.cov, variance_floor)

@dataclass(frozen=True)
class MixtureSDEFilter:
    """Algorithm 3 pathwise SDE filter with class-based API.
    
    Instance variables
    ------------------
    sde : SDESpec
        SDE specification containing drift, diffusion, initial state, and
        Brownian approximation factory.
    prior : IWP2Prior
        IWP prior used by Algorithm-2 posterior updates.
    config : MixtureSDEFilterConfig
        Numerical options for stepping and outputs.
    
    Public methods
    --------------
    initialize(t0=0.0)
        Build initial Gaussian state.
    step(key, state_k, delta=None)
        Advance one interval and return the next state.
    step_with_coeffs(state_k, coeffs_k, delta=None, key=None)
        Advance one interval using caller-provided coefficients.
    solve(key)
        Run Algorithm 3 over the full configured time grid.
    """

    sde: SDESpec
    prior: IWP2Prior
    config: MixtureSDEFilterConfig

    def initialize(self, t0: float = 0.0) -> MixtureSDEFilterState:
        """Build the initial filter state at time t0."""
        x0 = jnp.asarray(self.sde.x0)
        x0_dot = self.sde.drift(x0, t0)
        mean0 = jnp.array([jnp.squeeze(x0), jnp.squeeze(x0_dot)])
        cov0 = jnp.eye(2) * 1e-8
        return MixtureSDEFilterState(t=t0, x=x0, mean=mean0, cov=cov0)

    def step(
        self,
        key: jax.Array,
        state_k: MixtureSDEFilterState,
        delta: float | None = None
    ) -> MixtureSDEFilterState:
        """Run Algorithm 3 over the configured time grid.

        Parameters
        ----------
        key : jax.Array
            Root PRNG key split per time step. One subkey is used for Brownian
            approximation coefficients and, when enabled, one for posterior
            position sampling.
        
        Returns
        -------
        tuple
            If `config.return_uncertainty` is `False`:
                `(ts, traj)`
            Else:
                `(ts, traj, (means, covs))`
            where `ts` is the integration grid, `traj` is the path of sampled
            or mean positions (depending on config), and `means/covs` are the
            latent Gaussian states per step.
        """
        step_delta = self.config.delta if delta is None else delta
        rhs, key_sample = self._prepare_interval(key, state_k, step_delta)

        carry_in = carry_after_tk_initialization_from_rhs(
            state_k.to_carry(),
            rhs,
            self.config.ekf.use_ekf1_tk_initialization,
        )

        mean_new, cov_new = self._posterior_update(carry_in, rhs, step_delta)
        x_new = self._select_next_position(key_sample, mean_new, cov_new)
        return MixtureSDEFilterState(
            t=state_k.t + step_delta,
            x=x_new,
            mean=mean_new,
            cov=cov_new,
        )

    def step_with_coeffs(
        self,
        state_k: MixtureSDEFilterState,
        coeffs_k,
        delta: float | None = None,
        key: jax.Array | None = None,
    ) -> MixtureSDEFilterState:
        """Advance one Algorithm-3 step using provided interval coefficients.

        Parameters
        ----------
        state_k : MixtureSDEFilterState
            Current filter state at time t_k.
        coeffs_k : tuple
            Brownian-approximation coefficients for the current interval.
        delta : float | None, optional
            Step size override. If `None`, uses `config.delta`.
        key : jax.Array | None, optional
            PRNG key for posterior position sampling. Required only when
            `config.sample_posterior_position=True`.
        
        Returns
        -------
        MixtureSDEFilterState
            Next state at time t_{k+1}.
        """
        step_delta = self.config.delta if delta is None else delta
        _, eval_fn = self.sde.bm_factory()
        rhs = _make_step_rhs(
            self.sde,
            step_delta,
            state_k.t,
            coeffs_k,
            eval_fn=eval_fn,
        )
        carry_in = carry_after_tk_initialization_from_rhs(
            state_k.to_carry(),
            rhs,
            self.config.ekf.use_ekf1_tk_initialization,
        )
        step_cfg = self._make_step_cfg()
        mean_new, cov_new = ode_filter_step(
            carry_in[0],
            carry_in[1],
            rhs,
            (0.0, step_delta),
            step_cfg
        )
        x_new = self._select_next_position(key, mean_new, cov_new)
        return MixtureSDEFilterState(
            t=state_k.t + step_delta,
            x=x_new,
            mean=mean_new,
            cov=cov_new,
        )

    def solve(self, key: jax.Array):
        """Run Algorithm 3 over the configured time grid.

        At each step this method samples interval Brownian-approximation
        coefficients, builds the local random IVP vector field, applies the
        t_k initialization rule (Eq. (14) or Eq. (15)), performs one Gaussian
        ODE-filter update, and emits the next trajectory position (posterior
        mean or sampled posterior position, according to config).

        Parameters
        ----------
        key : jax.Array
            Root PRNG key for the rollout. It is split per step and reused for
            Brownian-coefficient sampling and, when enabled, posterior-position
            sampling.

        Returns
        -------
        tuple
            If `self.config.return_uncertainty` is `False`, returns:
                `(ts, traj)`
            where:
                - `ts` is the integration grid of shape `(num_steps + 1,)`,
                - `traj` is the trajectory of emitted positions.

            If `self.config.return_uncertainty` is `True`, returns:
                `(ts, traj, (means, covs))`
            where:
                - `means` are latent posterior means per grid point,
                - `covs` are latent posterior covariances per grid point.
        """
        x0 = jnp.asarray(self.sde.x0)
        state0 = self.initialize(t0=0.0)

        ts, inps = make_scan_inputs(key, self.config)
        step_cfg = self._make_step_cfg()
        ctx = make_scan_context(
            self.sde,
            step_cfg,
            self.config
        )

        _, outputs = jax.lax.scan(ctx.step, state0.to_carry(), inps)
        traj = insert(outputs[0], 0, x0)

        if not self.config.return_uncertainty:
            return ts, traj

        means, covs = prepend_uncertainty(self.sde, x0, outputs[1], outputs[2])
        return ts, traj, (means, covs)

    def _select_next_position(
        self,
        key: jax.Array | None,
        mean: jnp.ndarray,
        cov: jnp.ndarray,
    ) -> jnp.ndarray:
        """Select next trajectory position via shared posterior sampler.

        Builds a `PositionSamplingConfig` from `MixtureSDEFilterConfig` and
        delegates position extraction to `select_posterior_position`.
        This path enforces that a PRNG key is provided when sampling is enabled.
        """
        cfg = PositionSamplingConfig(
            sample_posterior_position=self.config.sample_posterior_position,
            variance_floor=self.config.variance_floor,
            require_key_when_sampling=True,
        )
        return select_posterior_position(mean=mean, cov=cov, cfg=cfg, key=key)

    def _make_step_cfg(self) -> ODEFilterConfig:
        """Build one-step ODE filter config with enforced posterior EKF mode."""
        return ODEFilterConfig(
            prior=self.prior,
            ekf_mode=self.config.ekf.posterior_ekf_mode,
            measurement_noise=self.prior.measurement_noise,
        )

    def _prepare_interval(
            self,
            key: jax.Array,
            state_k: MixtureSDEFilterState,
            step_delta: float):
        """Sample coefficients, build rhs, and split sampling key."""
        get_coeffs, eval_fn = self.sde.bm_factory()
        bm_key, key_sample = jax.random.split(key, 2)
        coeffs_k = get_coeffs(bm_key, step_delta)
        rhs = _make_step_rhs(
            self.sde,
            step_delta,
            state_k.t,
            coeffs_k,
            eval_fn=eval_fn,
        )
        return rhs, key_sample

    def _posterior_update(
            self,
            carry_in,
            rhs,
            step_delta: float):
        """Run one posterior Gaussian update on [t_k, t_{k+1}]."""
        step_cfg = self._make_step_cfg()
        mean_new, cov_new = ode_filter_step(
            carry_in[0], carry_in[1], rhs, (0.0, step_delta), step_cfg
        )
        return mean_new, cov_new


def _make_step_rhs(
    sde: SDESpec,
    delta,
    t_k,
    coeffs_k,
    eval_fn=None,
):
    """Build RHS for one local random ODE step."""
    return make_interval_rhs(
        sde=sde,
        t_k=t_k,
        delta=delta,
        coeffs_k=coeffs_k,
        eval_fn=eval_fn,
    )

def solve_sde_pathwise_mixture(
    key,
    sde: SDESpec,
    prior: IWP2Prior,
    config: MixtureSDEFilterConfig,
):
    """Solve one pathwise Algorithm-3 trajectory from model/config objects.
    
    Parameters
    ----------
    key : jax.Array
        Root PRNG key for the rollout.
    sde : SDESpec
        Drift, diffusion, initial value, and Brownian factory specification.
    prior : IWP2Prior
        IWP prior used for Algorithm-2 posterior updates.
    config : MixtureSDEFilterConfig
        Solver configuration.
    
    Returns
    -------
    tuple
        Forwarded output of `MixtureSDEFilter.solve(key)`.
    """
    solver = MixtureSDEFilter(
        sde=sde,
        prior=prior,
        config=config,
    )
    return solver.solve(key)

def solve_sde_pathwise_mixture_with_coeffs(
    sde: SDESpec,
    prior: IWP2Prior,
    config: MixtureSDEFilterConfig,
    coeffs_list,
):
    """Solve Algorithm 3 using caller-provided per-step Brownian coefficients.
    
    This is the coupled-noise variant typically used for fair strong-error
    comparisons against reference paths.
    
    Parameters
    ----------
    sde : SDESpec
        Drift, diffusion, initial value, and Brownian factory specification.
    prior : IWP2Prior
        IWP prior used for Algorithm-2 posterior updates.
    config : MixtureSDEFilterConfig
        Solver configuration.
    coeffs_list : sequence
        Per-step Brownian approximation coefficients. Length must equal
        `config.num_steps`.
    
    Returns
    -------
    tuple
        If `config.return_uncertainty` is `False`:
            `(ts, traj)`
        Else:
            `(ts, traj, (means, covs))`.
    
    Raises
    ------
    ValueError
        If `len(coeffs_list) != config.num_steps`.
    """
    num_steps = len(coeffs_list)
    if num_steps != config.num_steps:
        raise ValueError("len(coeffs_list) must match config.num_steps.")

    solver = MixtureSDEFilter(
        sde=sde,
        prior=prior,
        config=config,
    )

    traj_arr, means_arr, covs_arr = _run_coeff_rollout(
        solver=solver,
        config=config,
        sde=sde,
        coeffs_list=coeffs_list,
    )

    ts = jnp.linspace(
        0.0, config.num_steps * config.delta, config.num_steps + 1)
    if not config.return_uncertainty:
        return ts, traj_arr

    return ts, traj_arr, (means_arr, covs_arr)

def _run_coeff_rollout(
    solver: MixtureSDEFilter,
    config: MixtureSDEFilterConfig,
    sde: SDESpec,
    coeffs_list,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Roll out Algorithm 3 over provided Brownian coefficients."""
    state = solver.initialize(t0=0.0)
    traj = [jnp.asarray(sde.x0)]
    means = [state.mean]
    covs = [state.cov]

    if config.sample_posterior_position:
        if config.sampling_key is None:
            raise ValueError(
                "sample_posterior_position=True requires config.sampling_key."
            )
        step_keys = jax.random.split(config.sampling_key, len(coeffs_list))
    else:
        step_keys = [None] * len(coeffs_list)

    for k, coeffs_k in enumerate(coeffs_list):
        state = solver.step_with_coeffs(
            state_k=state,
            coeffs_k=coeffs_k,
            delta=config.delta,
            key=step_keys[k],
        )
        traj.append(state.x)
        means.append(state.mean)
        covs.append(state.cov)

    return jnp.asarray(traj), jnp.asarray(means), jnp.asarray(covs)
