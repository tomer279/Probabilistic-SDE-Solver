"""Algorithm 2 Gaussian SDE filter with interval-wise beta_k sampling.

Exports
-------
GaussianSDEFilterConfig
    Configuration for numerical settings and sampling policy.
GaussianSDEFilterState
    Immutable filter state at grid time t_k.
GaussianSDEFilter
    Algorithm 2 implementation that samples beta_k each step and applies
    one Gaussian ODE-filter update.
"""

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from prob_sde.brownian.brownian import piecewise_parabolic_brownian
from prob_sde.filtering.ode.ode_filter import (ode_filter_step, ODEFilterConfig)
from prob_sde.brownian.pathwise_rhs import make_interval_rhs
from .position_sampling import PositionSamplingConfig, select_posterior_position
from prob_sde.core.prior_models import IWP2Prior
from prob_sde.core.sde import SDESpec
from .state_helpers import (
    to_carry as _state_to_carry,
    position_marginal as _state_position_marginal,
)


VectorField = Callable[[jnp.ndarray, float], jnp.ndarray]


@dataclass(frozen=True)
class GaussianSDEFilterConfig:
    """Configuration for Algorithm 2.

    Instance variables
    ------------------
    measurement_noise : float
        Observation-noise variance for the residual update.
    sample_posterior_position : bool
        If True, sample X_{t_{k+1}} from N(m[0], P[0,0]).
        If False, use m[0] deterministically.
    variance_floor : float
        Minimum variance used for stable sampling.
    initial_cov_scale : float
        Initial covariance scale for Y_0(t_0).
    return_beta_coeffs : bool
        If True, step() returns sampled beta_k coefficients for diagnostics.
    ekf_mode : str
        EKF linearization mode for posterior update ("ekf0" or "ekf1").
    """

    measurement_noise: float = 1e-6
    sample_posterior_position: bool = True
    variance_floor: float = 1e-12
    initial_cov_scale: float = 1e-8
    return_beta_coeffs: bool = False
    ekf_mode: str = "ekf1"

    def __post_init__(self) -> None:
        """Validate configuration invariants."""
        if self.ekf_mode not in ("ekf0", "ekf1"):
            raise ValueError("ekf_mode must be 'ekf0' or 'ekf1'")


@dataclass(frozen=True)
class GaussianSDEFilterState:
    """Posterior Gaussian state at one integration grid point.

    Instance variables
    ------------------
    t : float
        Current grid time t_k.
    x : jnp.ndarray
        Discrete path value at t_k.
    mean : jnp.ndarray
        Posterior latent mean m_k.
    cov : jnp.ndarray
        Posterior latent covariance P_k.

    Public methods
    --------------
    to_carry()
        Return legacy tuple carry (mean, cov).
    position_marginal()
        Return first-component posterior mean and variance.
    """

    t: float
    x: jnp.ndarray
    mean: jnp.ndarray
    cov: jnp.ndarray

    def to_carry(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (mean, cov) for tuple-based carry compatibility."""
        return _state_to_carry(self.mean, self.cov)

    def position_marginal(
            self,
            variance_floor: float = 1e-12) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return first posterior component mean and variance."""
        return _state_position_marginal(self.mean, self.cov, variance_floor)


@dataclass(frozen=True)
class GaussianSDEFilter:
    """Algorithm 2 filter that samples beta_k and updates one interval.

    Public methods
    --------------
    initialize(prior, x0, model, bm_factory, t0=0.0)
        Build initial state from x0 and f_0(x0, t0).
    step(key, state_k, model, bm_factory, delta)
        Sample beta_k, build f_k, run EKF0 update, produce next state.
    step_from_carry(key, mean, cov, t_k, model, bm_factory, delta)
        Compatibility helper for tuple-based scan carry.
    """

    prior: IWP2Prior
    sde: SDESpec
    config: GaussianSDEFilterConfig
    get_coeffs: callable
    eval_fn: callable

    @classmethod
    def from_parabolic(cls, prior, sde: SDESpec, config):
        """Construct a filter configured with parabolic Brownian approximation.
        
        Parameters
        ----------
        prior : IWP2Prior
            IWP prior used by the Gaussian ODE filter update.
        sde : SDESpec
            SDE specification providing drift, diffusion, and initial value.
        config : GaussianSDEFilterConfig
            Numerical and sampling options for Algorithm 2.
        
        Returns
        -------
        GaussianSDEFilter
            Filter instance with parabolic Brownian coefficient sampler/evaluator.
        """
        get_coeffs, eval_fn = piecewise_parabolic_brownian()
        return cls(prior=prior, sde=sde, config=config,
                   get_coeffs=get_coeffs, eval_fn=eval_fn)

    def initialize(
        self,
        x0: jnp.ndarray,
        t0: float = 0.0,
    ) -> GaussianSDEFilterState:
        """Build the initial Algorithm-2 Gaussian state at time t0.
        
        The initialization sets:
        - x(t0) to the provided initial state x0,
        - the latent mean to [x0, drift(x0, t0)] for the IWP-2 state,
        - the latent covariance to a scaled identity matrix using
          config.initial_cov_scale.
        
        Parameters
        ----------
        x0 : jnp.ndarray
            Initial SDE state value at time t0.
        t0 : float, optional
            Initial time. Defaults to 0.0.
        
        Returns
        -------
        GaussianSDEFilterState
            Initial filter state containing time, path value, latent mean,
            and latent covariance.
        """
        self.prior.measurement_noise = self.config.measurement_noise

        x0_arr = jnp.asarray(x0)
        x0_dot = self.sde.drift(x0_arr, t0)

        mean0 = self.prior.initial_mean(x0_arr, derivatives=(x0_dot,))
        cov0 = self.prior.initial_covariance(scale=self.config.initial_cov_scale)

        return GaussianSDEFilterState(
            t=t0,
            x=x0_arr,
            mean=mean0,
            cov=cov0,
        )

    def step(self,
             key: jax.Array,
             state_k: GaussianSDEFilterState,
             delta: float):
        """Advance one Algorithm-2 step on [t_k, t_{k+1}].
        
        This method samples interval coefficients for beta_k, builds the random
        vector field f_k(x,t) = mu(x,t) + sigma(x,t) * d/dt beta_k(t), runs one
        EKF0 update, and returns the next filter state.
        
        Parameters
        ----------
        key : jax.Array
            PRNG key used for beta_k sampling and optional posterior sampling.
        state_k : GaussianSDEFilterState
            Current Gaussian state at time t_k.
        delta : float
            Step size t_{k+1} - t_k.
        
        Returns
        -------
        GaussianSDEFilterState or tuple[GaussianSDEFilterState, tuple]
            Next state; optionally includes sampled beta_k coefficients when
            return_beta_coeffs=True.
        """
        self.prior.measurement_noise = self.config.measurement_noise
        key_beta, key_sample = jax.random.split(key, 2)
        coeffs_k = self.get_coeffs(key_beta, delta)
        next_state = self._step_core(state_k, delta, coeffs_k, key_sample)

        if self.config.return_beta_coeffs:
            return next_state, coeffs_k
        return next_state

    def _step_core(
        self,
        state_k: GaussianSDEFilterState,
        delta: float,
        coeffs_k,
        key_sample: jax.Array,
    ) -> GaussianSDEFilterState:
        """Run one interval update using Algorithm-2 interval initialization."""
        self.prior.measurement_noise = self.config.measurement_noise

        f_k, mean_init, cov_init = self._interval_initial_moments(
            state_k, coeffs_k, delta
        )

        step_cfg= self._make_step_cfg()

        mean_k1, cov_k1 = ode_filter_step(
            mean_init,
            cov_init,
            f_k,
            (0.0, delta),
            step_cfg,
        )

        x_k1 = self._select_next_position(key_sample, mean_k1, cov_k1)
        return GaussianSDEFilterState(
            t=state_k.t + delta,
            x=x_k1,
            mean=mean_k1,
            cov=cov_k1,
        )

    def _interval_initial_moments(
        self,
        state_k: GaussianSDEFilterState,
        coeffs_k,
        delta: float,
    ) -> tuple[callable, jnp.ndarray, jnp.ndarray]:
        """Build interval-specific EKF initial moments from Algorithm 2."""
        f_k = make_interval_rhs(
            sde=self.sde,
            t_k=state_k.t,
            delta=delta,
            coeffs_k=coeffs_k,
            eval_fn=self.eval_fn)

        # Algorithm 2: anchor on the path value X_{t_k}
        x_k = jnp.asarray(state_k.x)

        # Algorithm 2: second latent component is f_k(X_{t_k}, t_k), i.e. tau=0 locally
        rhs_k = f_k(x_k, 0.0)
        mean_init = self.prior.initial_mean(x_k, derivatives=(rhs_k,))
        cov_init = self.prior.initial_covariance(scale=self.config.initial_cov_scale)
        return f_k, mean_init, cov_init

    def _make_step_cfg(self) -> ODEFilterConfig:
        """Build one-step ODE filter configuration from current solver config."""
        return ODEFilterConfig(
            prior=self.prior,
            ekf_mode=self.config.ekf_mode,
            measurement_noise=self.config.measurement_noise,
        )

    def step_with_coeffs(
        self,
        key: jax.Array,
        state_k: GaussianSDEFilterState,
        delta: float,
        coeffs_k):
        """Advance one step using supplied Brownian coefficients (coupling)."""
        next_state = self._step_core(state_k, delta, coeffs_k, key)
        if self.config.return_beta_coeffs:
            return next_state, coeffs_k
        return next_state

    def step_from_carry(
        self,
        key: jax.Array,
        carry: tuple[jnp.ndarray, jnp.ndarray],
        t_k: float,
        delta: float,
    ):
        """Compatibility wrapper for tuple-based scan carry.
        
        Parameters
        ----------
        
        key : jax.Array
            PRNG key for one integration step.
        carry : tuple[jnp.ndarray, jnp.ndarray]
            Legacy carry tuple (mean, covariance).
        t_k : float
            Current step start time.
        delta : float
            Step size.
        
        Returns
        -------
        tuple
            (mean_k1, cov_k1, x_k1), and beta_k coefficients as an extra element
            when return_beta_coeffs=True.
        """
        mean, cov = carry
        state_k = GaussianSDEFilterState(
            t=t_k,
            x=jnp.asarray(mean[0]),
            mean=mean,
            cov=cov,
        )
        result = self.step(key, state_k, delta)
        if self.config.return_beta_coeffs:
            state_k1, coeffs_k = result
            return state_k1.mean, state_k1.cov, state_k1.x, coeffs_k
        return result.mean, result.cov, result.x

    def _select_next_position(
        self,
        key: jax.Array,
        mean: jnp.ndarray,
        cov: jnp.ndarray,
    ) -> jnp.ndarray:
        """Select next trajectory position via shared posterior sampler.

        Builds a `PositionSamplingConfig` from `GaussianSDEFilterConfig` and
        delegates deterministic-or-sampled position extraction to
        `select_posterior_position`.
        """
        cfg = PositionSamplingConfig(
            sample_posterior_position=self.config.sample_posterior_position,
            variance_floor=self.config.variance_floor,
            require_key_when_sampling=False,
        )
        return select_posterior_position(mean=mean, cov=cov, cfg=cfg, key=key)
