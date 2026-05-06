"""
Algorithm-3 scan helpers for pathwise mixture SDE filtering.

Exported objects
----------------
ScanContext
    Immutable dependencies and per-step logic for `jax.lax.scan`.
make_scan_inputs
    Build uniform time grid and per-step `(key_k, t_k)` scan inputs.
make_scan_context
    Construct `ScanContext` from SDE model, step config, and solver config.
prepend_uncertainty
    Prepend initial latent moments to scan-emitted uncertainty arrays.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .mixture_tk_init import carry_after_tk_initialization_from_rhs
from prob_sde.filtering.ode.ode_filter import ODEFilterConfig, ode_filter_step
from prob_sde.brownian.pathwise_rhs import make_interval_rhs
from prob_sde.filtering.sde.position_sampling import (
    PositionSamplingConfig,
    select_posterior_position,
)
from prob_sde.core.sde import SDESpec
from prob_sde.utils.utils import insert


@dataclass(frozen=True)
class ScanContext:
    """Bundle per-step dependencies for `jax.lax.scan` SDE integration.

    Instance variables
    ------------------
    sde : SDESpec
        SDE specification with drift, diffusion, and Brownian factory.
    get_coeffs : callable
        Function mapping `(key, delta)` to Brownian coefficients.
    eval_fn : callable
        Brownian basis evaluator used by `make_interval_rhs`.
    step_cfg : ODEFilterConfig
        One-step posterior update configuration.
    delta : float
        Fixed step size used for every scan iteration.
    use_ekf1_tk_initialization : bool
        If True, apply EKF1 t_k initialization; otherwise EKF0.
    sampling : tuple[bool, float]
        `(sample_posterior_position, variance_floor)` output policy.

    Public methods
    --------------
    make_rhs(t_k, coeffs_k)
        Build interval-specific rhs for one scan step.
    step(carry, inp)
        Advance one scan step and emit `(x, mean, cov)`.
    """

    sde: SDESpec
    get_coeffs: Any
    eval_fn: Any
    step_cfg: ODEFilterConfig
    delta: float
    use_ekf1_tk_initialization: bool = True
    sampling: tuple[bool, float] = (False, 1e-12)

    def make_rhs(self, t_k: float, coeffs_k):
        """Construct the local random ODE right-hand side for one step.

        Parameters
        ----------
        t_k : float
            Global start time of the current step.
        coeffs_k : tuple
            Brownian-basis coefficients sampled for the current step.

        Returns
        -------
        callable
            Function `rhs(z, t)` used by the ODE integrator on `[0, delta]`.
        """
        return make_interval_rhs(
            sde=self.sde,
            t_k=t_k,
            delta=self.delta,
            coeffs_k=coeffs_k,
            eval_fn=self.eval_fn,
        )

    def step(self, carry, inp):
        """Advance one `jax.lax.scan` iteration.

        Parameters
        ----------
        carry : tuple[jnp.ndarray, jnp.ndarray]
            Current latent state `(mean, cov)`.
        inp : tuple[jax.Array, float]
            `(key_k, t_k)` for this interval.

        Returns
        -------
        tuple
            `((mean_new, cov_new), (x_new, mean_new, cov_new))`
        """
        key_k, t_k = inp
        bm_key, key_sample = jax.random.split(key_k, 2)
        coeffs_k = self.get_coeffs(bm_key, self.delta)
        rhs = self.make_rhs(t_k, coeffs_k)

        carry_in = carry_after_tk_initialization_from_rhs(
            carry=carry,
            rhs_fn=rhs,
            use_ekf1=self.use_ekf1_tk_initialization,
        )

        mean_new, cov_new = ode_filter_step(
            carry_in[0],
            carry_in[1],
            rhs,
            (0.0, self.delta),
            self.step_cfg,
        )
        x_new = self._select_next_x(key_sample, mean_new, cov_new)
        return (mean_new, cov_new), (x_new, mean_new, cov_new)

    def _select_next_x(
        self,
        key_sample: jax.Array,
        mean_new: jnp.ndarray,
        cov_new: jnp.ndarray,
    ) -> jnp.ndarray:
        """Select scan output position from posterior mean/marginal."""
        sample_posterior_position, variance_floor = self.sampling
        cfg = PositionSamplingConfig(
            sample_posterior_position=sample_posterior_position,
            variance_floor=variance_floor,
            require_key_when_sampling=False,
        )
        return select_posterior_position(
            mean=mean_new,
            cov=cov_new,
            cfg=cfg,
            key=key_sample,
        )


def make_scan_inputs(
        key, config) -> tuple[jnp.ndarray, tuple[jax.Array, jnp.ndarray]]:
    """Build the integration grid and per-step scan inputs.

    Parameters
    ----------
    key : jax.Array
        Root PRNG key split into one key per integration step.
    config : MixtureSDEFilterConfig
        Solver configuration providing `delta` and `num_steps`.

    Returns
    -------
    tuple[jnp.ndarray, tuple[jax.Array, jnp.ndarray]]
        `(ts, (keys, ts_left))` where:
        - `ts` has shape `(num_steps + 1,)` and is the inclusive time grid,
        - `keys` has shape `(num_steps, 2)` (JAX PRNG keys),
        - `ts_left = ts[:-1]` contains step start times used by scan.
    """
    ts = jnp.linspace(0.0, config.num_steps * config.delta, config.num_steps + 1)
    keys = jax.random.split(key, config.num_steps)
    return ts, (keys, ts[:-1])


def make_scan_context(
    sde: SDESpec,
    step_cfg: ODEFilterConfig,
    config,
) -> ScanContext:
    """Construct immutable scan dependencies for Algorithm-3 stepping.

    Parameters
    ----------
    sde : SDESpec
        SDE specification containing drift, diffusion, initial value, and
        Brownian approximation factory.
    step_cfg : ODEFilterConfig
        One-step Gaussian ODE-filter update configuration.
    config : MixtureSDEFilterConfig
        Solver options controlling step size, t_k initialization mode, and
        trajectory sampling policy.

    Returns
    -------
    ScanContext
        Context object consumed by `jax.lax.scan`, bundling Brownian coefficient
        sampling, rhs construction settings, one-step update config, and output
        sampling policy.
    """
    get_coeffs, eval_fn = sde.bm_factory()
    return ScanContext(
        sde=sde,
        get_coeffs=get_coeffs,
        eval_fn=eval_fn,
        step_cfg=step_cfg,
        delta=config.delta,
        use_ekf1_tk_initialization=config.ekf.use_ekf1_tk_initialization,
        sampling=(config.sample_posterior_position, config.variance_floor),
    )


def prepend_uncertainty(
    sde: SDESpec,
    x0: jnp.ndarray,
    means: jnp.ndarray,
    covs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Prepend initial latent moments to scan-emitted uncertainty arrays.

    Parameters
    ----------
    sde : SDESpec
        SDE specification used to compute the initial derivative component.
    x0 : jnp.ndarray
        Initial path value at t0.
    means : jnp.ndarray
        Posterior latent means emitted by scan for steps `1..K`.
    covs : jnp.ndarray
        Posterior latent covariances emitted by scan for steps `1..K`.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        `(means_out, covs_out)` with initial moments inserted at index 0, so
        outputs align with the full grid `[t0, ..., tK]`.

    Notes
    -----
    The initial covariance is set to `1e-8 * I` in the latent state dimension,
    matching the solver initialization convention.
    """
    state_dim = means.shape[1]
    init_mean = jnp.array([jnp.squeeze(x0), jnp.squeeze(sde.drift(x0, 0.0))])
    if jnp.size(init_mean) != state_dim:
        init_mean = jnp.concatenate(
            [jnp.ravel(x0), jnp.ravel(jnp.asarray(sde.drift(x0, 0.0)))]
        )
    mean_out = insert(means, 0, init_mean)
    cov_out = insert(covs, 0, jnp.eye(state_dim) * 1e-8)
    return mean_out, cov_out