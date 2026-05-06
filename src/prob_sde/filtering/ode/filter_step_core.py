"""Shared one-step posterior update wrapper for SDE filters.

Exports:
- run_shared_posterior_step: Execute one EKF0/1 posterior step using ode_filter_step.
"""

from dataclasses import dataclass
import jax.numpy as jnp

from .ode_filter import ODEFilterConfig, ode_filter_step


@dataclass(frozen=True)
class PosteriorStepResult:
    """Posterior moments returned by one shared EKF step.

    Public methods:
    - position_marginal: Return first component mean/variance.

    Instance variables:
    - mean: Posterior latent mean.
    - cov: Posterior latent covariance.
    """
    mean: jnp.ndarray
    cov: jnp.ndarray

    def position_marginal(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return first component posterior mean and variance."""
        return self.mean[0], self.cov[0, 0]


def run_shared_posterior_step(
    mean_init: jnp.ndarray,
    cov_init: jnp.ndarray,
    vector_field,
    delta: float,
    cfg: ODEFilterConfig,
) -> PosteriorStepResult:
    """Run one shared EKF0/1 posterior step over [0, delta]."""
    mean_new, cov_new = ode_filter_step(
        mean_init,
        cov_init,
        vector_field,
        (0.0, delta),
        cfg,
    )
    return PosteriorStepResult(mean=mean_new, cov=cov_new)