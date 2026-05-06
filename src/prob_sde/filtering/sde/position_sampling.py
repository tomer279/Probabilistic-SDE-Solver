"""Shared posterior-position selection for probabilistic SDE filters.

This module centralizes the logic for extracting the position component from a
posterior Gaussian state and, when configured, sampling that position from its
marginal distribution. It is intended to remove duplicated sampling code across
Gaussian, mixture, and marginalised SDE filter implementations.

Exports
-------
PositionSamplingConfig
    Immutable sampling policy for posterior position extraction, including
    deterministic vs sampled mode, variance flooring, and key requirements.
select_posterior_position
    Return the next position from posterior moments either deterministically
    (posterior mean) or stochastically (Gaussian sample).
"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp

@dataclass(frozen=True)
class PositionSamplingConfig:
    """Sampling policy for posterior position extraction.
    Instance variables
    ------------------
    sample_posterior_position : bool
        If True, sample from N(mean[0], cov[0,0]); else return mean[0].
    variance_floor : float
        Minimum variance for stable sampling.
    require_key_when_sampling : bool
        If True, raise ValueError when sampling is requested but key is None.
    """
    sample_posterior_position: bool
    variance_floor: float
    require_key_when_sampling: bool = False

def select_posterior_position(
    mean: jnp.ndarray,
    cov: jnp.ndarray,
    cfg: PositionSamplingConfig,
    key: jax.Array | None = None,
) -> jnp.ndarray:
    """Return posterior position as mean or Gaussian sample.

    This helper extracts the position component from a latent Gaussian state and
    applies a unified sampling policy used by SDE filters. In deterministic mode
    it returns the posterior position mean. In sampling mode it draws from the
    scalar marginal ``N(mean[0], cov[0,0])`` with variance flooring for numerical
    stability.

    Parameters
    ----------
    mean : jnp.ndarray
        Posterior latent mean vector. The position component is read at index 0.
    cov : jnp.ndarray
        Posterior latent covariance matrix. The position variance is read from
        entry ``cov[0,0]``.
    cfg : PositionSamplingConfig
        Sampling policy:
        - ``sample_posterior_position`` controls deterministic vs sampled output.
        - ``variance_floor`` defines the minimum variance used for sampling.
        - ``require_key_when_sampling`` controls whether missing ``key`` raises.
    key : jax.Array | None, optional
        PRNG key used only when sampling is enabled.

    Returns
    -------
    jnp.ndarray
        Posterior position value. This is ``mean[0]`` in deterministic mode, or
        a sample from ``N(mean[0], max(cov[0,0], variance_floor))`` in sampling
        mode.

    Raises
    ------
    ValueError
        If sampling is enabled and no PRNG key is provided.
    """
    mean_pos = jnp.asarray(mean[0])
    if not cfg.sample_posterior_position:
        return mean_pos
    if cfg.require_key_when_sampling and key is None:
        raise ValueError(
            "Sampling requested but no PRNG key provided for posterior position."
        )
    if key is None:
        raise ValueError("Sampling requested but key is None.")
    var_pos = jnp.maximum(jnp.asarray(cov[0, 0]), cfg.variance_floor)
    eps = jax.random.normal(key, shape=mean_pos.shape)
    return mean_pos + jnp.sqrt(var_pos) * eps
