"""Utility functions for Gaussian filter state representations.

This module centralizes shared operations used by public filter state classes.

Exports
-------
to_carry
    Return tuple-form latent Gaussian carry `(mean, cov)` for scan-compatible APIs.
position_marginal
    Return posterior position marginal statistics from latent Gaussian moments.
"""

import jax.numpy as jnp


def to_carry(mean: jnp.ndarray, cov: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return latent moments in tuple form.
    
    Parameters
    ----------
    mean : jnp.ndarray
        Latent Gaussian mean vector.
    cov : jnp.ndarray
        Latent Gaussian covariance matrix.
    
    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        The pair `(mean, cov)`, suitable for tuple-based scan/integrator carries.
    """
    return mean, cov


def position_marginal(
    mean: jnp.ndarray,
    cov: jnp.ndarray,
    variance_floor: float = 1e-12,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return first-component posterior marginal statistics.

    Parameters
    ----------
    mean : jnp.ndarray
        Latent Gaussian mean vector.
    cov : jnp.ndarray
        Latent Gaussian covariance matrix.
    variance_floor : float, optional
        Minimum variance returned for numerical stability.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        `(mean_pos, var_pos)` where `mean_pos = mean[0]` and
        `var_pos = max(cov[0, 0], variance_floor)`.
    """
    mean_pos = jnp.asarray(mean[0])
    var_pos = jnp.maximum(jnp.asarray(cov[0, 0]), variance_floor)
    return mean_pos, var_pos
