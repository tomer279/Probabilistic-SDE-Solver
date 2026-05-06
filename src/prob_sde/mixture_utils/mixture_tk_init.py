"""
Algorithm-3 t_k initialization helpers.

Exported functions
------------------
carry_after_tk_initialization_from_rhs
    Dispatch to EKF0 or EKF1 t_k initialization using interval rhs.
apply_ekf0_initialization_at_tk_from_rhs
    Apply Eq. (15) update to latent mean/covariance at t_k.
apply_ekf1_initialization_at_tk_from_rhs
    Apply Eq. (14) update to latent mean/covariance at t_k.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp



def carry_after_tk_initialization_from_rhs(
        carry: tuple[jnp.ndarray, jnp.ndarray],
        rhs_fn: Callable[[jnp.ndarray, float], jnp.ndarray],
        use_ekf1: bool
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Prepare carry at t_k using interval-specific vector field f_k.

    Parameters
    ----------
    carry : tuple[jnp.ndarray, jnp.ndarray]
        Current latent Gaussian carry `(mean, cov)`.
    rhs_fn : Callable[[jnp.ndarray, float], jnp.ndarray]
        Interval-specific rhs function on local time.
    use_ekf1 : bool
        If True, apply Eq. (14); otherwise apply Eq. (15).

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Updated `(mean, cov)` at t_k.

    Notes
    -----
    EKF0 branch follows Eq. (15):
    - m1 <- f_k(m0, t_k)  (local tau=0 in rhs_fn)
    - P01 <- 0, P11 <- 0

    EKF1 branch follows Eq. (14):
    - m1 <- f_k(m0, t_k)
    - P01 <- P00 * d/dx f_k(m0, t_k)
    - P11 <- P00 * (d/dx f_k(m0, t_k))^2
    """
    mean_k, cov_k = carry

    if use_ekf1:
        return _apply_ekf1_initialization_at_tk_from_rhs(mean_k, cov_k, rhs_fn)
    return _apply_ekf0_initialization_at_tk_from_rhs(mean_k, cov_k, rhs_fn)


def _apply_ekf0_initialization_at_tk_from_rhs(
    mean: jnp.ndarray,
    cov: jnp.ndarray,
    rhs_fn: Callable[[jnp.ndarray, float], jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply Eq. (15) at t_k using interval-specific rhs.

    Sets m1 <- f_k(m0, t_k) and zeroes P01, P10, P11.
    """
    x0 = mean[0]
    m1 = rhs_fn(x0, 0.0)
    mean_up = mean.at[1].set(jnp.squeeze(jnp.asarray(m1)))
    cov_up = cov.at[0, 1].set(0.0).at[1, 0].set(0.0).at[1, 1].set(0.0)
    return mean_up, cov_up

def _apply_ekf1_initialization_at_tk_from_rhs(
    mean: jnp.ndarray,
    cov: jnp.ndarray,
    rhs_fn: Callable[[jnp.ndarray, float], jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply Eq. (14) at t_k using interval-specific rhs.

    Uses Jacobian d/dx f_k(m0, t_k) to set P01 and P11.
    """
    p00 = cov[0, 0]

    def rhs_in_x0(x0: jnp.ndarray) -> jnp.ndarray:
        vec = jnp.ravel(jnp.asarray(rhs_fn(x0, 0.0)))
        if vec.size != 1:
            raise ValueError("EKF1 tk init requires scalar rhs output.")
        return jnp.squeeze(vec)

    x0 = mean[0]
    m1 = rhs_in_x0(x0)
    dfdx_s = jnp.squeeze(jnp.asarray(jax.jacfwd(rhs_in_x0)(x0)))

    p01 = p00 * dfdx_s
    p11 = p00 * (dfdx_s ** 2)

    mean_up = mean.at[1].set(jnp.squeeze(jnp.asarray(m1)))
    cov_up = cov.at[0, 1].set(p01).at[1, 0].set(p01).at[1, 1].set(p11)
    return mean_up, cov_up
