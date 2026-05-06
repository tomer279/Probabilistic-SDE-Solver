"""Shared interval-local RHS builders for pathwise SDE filters.

Exports
-------
make_interval_rhs
    Build interval-local random vector field on [0, delta] from Brownian coefficients.
"""
from __future__ import annotations

from typing import Callable

import jax

from prob_sde.core.sde import SDESpec
from .brownian import parabolic_dbeta_dt


def make_interval_rhs(
    sde: SDESpec,
    t_k: float,
    delta: float,
    coeffs_k,
    eval_fn: Callable | None = None,
):
    """Build interval-local random RHS f_k(x, tau_local) on [0, delta].

    Parameters
    ----------
    sde : SDESpec
        Drift/diffusion model.
    t_k : float
        Global interval start time.
    delta : float
        Interval length.
    coeffs_k : tuple
        Brownian approximation coefficients for this interval.
    eval_fn : callable | None
        Basis evaluator used for non-parabolic families. Required when
        coeffs_k does not match parabolic triple shape.

    Returns
    -------
    callable
        f_k(x, tau_local) = drift(x, t_k + tau_local) + diffusion(x, t_k + tau_local) * d_beta
    """
    def rhs(x, tau_local):
        t_global = t_k + tau_local

        if len(coeffs_k) == 3:
            w0, w_delta, i_delta = coeffs_k
            d_beta = parabolic_dbeta_dt(tau_local, delta, w0, w_delta, i_delta)
        else:
            if eval_fn is None:
                raise ValueError("eval_fn is required for non-parabolic coefficient families.")
            d_beta = jax.jacfwd(eval_fn, 0)(tau_local, delta, *coeffs_k)

        drift_val = sde.drift(x, t_global)
        diffusion_val = sde.diffusion(x, t_global)
        return drift_val + diffusion_val * d_beta

    return rhs
