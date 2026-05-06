"""Brownian approximation and pathwise RHS utilities.

Exports:
- piecewise_linear_brownian, piecewise_parabolic_brownian: Brownian path approximations.
- brownian_and_parabolic_coeffs, parabolic_coeffs_from_fine_window: coefficient constructors.
- parabolic_dbeta_dt: derivative helper for parabolic Brownian approximation.
- make_interval_rhs: interval RHS constructor for pathwise methods.
"""

from .brownian import (
    brownian_and_parabolic_coeffs,
    parabolic_coeffs_from_fine_window,
    parabolic_dbeta_dt,
    piecewise_linear_brownian,
    piecewise_parabolic_brownian,
)
from .pathwise_rhs import make_interval_rhs

__all__ = [
    "piecewise_linear_brownian",
    "piecewise_parabolic_brownian",
    "brownian_and_parabolic_coeffs",
    "parabolic_coeffs_from_fine_window",
    "parabolic_dbeta_dt",
    "make_interval_rhs",
]