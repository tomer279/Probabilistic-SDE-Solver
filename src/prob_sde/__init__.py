"""Public package API for probabilistic SDE solvers.

Exports:
- SDESpec: SDE specification container.
- IWPPrior, IWP2Prior, IWP3Prior: integrated Wiener process priors.
- piecewise_linear_brownian, piecewise_parabolic_brownian: Brownian approximations.
- brownian_and_parabolic_coeffs, parabolic_coeffs_from_fine_window: Brownian coefficient helpers.
- ode_filter_step, ode_integrator_factory: ODE filtering routines.
- solve_sde_pathwise_mixture, solve_sde_pathwise_mixture_with_coeffs: mixture pathwise solvers.
- solve_sde_marginalised: marginalised SDE solver.
- solve_sde: high-level solver entry point.
- insert, time_grid, split_key: utility helpers.
"""

from .brownian.brownian import (
    brownian_and_parabolic_coeffs,
    parabolic_coeffs_from_fine_window,
    piecewise_linear_brownian,
    piecewise_parabolic_brownian,
)
from .core.prior_models import IWP2Prior, IWP3Prior, IWPPrior
from .core.sde import SDESpec
from .filtering.ode.ode_filter import ode_filter_step, ode_integrator_factory
from .filtering.sde.marginalised import solve_sde_marginalised
from .filtering.sde.mixture_sde_filter import (
    solve_sde_pathwise_mixture,
    solve_sde_pathwise_mixture_with_coeffs,
)
from .solvers.sde_solver import solve_sde
from .utils.utils import insert, split_key, time_grid

__all__ = [
    "SDESpec",
    "IWPPrior",
    "IWP2Prior",
    "IWP3Prior",
    "piecewise_linear_brownian",
    "piecewise_parabolic_brownian",
    "brownian_and_parabolic_coeffs",
    "parabolic_coeffs_from_fine_window",
    "ode_filter_step",
    "ode_integrator_factory",
    "solve_sde_pathwise_mixture",
    "solve_sde_pathwise_mixture_with_coeffs",
    "solve_sde_marginalised",
    "solve_sde",
    "insert",
    "time_grid",
    "split_key",
]