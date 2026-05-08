"""SDE filtering algorithms.

Exports:
- GaussianSDEFilterConfig, GaussianSDEFilterState, GaussianSDEFilter: Gaussian pathwise filter.
- MixtureSDEFilterConfig, MixtureSDEFilterState, MixtureSDEFilter: mixture pathwise filter.
- solve_sde_pathwise_mixture, solve_sde_pathwise_mixture_with_coeffs: mixture solver entrypoints.
- MarginalisedConfig, solve_sde_marginalised: marginalised filtering solver.
- PositionSamplingConfig, select_posterior_position: posterior position selection.
- to_carry, position_marginal: state helper utilities.
"""

from .gaussian_sde_filter import GaussianSDEFilter, GaussianSDEFilterConfig, GaussianSDEFilterState
from .marginalised import (
    MarginalisedConfig,
    solve_sde_marginalised,
    solve_sde_marginalised_batch
)
from .mixture_sde_filter import (
    MixtureSDEFilter,
    MixtureSDEFilterConfig,
    MixtureSDEFilterState,
    solve_sde_pathwise_mixture,
    solve_sde_pathwise_mixture_with_coeffs,
)
from .position_sampling import PositionSamplingConfig, select_posterior_position
from .state_helpers import position_marginal, to_carry

__all__ = [
    "GaussianSDEFilter",
    "GaussianSDEFilterConfig",
    "GaussianSDEFilterState",
    "MixtureSDEFilter",
    "MixtureSDEFilterConfig",
    "MixtureSDEFilterState",
    "solve_sde_pathwise_mixture",
    "solve_sde_pathwise_mixture_with_coeffs",
    "MarginalisedConfig",
    "solve_sde_marginalised",
    "solve_sde_marginalised_batch",
    "PositionSamplingConfig",
    "select_posterior_position",
    "to_carry",
    "position_marginal",
]