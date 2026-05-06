"""
Algorithm-3 coefficient-driven rollout helpers.

Exported objects
----------------
solve_sde_pathwise_mixture_with_coeffs
    Run Algorithm 3 using caller-provided Brownian coefficients.
run_coeff_rollout
    Execute per-step rollout with optional posterior-position sampling keys.
"""

from collections.abc import Sequence
from typing import Any, Protocol

import jax
import jax.numpy as jnp

from prob_sde.core.prior_models import IWP2Prior
from prob_sde.core.sde import SDESpec


class _SupportsCoeffStep(Protocol):
    """Protocol for solver objects used by coefficient-driven rollout.

    Public methods
    --------------
    initialize(t0=0.0)
        Build initial state.
    step_with_coeffs(state_k, coeffs_k, delta=None, key=None)
        Advance one step using supplied interval coefficients.
    """

    def initialize(self, t0: float = 0.0) -> Any:
        """Return initial solver state."""
        ...

    def step_with_coeffs(
        self,
        state_k: Any,
        coeffs_k: Any,
        delta: float | None = None,
        key: jax.Array | None = None,
    ) -> Any:
        """Advance one coefficient-driven step."""
        ...


def run_coeff_rollout(
    solver: _SupportsCoeffStep,
    config: Any,
    sde: SDESpec,
    coeffs_list: Sequence[Any],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Roll out Algorithm 3 over supplied Brownian coefficients.

    Parameters
    ----------
    solver : _SupportsCoeffStep
        Initialized Algorithm-3 solver object exposing `initialize` and
        `step_with_coeffs`.
    config : MixtureSDEFilterConfig
        Solver configuration with `delta`, `sample_posterior_position`,
        and optional `sampling_key`.
    sde : SDESpec
        SDE specification used for initial trajectory value `x0`.
    coeffs_list : sequence
        Per-step Brownian approximation coefficients.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        `(traj, means, covs)` arrays aligned with the full grid
        `[t0, ..., tK]`.
    """
    state = solver.initialize(t0=0.0)
    traj = [jnp.asarray(sde.x0)]
    means = [state.mean]
    covs = [state.cov]

    if config.sample_posterior_position:
        if config.sampling_key is None:
            raise ValueError("sample_posterior_position=True requires config.sampling_key.")
        step_keys = jax.random.split(config.sampling_key, len(coeffs_list))
    else:
        step_keys = [None] * len(coeffs_list)

    for k, coeffs_k in enumerate(coeffs_list):
        state = solver.step_with_coeffs(
            state_k=state,
            coeffs_k=coeffs_k,
            delta=config.delta,
            key=step_keys[k],
        )
        traj.append(state.x)
        means.append(state.mean)
        covs.append(state.cov)

    return jnp.asarray(traj), jnp.asarray(means), jnp.asarray(covs)


def solve_sde_pathwise_mixture_with_coeffs(
    sde: SDESpec,
    prior: IWP2Prior,
    config: Any,
    coeffs_list: Sequence[Any],
    solver_factory,
):
    """Run Algorithm 3 with caller-provided coefficients.

    Parameters
    ----------
    sde : SDESpec
        Drift, diffusion, initial value, and Brownian-factory specification.
    prior : IWP2Prior
        IWP prior used for Algorithm-2 posterior updates.
    config : MixtureSDEFilterConfig
        Solver configuration.
    coeffs_list : sequence
        Per-step Brownian coefficients. Length must equal `config.num_steps`.
    solver_factory : callable
        Constructor-like callable used as:
        `solver_factory(sde=sde, prior=prior, config=config)`.

    Returns
    -------
    tuple
        If `config.return_uncertainty` is False:
            `(ts, traj)`
        else:
            `(ts, traj, (means, covs))`.
    """
    num_steps = len(coeffs_list)
    if num_steps != config.num_steps:
        raise ValueError("len(coeffs_list) must match config.num_steps.")

    solver = solver_factory(sde=sde, prior=prior, config=config)
    traj_arr, means_arr, covs_arr = run_coeff_rollout(
        solver=solver,
        config=config,
        sde=sde,
        coeffs_list=coeffs_list,
    )

    ts = jnp.linspace(0.0, config.num_steps * config.delta, config.num_steps + 1)
    if not config.return_uncertainty:
        return ts, traj_arr
    return ts, traj_arr, (means_arr, covs_arr)