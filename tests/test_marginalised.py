"""
Tests for the marginalised SDE solver.

Covers: solve_sde_marginalised returns correct shapes, variance is positive.
"""

import jax
import jax.numpy as jnp
from prob_sde.brownian import piecewise_linear_brownian
from prob_sde.ode_filter import ode_integrator_factory
from prob_sde.prior_models import IWP2Prior
from prob_sde.marginalised import solve_sde_marginalised


def test_solve_sde_marginalised_returns_three_arrays():
    """solve_sde_marginalised returns (ts, mean_trajectory, var_trajectory)."""
    key = jax.random.PRNGKey(0)
    drift = lambda x, t: -0.5 * x
    diffusion = lambda x, t: 0.3
    x0 = jnp.array(1.0)
    prior = IWP2Prior(1.0)
    ode_int = ode_integrator_factory(prior)
    ts, mean_traj, var_traj = solve_sde_marginalised(
        key, drift, diffusion, x0, piecewise_linear_brownian, 0.02, 5, ode_int, num_samples=20
    )
    assert ts.shape == (6,)
    assert mean_traj.shape == (6,)
    assert var_traj.shape == (6,)
    assert jnp.all(var_traj >= -1e-10)


def test_marginalised_mean_near_pathwise_for_small_variance():
    """With one sample, marginalised mean equals that path (no averaging)."""
    key = jax.random.PRNGKey(0)
    drift = lambda x, t: -x
    diffusion = lambda x, t: 0.1
    x0 = jnp.array(1.0)
    prior = IWP2Prior(1.0)
    ode_int = ode_integrator_factory(prior)
    ts, mean_traj, var_traj = solve_sde_marginalised(
        key, drift, diffusion, x0, piecewise_linear_brownian, 0.05, 4, ode_int, num_samples=1
    )
    assert jnp.allclose(var_traj, 0.0, atol=1e-10)
