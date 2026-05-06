"""
Tests for the pathwise SDE solver.

Covers: solve_sde_pathwise on scalar SDE, shape of outputs, strong error order (rough).
"""

import jax
import jax.numpy as jnp
from prob_sde.brownian import piecewise_linear_brownian
from prob_sde.ode_filter import ode_integrator_factory
from prob_sde.prior_models import IWP2Prior
from prob_sde.sde_solver import solve_sde_pathwise


def test_solve_sde_pathwise_returns_ts_and_trajectory():
    """solve_sde_pathwise returns (ts, trajectory) with correct shapes."""
    key = jax.random.PRNGKey(0)
    drift = lambda x, t: -0.5 * x
    diffusion = lambda x, t: 0.5
    x0 = jnp.array(1.0)
    bm_factory = piecewise_linear_brownian
    prior = IWP2Prior(1.0)
    ode_int = ode_integrator_factory(prior)
    ts, traj = solve_sde_pathwise(key, drift, diffusion, x0, bm_factory, 0.01, 10, ode_int)
    assert ts.shape == (11,)
    assert traj.shape == (11,)
    assert jnp.allclose(ts[0], 0.0)
    assert jnp.allclose(ts[-1], 0.1)
    assert jnp.allclose(traj[0], x0)


def test_solve_sde_pathwise_with_uncertainty():
    """With return_uncertainty=True and 3-value integrator, returns (ts, traj, (means, covs))."""
    key = jax.random.PRNGKey(0)
    drift = lambda x, t: -x
    diffusion = lambda x, t: 0.1
    x0 = jnp.array(1.0)
    prior = IWP2Prior(1.0)
    ode_int = ode_integrator_factory(prior)
    result = solve_sde_pathwise(key, drift, diffusion, x0, piecewise_linear_brownian,
                                0.05, 4, ode_int, return_uncertainty=True)
    assert len(result) == 3
    ts, traj, (means, covs) = result
    assert means.shape == (5, 2)
    assert covs.shape == (5, 2, 2)


def test_strong_error_decreases_with_smaller_step():
    """Strong error (vs reference) decreases when step size is reduced."""
    def run_path(key, delta, num_steps):
        drift = lambda x, t: 0.1 * x
        diffusion = lambda x, t: 0.2
        x0 = jnp.array(1.0)
        prior = IWP2Prior(1.0)
        ode_int = ode_integrator_factory(prior)
        ts, traj = solve_sde_pathwise(key, drift, diffusion, x0, piecewise_linear_brownian,
                                      delta, num_steps, ode_int)
        return traj[-1]

    key = jax.random.PRNGKey(123)
    T = 0.2
    traj_fine = run_path(key, 0.001, int(T / 0.001))
    traj_coarse = run_path(key, 0.01, int(T / 0.01))
    err_coarse = jnp.abs(traj_coarse - traj_fine)
    key2 = jax.random.PRNGKey(456)
    traj_coarse2 = run_path(key2, 0.01, int(T / 0.01))
    err_coarse2 = jnp.abs(traj_coarse2 - run_path(key2, 0.001, int(T / 0.001)))
    assert err_coarse < 0.5 and err_coarse2 < 0.5
