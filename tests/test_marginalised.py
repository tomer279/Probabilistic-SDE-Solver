"""
Tests for the marginalised SDE solver facade.

Covers: marginalised solve_sde returns mean/variance trajectories with expected shapes.
"""

import jax
import jax.numpy as jnp

from prob_sde import SDESpec, piecewise_linear_brownian
from prob_sde.solvers import MarginalisedRunConfig, SDESolverConfig, TimeGridConfig, solve_sde


def test_solve_sde_marginalised_returns_three_arrays():
    """Facade marginalised method returns ts, mean_trajectory, var_trajectory."""
    key = jax.random.PRNGKey(0)
    drift = lambda x, t: -0.5 * x
    diffusion = lambda x, t: 0.3
    x0 = jnp.array(1.0)

    sde = SDESpec.from_args(drift, diffusion, x0, piecewise_linear_brownian)
    cfg = SDESolverConfig(
        method="marginalised",
        grid=TimeGridConfig(delta=0.02, num_steps=5),
        marginalised=MarginalisedRunConfig(prior_scale=1.0, num_samples=20),
    )
    out = solve_sde(key, sde, cfg)

    assert out.ts.shape == (6,)
    assert out.mean_trajectory.shape == (6,)
    assert out.var_trajectory.shape == (6,)
    assert jnp.all(out.var_trajectory >= -1e-10)


def test_marginalised_mean_near_pathwise_for_small_variance():
    """With one sample, marginalised variance is ~0 pointwise."""
    key = jax.random.PRNGKey(0)
    drift = lambda x, t: -x
    diffusion = lambda x, t: 0.1
    x0 = jnp.array(1.0)

    sde = SDESpec.from_args(drift, diffusion, x0, piecewise_linear_brownian)
    cfg = SDESolverConfig(
        method="marginalised",
        grid=TimeGridConfig(delta=0.05, num_steps=4),
        marginalised=MarginalisedRunConfig(prior_scale=1.0, num_samples=1),
    )
    out = solve_sde(key, sde, cfg)

    assert jnp.allclose(out.var_trajectory, 0.0, atol=1e-10)
