"""
Tests for the marginalised SDE solver facade.

Covers: marginalised solve_sde returns mean/variance trajectories with expected shapes.
"""

import jax
import jax.numpy as jnp

from prob_sde import SDESpec, piecewise_linear_brownian
from prob_sde.solvers import (
    MarginalisedRunConfig,
    SDESolverConfig, 
    TimeGridConfig, 
    solve_sde
)

from prob_sde.filtering.sde.marginalised import (
    MarginalisedConfig,
    solve_sde_marginalised,
    solve_sde_marginalised_batch,
)


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


def test_solve_sde_marginalised_batch_matches_manual_loop():
    """Batched marginalised solve matches manually stacked single-key solves."""
    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, 4)

    drift = lambda x, t: -0.5 * x
    diffusion = lambda x, t: 0.3
    x0 = jnp.array(1.0)

    sde = SDESpec.from_args(drift, diffusion, x0, piecewise_linear_brownian)
    config = MarginalisedConfig(
        delta=0.05,
        num_steps=4,
        sample_posterior_position=True,
        use_ekf1=True,
        variance_floor=1e-12,
        prior_diffusion=1.0,
        return_uncertainty=False,
    )

    ts_batch, trajectories_batch = solve_sde_marginalised_batch(keys, sde, config)

    loop_trajectories = []
    for key_i in keys:
        ts_loop, trajectory_i = solve_sde_marginalised(key_i, sde, config)
        loop_trajectories.append(trajectory_i)

    trajectories_loop = jnp.stack(loop_trajectories)

    assert jnp.allclose(ts_batch, ts_loop)
    assert trajectories_batch.shape == (4, 5)
    assert jnp.allclose(trajectories_batch, trajectories_loop, atol=1e-10)

def test_solve_sde_marginalised_facade_matches_manual_batch_aggregation():
    """Facade mean and variance match manual aggregation over batched paths."""
    key = jax.random.PRNGKey(321)
    num_samples = 5

    drift = lambda x, t: -0.5 * x
    diffusion = lambda x, t: 0.3
    x0 = jnp.array(1.0)

    sde = SDESpec.from_args(drift, diffusion, x0, piecewise_linear_brownian)
    grid = TimeGridConfig(delta=0.05, num_steps=4)
    run_cfg = MarginalisedRunConfig(prior_scale=1.0, num_samples=num_samples)

    solver_cfg = SDESolverConfig(
        method="marginalised",
        grid=grid,
        marginalised=run_cfg,
    )

    out = solve_sde(key, sde, solver_cfg)

    marginalised_cfg = MarginalisedConfig(
        delta=grid.delta,
        num_steps=grid.num_steps,
        sample_posterior_position=True,
        use_ekf1=True,
        variance_floor=1e-12,
        prior_diffusion=run_cfg.prior_scale,
        return_uncertainty=False,
    )
    sample_keys = jax.random.split(key, num_samples)
    expected_ts, samples = solve_sde_marginalised_batch(
        sample_keys,
        sde,
        marginalised_cfg,
    )

    expected_mean = jnp.mean(samples, axis=0)
    expected_var = jnp.var(samples, axis=0)

    assert jnp.allclose(out.ts, expected_ts)
    assert jnp.allclose(out.mean_trajectory, expected_mean, atol=1e-10)
    assert jnp.allclose(out.var_trajectory, expected_var, atol=1e-10)
    assert jnp.allclose(out.trajectory, expected_mean, atol=1e-10)
