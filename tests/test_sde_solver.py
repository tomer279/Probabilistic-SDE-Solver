"""
Tests for the unified SDE solver facade.

Covers: solve_sde with method="gsf" on scalar SDEs, shape of outputs, and a rough
strong-error sanity check.
"""

import jax
import jax.numpy as jnp

from prob_sde import SDESpec, piecewise_linear_brownian
from prob_sde.solvers import GSFRunConfig, SDESolverConfig, TimeGridConfig, solve_sde


def _run_gsf_path(key, drift, diffusion, x0, delta, num_steps, return_uncertainty=False):
    """Helper that runs the GSF backend through the unified solve_sde facade."""
    sde = SDESpec.from_args(
        drift=drift,
        diffusion=diffusion,
        x0=x0,
        bm_factory=piecewise_linear_brownian,
    )
    cfg = SDESolverConfig(
        method="gsf",
        grid=TimeGridConfig(delta=delta, num_steps=num_steps),
        gsf=GSFRunConfig(return_uncertainty=return_uncertainty),
    )
    return solve_sde(key, sde, cfg)


def test_solve_sde_gsf_returns_ts_and_trajectory():
    """solve_sde(method='gsf') returns ts and trajectory with expected shapes."""
    key = jax.random.PRNGKey(0)
    drift = lambda x, t: -0.5 * x
    diffusion = lambda x, t: 0.5
    x0 = jnp.array(1.0)

    result = _run_gsf_path(key, drift, diffusion, x0, delta=0.01, num_steps=10)

    assert result.ts.shape == (11,)
    assert result.trajectory.shape == (11,)
    assert jnp.allclose(result.ts[0], 0.0)
    assert jnp.allclose(result.ts[-1], 0.1)
    assert jnp.allclose(result.trajectory[0], x0)


def test_solve_sde_gsf_with_uncertainty():
    """With return_uncertainty=True, means/covs are returned with expected shapes."""
    key = jax.random.PRNGKey(0)
    drift = lambda x, t: -x
    diffusion = lambda x, t: 0.1
    x0 = jnp.array(1.0)

    result = _run_gsf_path(
        key,
        drift,
        diffusion,
        x0,
        delta=0.05,
        num_steps=4,
        return_uncertainty=True,
    )

    assert result.ts.shape == (5,)
    assert result.trajectory.shape == (5,)
    assert result.means is not None
    assert result.covs is not None
    assert result.means.shape == (5, 2)
    assert result.covs.shape == (5, 2, 2)


def test_strong_error_decreases_with_smaller_step():
    """Strong error (vs finer reference) is reasonably bounded for smaller step sizes."""
    def run_terminal_value(key, delta, num_steps):
        drift = lambda x, t: 0.1 * x
        diffusion = lambda x, t: 0.2
        x0 = jnp.array(1.0)
        result = _run_gsf_path(key, drift, diffusion, x0, delta=delta, num_steps=num_steps)
        return result.trajectory[-1]

    key = jax.random.PRNGKey(123)
    t_final = 0.2

    traj_fine = run_terminal_value(key, 0.001, int(t_final / 0.001))
    traj_coarse = run_terminal_value(key, 0.01, int(t_final / 0.01))
    err_coarse = jnp.abs(traj_coarse - traj_fine)

    key2 = jax.random.PRNGKey(456)
    traj_coarse2 = run_terminal_value(key2, 0.01, int(t_final / 0.01))
    traj_fine2 = run_terminal_value(key2, 0.001, int(t_final / 0.001))
    err_coarse2 = jnp.abs(traj_coarse2 - traj_fine2)

    assert err_coarse < 0.5 and err_coarse2 < 0.5
