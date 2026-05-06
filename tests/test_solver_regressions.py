"""
Regression tests for unified solver facade contracts.

Covers:
- config validation paths,
- deterministic behavior by PRNG key,
- uncertainty output contract for GSF.
"""

import jax
import jax.numpy as jnp
import pytest

from prob_sde import SDESpec, piecewise_linear_brownian
from prob_sde.solvers import GSFRunConfig, SDESolverConfig, TimeGridConfig, solve_sde


def _simple_sde():
    drift = lambda x, t: -0.2 * x
    diffusion = lambda x, t: 0.3
    x0 = jnp.array(1.0)
    return SDESpec.from_args(drift, diffusion, x0, piecewise_linear_brownian)


def test_solve_sde_rejects_unknown_method():
    """Unknown facade method should raise ValueError."""
    sde = _simple_sde()
    cfg = SDESolverConfig(
        method="unknown",
        grid=TimeGridConfig(delta=0.01, num_steps=5),
    )
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError, match="Unknown method"):
        solve_sde(key, sde, cfg)


def test_solve_sde_rejects_nonpositive_delta():
    """Grid validation should reject delta <= 0."""
    sde = _simple_sde()
    cfg = SDESolverConfig(
        method="gsf",
        grid=TimeGridConfig(delta=0.0, num_steps=5),
        gsf=GSFRunConfig(),
    )
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError, match="delta must be positive"):
        solve_sde(key, sde, cfg)


def test_gsf_deterministic_for_same_key():
    """Same key and same config produce identical trajectory."""
    sde = _simple_sde()
    cfg = SDESolverConfig(
        method="gsf",
        grid=TimeGridConfig(delta=0.01, num_steps=10),
        gsf=GSFRunConfig(return_uncertainty=False),
    )
    key = jax.random.PRNGKey(123)

    out1 = solve_sde(key, sde, cfg)
    out2 = solve_sde(key, sde, cfg)

    assert jnp.allclose(out1.ts, out2.ts)
    assert jnp.allclose(out1.trajectory, out2.trajectory)


def test_gsf_changes_with_different_keys():
    """Different keys should usually produce different trajectories."""
    sde = _simple_sde()
    cfg = SDESolverConfig(
        method="gsf",
        grid=TimeGridConfig(delta=0.01, num_steps=10),
        gsf=GSFRunConfig(return_uncertainty=False),
    )
    key1 = jax.random.PRNGKey(123)
    key2 = jax.random.PRNGKey(456)

    out1 = solve_sde(key1, sde, cfg)
    out2 = solve_sde(key2, sde, cfg)

    # Compare terminal value to avoid over-constraining pathwise equality checks.
    assert not jnp.allclose(out1.trajectory[-1], out2.trajectory[-1])


def test_gsf_uncertainty_contract():
    """return_uncertainty toggles means/covs presence and shapes."""
    sde = _simple_sde()
    grid = TimeGridConfig(delta=0.02, num_steps=6)
    key = jax.random.PRNGKey(0)

    out_no_uq = solve_sde(
        key,
        sde,
        SDESolverConfig(method="gsf", grid=grid, gsf=GSFRunConfig(return_uncertainty=False)),
    )
    assert out_no_uq.means is None
    assert out_no_uq.covs is None

    out_uq = solve_sde(
        key,
        sde,
        SDESolverConfig(method="gsf", grid=grid, gsf=GSFRunConfig(return_uncertainty=True)),
    )
    assert out_uq.means is not None
    assert out_uq.covs is not None
    assert out_uq.means.shape == (7, 2)
    assert out_uq.covs.shape == (7, 2, 2)