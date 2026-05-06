"""
Tests for the piecewise linear Brownian motion approximation.

Covers: get_coeffs/eval_fn interface, variance of increment, continuity at boundaries.
"""

import jax
import jax.numpy as jnp
from prob_sde import piecewise_linear_brownian


def test_piecewise_linear_brownian_returns_two_callables():
    """Factory returns (get_coeffs, eval_fn)."""
    get_coeffs, eval_fn = piecewise_linear_brownian()
    assert callable(get_coeffs)
    assert callable(eval_fn)


def test_get_coeffs_returns_tuple_of_two():
    """get_coeffs(key, delta) returns (w0, w_delta)."""
    get_coeffs, _ = piecewise_linear_brownian()
    key = jax.random.PRNGKey(0)
    coeffs = get_coeffs(key, 0.1)
    assert len(coeffs) == 2
    assert jnp.shape(coeffs[0]) == ()
    assert jnp.shape(coeffs[1]) == ()


def test_eval_linear_at_zero_and_delta():
    """eval_fn(0)=w0, eval_fn(delta)=w_delta."""
    _, eval_fn = piecewise_linear_brownian()
    delta = 0.25
    w0 = 0.0
    w_delta = 0.5
    at_zero = eval_fn(jnp.array(0.0), delta, w0, w_delta)
    at_delta = eval_fn(jnp.array(delta), delta, w0, w_delta)
    assert jnp.allclose(at_zero, w0)
    assert jnp.allclose(at_delta, w_delta)


def test_eval_linear_differentiable():
    """eval_fn is differentiable in t; derivative is (w_delta - w0)/delta."""
    _, eval_fn = piecewise_linear_brownian()
    delta = 0.2
    w0, w_delta = 0.0, 0.3
    t = jnp.array(0.1)
    deriv = jax.jacfwd(eval_fn, 0)(t, delta, w0, w_delta)
    expected = (w_delta - w0) / delta
    assert jnp.allclose(deriv, expected)


def test_increment_variance():
    """Over many samples, Var(W_delta) ≈ delta."""
    get_coeffs, _ = piecewise_linear_brownian()
    key = jax.random.PRNGKey(42)
    delta = 0.1
    keys = jax.random.split(key, 10000)
    increments = jax.vmap(lambda k: get_coeffs(k, delta)[1])(keys)
    var_est = jnp.var(increments)
    assert jnp.allclose(var_est, delta, rtol=0.1)


def test_continuity_at_boundary():
    """Linear interpolation is continuous: value at delta matches next segment's w0."""
    get_coeffs, eval_fn = piecewise_linear_brownian()
    key = jax.random.PRNGKey(0)
    delta = 0.1
    w0, w_delta = get_coeffs(key, delta)
    end_value = eval_fn(jnp.array(delta), delta, w0, w_delta)
    assert jnp.allclose(end_value, w_delta)
