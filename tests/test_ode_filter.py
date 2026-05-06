"""
Tests for the Gaussian ODE filter (Algorithm 2) and prior model.

Covers: prior transition/covariance, low-level step API with ODEFilterConfig,
and integrator factory signature.
"""

import jax.numpy as jnp

from prob_sde import IWP2Prior, ode_filter_step, ode_integrator_factory
from prob_sde.filtering.ode import ODEFilterConfig


def test_prior_transition_matrix():
    """IWP2 transition F is [[1, dt], [0, 1]]."""
    prior = IWP2Prior(diffusion=1.0)
    f = prior.transition_matrix(0.1)
    expected = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    assert jnp.allclose(f, expected)


def test_prior_process_covariance_shape():
    """Process covariance Q is 2x2 symmetric."""
    prior = IWP2Prior(diffusion=1.0)
    q = prior.process_covariance(0.1)
    assert q.shape == (2, 2)
    assert jnp.allclose(q, q.T)


def test_ode_filter_step_returns_mean_cov():
    """ode_filter_step returns (mean_new, cov_new) with correct shapes."""
    prior = IWP2Prior(1.0)
    cfg = ODEFilterConfig(prior=prior, ekf_mode="ekf1", measurement_noise=1e-6)
    mean = jnp.array([1.0, 0.0])
    cov = jnp.eye(2) * 0.01
    vector_field = lambda x, t: -x

    mean_new, cov_new = ode_filter_step(mean, cov, vector_field, (0.0, 0.1), cfg)

    assert mean_new.shape == (2,)
    assert cov_new.shape == (2, 2)


def test_ode_filter_on_exponential_decay():
    """On dx/dt = -x, filter should keep derivative negative and position plausible."""
    prior = IWP2Prior(0.5)
    cfg = ODEFilterConfig(prior=prior, ekf_mode="ekf1", measurement_noise=1e-6)
    mean = jnp.array([1.0, -0.5])
    cov = jnp.eye(2) * 0.1
    vector_field = lambda x, t: -x

    mean_new, cov_new = ode_filter_step(mean, cov, vector_field, (0.0, 0.1), cfg)

    assert mean_new[1] < 0
    assert jnp.allclose(mean_new[0], jnp.exp(-0.1), atol=0.2)
    assert cov_new.shape == (2, 2)


def test_ode_integrator_factory_returns_callable():
    """ode_integrator_factory(prior) returns a callable."""
    prior = IWP2Prior(1.0)
    step = ode_integrator_factory(prior)
    assert callable(step)


def test_ode_integrator_step_signature():
    """Stateful integrator step: (mean, cov), vector_field, dt -> (mean_new, cov_new)."""
    prior = IWP2Prior(1.0)
    step = ode_integrator_factory(prior)
    vector_field = lambda x, t: -x

    x0 = jnp.array(1.0)
    mean0 = prior.initial_mean(x0, derivatives=(vector_field(x0, 0.0),))
    cov0 = prior.initial_covariance(scale=1e-8)
    carry0 = (mean0, cov0)

    mean_new, cov_new = step(carry0, vector_field, 0.1)

    assert mean_new.shape == (2,)
    assert cov_new.shape == (2, 2)
