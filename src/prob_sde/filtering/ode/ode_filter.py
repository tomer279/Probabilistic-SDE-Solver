"""
Algorithm 2 (Gaussian SDE Filter / EKF0) utilities for probabilistic ODE integration.

This module implements the one-step Gaussian filter update used as Algorithm 2 
from Le Fay, Särkkä & Corenflos (2025) in the method description.
A filter step consists of:
1) prior prediction with the IWP-2 state model, and
2) residual-based Kalman update enforcing the ODE relation
   ``x'(t) = f(x(t), t)``.

Exported objects
----------------
ODEFilterState
    Immutable container for the Gaussian filter state (mean and covariance).

ode_filter_init(prior, x0, rhs0, cov_scale=1e-6)
    Build the initial filter state from an ODE initial condition.

ode_filter_advance(state, vector_field, dt, cfg)
    Advance an ``ODEFilterState`` by one Algorithm-2 step using ``cfg``.

ode_filter_step(mean, cov, vector_field, step_params, cfg)
    Low-level Algorithm-2 update on mean/covariance arrays.

ode_integrator_factory(prior, measurement_noise=1e-6, ekf_mode="ekf1")
    Create a one-step integrator callable that applies Algorithm 2.

Notes
-----
- This module contains the concrete implementation of Algorithm 2.
- The pathwise SDE solver consumes it via an injected ``ode_integrator``
  (typically produced by ``ode_integrator_factory``).
- Jacobians of the vector field are computed with JAX automatic differentiation.

Algorithm mapping
-----------------
- Algorithm 2: implemented directly in this module.
- Algorithm 1: not implemented here; its Brownian approximation enters
  indirectly through the local RHS passed into the integrator.
- Algorithm 3: this module supplies the per-step update used by the outer
  pathwise loop.
"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from prob_sde.core.prior_models import IWP2Prior

@dataclass(frozen=True)
class ODEFilterConfig:
    """Configuration for one-step ODE filtering updates.
    
    Instance variables
    ------------------
    prior : IWP2Prior
        Prior dynamics model providing transition matrix and process covariance.
    ekf_mode : str
        Linearization mode for the observation update. Must be ``"ekf0"`` or
        ``"ekf1"``.
    measurement_noise : float
        Observation-noise variance used in the Kalman update.
    """
    prior: IWP2Prior
    ekf_mode: str = "ekf1"
    measurement_noise: float = 1e-6

@dataclass(frozen=True)
class ODEFilterState:
    """Posterior EKF0 state for a single ODE filtering time point.

    This dataclass stores the Gaussian belief over the latent IWP-2 state used
    by the ODE filter. The state is represented by a mean vector and covariance
    matrix, where the first block corresponds to the ODE position and the second
    block corresponds to its first derivative.

    Attributes
    ----------
    mean : jnp.ndarray
        Posterior state mean at the current time.
        For scalar ODEs in this module, this is shape ``(2,)`` and encodes
        ``[x, x_dot]``.
    cov : jnp.ndarray
        Posterior state covariance at the current time.
        For scalar ODEs in this module, this is shape ``(2, 2)``.

    Notes
    -----
    The dataclass is frozen, so updates are represented by constructing a new
    ``ODEFilterState`` (e.g., via :func:`ode_filter_advance`) rather than
    mutating fields in-place.
    """
    mean: jnp.ndarray
    cov: jnp.ndarray


def ode_filter_init(prior, x0, rhs0, cov_scale=1e-6):
    """Construct the initial EKF0 state from an ODE initial condition.

    The initializer maps the deterministic ODE initial value and its derivative
    into the latent prior state representation, then attaches an initial
    covariance controlled by ``cov_scale``.

    Parameters
    ----------
    prior : IWP2Prior
        Prior model providing:
        - ``initial_mean(x0, rhs0)`` for latent-state initialization.
        - ``initial_covariance(scale=...)`` for initial uncertainty.
    x0 : jnp.ndarray or float
        Initial ODE position at the start time.
    rhs0 : jnp.ndarray or float
        Initial derivative value, typically ``vector_field(x0, t0)``.
    cov_scale : float, optional
        Scalar multiplier for the initial covariance returned by the prior.
        Smaller values produce a more confident initial state.

    Returns
    -------
    ODEFilterState
        Initial filter state with prior-consistent mean and covariance.

    Notes
    -----
    This function does not evaluate the vector field; it assumes ``rhs0`` has
    already been computed by the caller.
    """
    mean0 = prior.initial_mean(x0, derivatives=(rhs0,))
    cov0 = prior.initial_covariance(scale=cov_scale)
    return ODEFilterState(mean=mean0, cov=cov0)


def ode_filter_advance(
        state,
        vector_field,
        dt,
        cfg: ODEFilterConfig):
    """Advance an ODE filter state by one time step.
    
    This is a convenience wrapper around :func:`ode_filter_step` that accepts
    and returns :class:`ODEFilterState` objects. Internally it performs:
    (1) prior prediction over ``dt`` and
    (2) Kalman-style update using the ODE residual at the step end time.
    
    Parameters
    ----------
    state : ODEFilterState
        Current filter state containing mean and covariance.
    vector_field : callable
        ODE right-hand side with signature ``vector_field(x, t)``.
        For scalar ODEs, returns a scalar derivative.
    dt : float
        Positive step size for the propagation interval.
    cfg : ODEFilterConfig
        Filter configuration containing prior dynamics, EKF mode, and
        measurement-noise variance.
    
    Returns
    -------
    ODEFilterState
        Updated filter state after one step over ``[t, t + dt]``.
    """
    mean_new, cov_new = ode_filter_step(
        state.mean, state.cov, vector_field, (0.0, dt), cfg
    )
    return ODEFilterState(mean=mean_new, cov=cov_new)

def ode_filter_step(
        mean: jnp.ndarray,
        cov: jnp.ndarray,
        vector_field,
        step_params: tuple,
        cfg: ODEFilterConfig) -> tuple:
    """
    Run one EKF-style ODE filter step.

    The observation residual is ``x_dot - vector_field(x, t_end)`` at
    ``t_end = t0 + dt``. Prediction uses ``cfg.prior`` and update uses
    ``cfg.measurement_noise`` with linearization mode ``cfg.ekf_mode``.

    Parameters
    ----------
    mean : jnp.ndarray
        Current state mean, shape ``(2,)`` for ``[x, x_dot]``.
    cov : jnp.ndarray
        Current state covariance, shape ``(2, 2)``.
    vector_field : callable
        ODE right-hand side ``vector_field(x, t)``.
    step_params : tuple[float, float]
        Pair ``(t0, dt)`` with step start time and step size.
    cfg : ODEFilterConfig
        Filter configuration containing prior dynamics, EKF mode, and
        measurement-noise variance.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Posterior mean and covariance at ``t0 + dt``.
    """
    t0, dt = step_params

    # prior prediction on [t_k, t_{k+1}]
    mean_pred, cov_pred = _predict_step(mean, cov, cfg.prior, dt)

    t_end = t0 + dt

    # posterior update using ODE residual at t_{k+1}
    mean_new, cov_new = _update_step(
        mean_pred, cov_pred, vector_field, t_end, cfg)

    return mean_new, cov_new

def _predict_step(
        mean: jnp.ndarray,
        cov: jnp.ndarray,
        prior: IWP2Prior, dt: float) -> tuple:
    """Predict mean and covariance one step via IWP-2 prior.
        Returns (mean_pred, cov_pred)."""
    f = prior.transition_matrix(dt)
    q = prior.process_covariance(dt)
    mean_pred = f @ mean
    cov_pred = f @ cov @ f.T + q
    return mean_pred, cov_pred

def _update_step(
        mean_pred: jnp.ndarray,
        cov_pred: jnp.ndarray,
        vector_field,
        t_end: float,
        cfg: ODEFilterConfig) -> tuple:
    """Kalman update using ODE residual at t_end. Returns (mean_new, cov_new)."""
    innovation, x_pred, state_dim = _compute_residual_terms(
        mean_pred, vector_field, t_end)

    h_jac = _observation_jacobian(
        vector_field, x_pred, t_end, state_dim, cfg.ekf_mode)

    mean_new, cov_new = _kalman_posterior(
        mean_pred, cov_pred, innovation, h_jac, cfg.measurement_noise)
    return mean_new, cov_new


def _observation_jacobian(
        vector_field,
        x,
        t,
        state_dim,
        ekf_mode: str) -> jnp.ndarray:
    """Return 1xN observation Jacobian for EKF0/EKF1 residual update."""
    h = jnp.zeros((state_dim,))
    if ekf_mode == "ekf0":
        h = h.at[0].set(0.0)
    elif ekf_mode == "ekf1":
        dfdx = jax.jacfwd(vector_field, 0)(x, t)
        dfdx0 = jnp.ravel(jnp.asarray(dfdx))[0]
        h = h.at[0].set(-dfdx0)
    else:
        raise ValueError("ekf_mode must be 'ekf0' or 'ekf1'")
    h = h.at[1].set(1.0)
    return h.reshape(1, -1)

def _compute_residual_terms(
        mean_pred,
        vector_field,
        t_end) -> tuple:
    """Return innovation, predicted position, and state dimension."""
    state_dim = mean_pred.shape[0]
    x_pred = mean_pred[0]
    xdot_pred = mean_pred[1]  # still second component by model definition

    h_pred = xdot_pred - vector_field(x_pred, t_end)
    innovation = -h_pred

    return innovation, x_pred, state_dim

def _kalman_posterior(
        mean_pred,
        cov_pred,
        innovation,
        h_jac,
        measurement_noise):
    """Return posterior mean/covariance from linearized Kalman update."""
    state_dim = mean_pred.shape[0]
    s = jnp.squeeze(h_jac @ cov_pred @ h_jac.T + measurement_noise)
    kalman_gain = (cov_pred @ h_jac.T) / s
    mean_new = mean_pred + jnp.ravel(kalman_gain * innovation)
    cov_new = (jnp.eye(state_dim) - kalman_gain @ h_jac) @ cov_pred
    return mean_new, cov_new

def ode_integrator_factory(
        prior: IWP2Prior,
        measurement_noise: float = 1e-6,
        ekf_mode: str = "ekf1"):
    """
    Build a one-step ODE integrator based on Algorithm 2 filtering.
    
    The returned callable has signature ``step(carry, vector_field, dt)``,
    where ``carry`` is ``(mean, cov)``. The step uses an internal
    :class:`ODEFilterConfig` with the provided prior, measurement noise,
    and EKF mode.
    
    Parameters
    ----------
    prior : IWP2Prior
        Prior dynamics model used for prediction.
    measurement_noise : float, optional
        Observation-noise variance used in the update.
    ekf_mode : str, optional
        Linearization mode for the observation update:
        ``"ekf0"`` or ``"ekf1"``.
    
    Returns
    -------
    callable
        Function ``(carry, vector_field, dt) -> (mean_new, cov_new)``.
    """
    if ekf_mode not in ("ekf0", "ekf1"):
        raise ValueError("ekf_mode must be 'ekf0' or 'ekf1'")

    cfg = ODEFilterConfig(
        prior=prior,
        ekf_mode=ekf_mode,
        measurement_noise=measurement_noise
    )

    def step(carry, vector_field, dt):
        mean, cov = carry
        mean_new, cov_new = ode_filter_step(
            mean, cov, vector_field, (0.0,dt), cfg)
        return mean_new, cov_new

    return step
