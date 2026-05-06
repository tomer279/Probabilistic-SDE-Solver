"""
Algorithm 1 Brownian-path approximation utilities.

This module implements interval-wise Brownian approximations used by the SDE
filters. On each coarse interval [t_k, t_{k+1}] with delta = t_{k+1} - t_k,
the approximation is represented in local time tau in [0, delta] where
tau = t - t_k.

For the piecewise-parabolic family, each interval uses coefficients
(w0, w_delta, i_delta) and defines a local polynomial beta_k(tau). The global
piecewise approximation on [0, T] is obtained by concatenating interval pieces.

In this codebase, parabolic coefficients are primarily used through d/dt beta_k
inside random vector fields; therefore the derivative helper
`parabolic_dbeta_dt` is exported alongside the value evaluator.

Exported objects
----------------
piecewise_linear_brownian
    Return (get_coeffs, eval_fn) for piecewise linear Brownian approximation.
piecewise_parabolic_brownian
    Return (get_coeffs, eval_fn) for piecewise parabolic Brownian approximation.
parabolic_dbeta_dt
    Closed-form time derivative of one local parabolic piece beta_k on [0, delta].
parabolic_coeffs_from_fine_window
    Build (w0, w_delta, i_delta) for one coarse window from a fine Brownian path.
brownian_and_parabolic_coeffs
    Sample a fine Brownian path and return coarse increments, parabolic coefficients,
    and parabolic evaluator.

Algorithm mapping
-----------------
- Algorithm 1: implemented here by sampling per-interval coefficients beta_k
  and evaluating beta_k(t) on each interval.
  
Implementation notes
--------------------
- This module currently mixes JAX and NumPy operations in parts of the
  fine-path/parabolic-coefficient construction pipeline.
- The mixed backend is functional for eager execution, but can limit full
  JAX-transform compatibility (e.g., vmap/jit) in downstream benchmarks.

Future work
-----------
- Migrate fine Brownian path and bridge-area coefficient construction to
  fully JAX-native array operations.
- Revisit coefficient-list assembly for better vectorization and reduced
  Python-loop overhead in large Monte Carlo runs.
"""

import jax.numpy as jnp
from jax import random
import numpy as np


def piecewise_linear_brownian():
    """
    Factory for piecewise linear (Brownian bridge) approximation of Brownian motion.

    On each interval [0, delta], W_t is approximated by the linear interpolation
    between W_0=0 and W_delta ~ N(0, delta). The derivative dW/dt is constant
    on the interval, so the resulting random ODE has piecewise constant drift.

    Returns
    -------
    get_coeffs : callable
        (key, delta) -> coeffs. Samples increment and returns (w0, w_delta).
    eval_fn : callable
        (t, delta, *coeffs) -> value. JAX-differentiable in t; use jax.jacfwd for derivative.
    """
    # Algorithm 1 component: returns per-interval beta_k sampling/evaluation callables.
    def get_coeffs(key, delta):
        return _get_coeffs_linear(key, delta)

    def eval_fn(t, delta, *coeffs):
        return _eval_linear(t, delta, coeffs[0], coeffs[1])

    return get_coeffs, eval_fn

def _eval_linear(
        t: jnp.ndarray,
        delta: float,
        w0: jnp.ndarray,
        w_delta: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate piecewise linear Brownian bridge on [0, :math:`\\delta`]: 
    
    .. math::
        W_t = w_0 + (w_\\delta - w_0) \\frac{t}{\\delta}.

    Parameters
    ----------
    t : ndarray
        Time(s) in [0, delta].
    delta : float
        Step length.
    w0 : ndarray
        Value at 0 (typically 0 for standard BM).
    w_delta : ndarray
        Value at delta (increment).

    Returns
    -------
    ndarray
        Approximation W_t, same shape as t (or broadcast).
    """
    return w0 + (w_delta - w0) * (t / delta)


def _get_coeffs_linear(key, delta: float):
    """
    Sample Brownian increment W_delta ~ N(0, delta)
    for piecewise linear approximation.

    Parameters
    ----------
    key : jax.Array
        JAX random key.
    delta : float
        Step length (variance of increment).

    Returns
    -------
    tuple of ndarray
        (w0, w_delta). For standard BM from 0, w0=0 and w_delta is the sampled increment.
    """
    # Algorithm 1 step: sample coefficients defining beta_k on one interval.
    w_delta = random.normal(key, ()) * jnp.sqrt(delta)
    w0 = jnp.array(0.0)
    return (w0, w_delta)

def piecewise_parabolic_brownian():
    """
    Factory for piecewise parabolic Brownian approximation.

    On each interval [0, delta], returns callables:
    - get_coeffs(key, delta) -> (w0, w_delta, i_delta)
    - eval_fn(t, delta, *coeffs) -> beta(t)

    Returns
    -------
    tuple[callable, callable]
        (get_coeffs, eval_fn)
    """
    # Algorithm 1 component: returns per-interval beta_k sampling/evaluation callables.
    def get_coeffs(key, delta):
        return _get_coeffs_parabolic(key, delta)

    def eval_fn(t, delta, *coeffs):
        return _eval_parabolic(t, delta, coeffs[0], coeffs[1], coeffs[2])

    return get_coeffs, eval_fn


def parabolic_dbeta_dt(
    t: jnp.ndarray,
    delta: float,
    w0: jnp.ndarray,
    w_delta: jnp.ndarray,
    i_delta: jnp.ndarray,
) -> jnp.ndarray:
    """Return d/dt of one local parabolic Brownian approximation piece.

    This function differentiates the interval-local parabolic approximation
    beta_k(tau) on tau in [0, delta], where tau = t - t_k for a coarse interval
    [t_k, t_{k+1}]. It is intended for use in random vector fields of the form
    drift(x, t) + diffusion(x, t) * d/dt beta_k(t).

    The local parabolic representation is
        beta_k(t) = w0 + (w_delta - w0) * (t / delta)
                    + sqrt(6) * (t / delta) * ((t / delta) - 1) * i_delta,

    so the derivative is
        d/dt beta_k(t) = (w_delta - w0) / delta
                         + sqrt(6) * (2 * (t / delta) - 1) * i_delta / delta.

    Parameters
    ----------
    t : jnp.ndarray
        Local time tau in [0, delta].
    delta : float
        Interval length.
    w0 : jnp.ndarray
        Left-endpoint value beta_k(0).
    w_delta : jnp.ndarray
        Right-endpoint value beta_k(delta).
    i_delta : jnp.ndarray
        Parabolic curvature coefficient.

    Returns
    -------
    jnp.ndarray
        Time derivative d/dt beta_k(t) evaluated at local time t.
    """
    u = t / delta
    linear_term = (w_delta - w0) / delta
    curvature_term = jnp.sqrt(6.0) * (2.0 * u - 1.0) * i_delta / delta
    return linear_term + curvature_term

def _eval_parabolic(
        t: jnp.ndarray,
        delta: float,
        w0: jnp.ndarray,
        w_delta: jnp.ndarray,
        i_delta: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate piecewise parabolic Brownian approximation on [0, delta].

    The approximation is:
        beta(t) = w0 + (w_delta - w0) * u + sqrt(6) * u * (u - 1) * i_delta
    where u = t / delta.

    Parameters
    ----------
    t : ndarray
        Local time(s) in [0, delta].
    delta : float
        Step length.
    w0 : ndarray
        Value at 0 (typically 0 for standard BM).
    w_delta : ndarray
        Value at delta (Brownian increment over the interval).
    i_delta : ndarray
        Independent Gaussian coefficient with variance delta / 2.

    Returns
    -------
    ndarray
        Approximation beta(t), broadcast-compatible with inputs.
    """
    u = t / delta
    return w0 + (w_delta - w0) * u + jnp.sqrt(6.0) * u * (u - 1.0) * i_delta


def _get_coeffs_parabolic(key, delta: float):
    """
    Sample coefficients for piecewise parabolic Brownian approximation.

    Parameters
    ----------
    key : jax.Array
        JAX random key.
    delta : float
        Step length.

    Returns
    -------
    tuple of ndarray
        (w0, w_delta, i_delta), where:
        - w0 = 0 for standard Brownian motion,
        - w_delta ~ N(0, delta),
        - i_delta ~ N(0, delta / 2),
        and w_delta, i_delta are independent.
    """
    # Algorithm 1 step: sample coefficients defining beta_k on one interval.
    key_w, key_i = random.split(key, 2)
    w0 = jnp.array(0.0)
    w_delta = random.normal(key_w, ()) * jnp.sqrt(delta)
    i_delta = random.normal(key_i, ()) * jnp.sqrt(delta / 2.0)
    return (w0, w_delta, i_delta)

def parabolic_coeffs_from_fine_window(
        times,
        w,
        left_idx,
        right_idx,
        delta_coarse):
    """
    Parabolic Algorithm-1 coefficients matched to a fine Brownian segment.

    For the time window ``[times[left_idx], times[right_idx]]`` with values
    ``w``, compute ``(w0, w_delta, i_delta)`` compatible with :func:`_eval_parabolic`
    and :func:`piecewise_parabolic_brownian`. The increment ``w_delta`` is the
    path increment over the window; ``i_delta`` is inferred from the bridge-area
    map so the parabolic path matches that segment (not an independent Gaussian).
    
    Parameters
    ----------
    times : ndarray
        Fine time grid (monotone), length at least 2.
    w : ndarray
        Cumulative Brownian path on ``times``, with ``w[0] = 0``.
    left_idx : int
        Index of the coarse interval start on the fine grid.
    right_idx : int
        Index of the coarse interval end on the fine grid.
    delta_coarse : float
        Coarse step length (``times[right_idx] - times[left_idx]`` up to rounding).
    
    Returns
    -------
    tuple
        ``(w0, w_delta, i_delta)`` for one local interval ``[0, delta_coarse]``.
    """
    interval_times = times[left_idx : right_idx + 1]
    interval_values = w[left_idx : right_idx + 1]
    left_value = float(w[left_idx])
    right_value = float(w[right_idx])
    area = _interval_bridge_area(interval_times, interval_values, left_value, right_value)
    i_delta = jnp.asarray(-(np.sqrt(6.0) / delta_coarse) * area)
    w0 = jnp.array(0.0)
    w_delta = jnp.asarray(right_value - left_value)
    return (w0, w_delta, i_delta)

def _interval_bridge_area(
        interval_times,
        interval_values,
        left_value,
        right_value):
    """
    Integral of the Brownian-bridge component over one interval.
    
    Subtracts the chord (linear interpolation between endpoints) from the path
    values to obtain the centered bridge, then integrates that residual with
    respect to time using the trapezoidal rule. This scalar is used together with
    the coarse step length to recover ``i_delta`` for a parabolic ``beta`` that
    matches a given fine path segment (see :func:`parabolic_coeffs_from_fine_window`).
    
    Parameters
    ----------
    interval_times : ndarray
        Strictly increasing times spanning one coarse interval (includes both ends).
    interval_values : ndarray
        Path values on ``interval_times``.
    left_value : float
        Path value at the left endpoint (same as ``interval_values[0]``).
    right_value : float
        Path value at the right endpoint (same as ``interval_values[-1]``).
    
    Returns
    -------
    float
        Approximate integral of ``(W_t - chord(t))`` over the interval.
    """
    delta = interval_times[-1] - interval_times[0]
    u = (interval_times - interval_times[0]) / delta
    linear_part = left_value + (right_value - left_value) * u
    bridge_part = interval_values - linear_part
    return np.trapezoid(bridge_part, interval_times)

def brownian_and_parabolic_coeffs(
        key,
        t_final: float,
        delta_fine: float,
        delta_coarse: float,
        as_dict: bool = True):
    """Sample Brownian motion on a fine grid and parabolic coeffs per coarse step.
    
    Draws independent Gaussian increments with variance ``delta_fine`` on
    ``[0, t_final]``, forming a reference path ``W``. For each coarse interval
    of length ``delta_coarse``, computes parabolic Algorithm-1 coefficients
    ``(w0, w_delta, i_delta)`` matched to that path segment via bridge area (see
    :func:`parabolic_coeffs_from_fine_window`). Also returns coarse increments
    obtained by summing fine increments over each block (consistent with EM when
    the coarse step uses those sums). ``eval_parabolic`` is the evaluator from
    :func:`piecewise_parabolic_brownian` for local ``beta(t)``.
    
    Parameters
    ----------
    key : jax.Array
        PRNG key for fine increments.
    t_final : float
        Horizon ``T``; must be divisible by ``delta_fine`` in the grid sense below.
    delta_fine : float
        Fine time step for the sampled Brownian path.
    delta_coarse : float
        Coarse time step; the fine grid must refine each coarse interval evenly.
    
    Returns
    -------
    times : ndarray
        Fine grid ``0, delta_fine, ..., t_final``.
    w : ndarray
        Cumulative ``W(t)`` on ``times``, starting at ``0``.
    dw_fine : jax.Array
        Fine increments, shape ``(n_ref,)``.
    dw_coarse : jax.Array
        Block sums of ``dw_fine``, shape ``(num_steps,)``.
    coeffs_list : list
        Length ``num_steps``; each element is ``(w0, w_delta, i_delta)``.
    eval_parabolic : callable
        ``(t, delta_coarse, *coeffs) -> beta(t)``.
    block_size : int
        Number of fine steps per coarse step.
    """
    n_ref, times, w, dw_fine = _sample_fine_brownian_path(key, t_final, delta_fine)
    num_steps, block_size = _coarse_grid_layout(n_ref, t_final, delta_coarse)
    coeffs_list = _parabolic_coeffs_list(
        times, w, num_steps, block_size, delta_coarse)
    dw_coarse = _coarse_increments_from_fine(dw_fine, num_steps, block_size)
    _, eval_parabolic = piecewise_parabolic_brownian()


    result = {
        "times": times,
        "w": w,
        "dw_fine": dw_fine,
        "dw_coarse": dw_coarse,
        "coeffs_list": coeffs_list,
        "eval_parabolic": eval_parabolic,
        "block_size": block_size,
    }

    if as_dict:
        return result

    # Backward-compatible legacy return
    return (
        result["times"],
        result["w"],
        result["dw_fine"],
        result["dw_coarse"],
        result["coeffs_list"],
        result["eval_parabolic"],
        result["block_size"],
    )


def _sample_fine_brownian_path(key, t_final: float, delta_fine: float):
    """Sample Gaussian increments on a uniform grid and build cumulative ``W``."""
    n_ref = int(round(t_final / delta_fine))
    if not np.isclose(n_ref * delta_fine, t_final):
        raise ValueError("t_final must be an integer multiple of delta_fine.")
    dw_fine = random.normal(key, (n_ref,)) * jnp.sqrt(delta_fine)
    dw_np = np.asarray(dw_fine)
    times = np.linspace(0.0, t_final, n_ref + 1)
    w = np.concatenate([[0.0], np.cumsum(dw_np)])
    return n_ref, times, w, dw_fine

def _coarse_grid_layout(n_ref: int, t_final: float, delta_coarse: float):
    """Return coarse step count and fine steps per coarse interval."""
    num_steps = int(round(t_final / delta_coarse))
    block_size = n_ref // num_steps
    if num_steps * block_size != n_ref:
        raise ValueError("Fine grid must align with coarse steps.")
    return num_steps, block_size

def _parabolic_coeffs_list(
        times,
        w,
        num_steps: int,
        block_size: int,
        delta_coarse: float):
    """Build parabolic coefficient tuples for each coarse interval."""
    coeffs_list = []
    for k in range(num_steps):
        left_idx = k * block_size
        right_idx = (k + 1) * block_size
        coeffs_list.append(
            parabolic_coeffs_from_fine_window(
                times, w, left_idx, right_idx, delta_coarse)
        )
    return coeffs_list

def _coarse_increments_from_fine(dw_fine, num_steps: int, block_size: int):
    """Sum fine increments into coarse-step increments."""
    reshaped = jnp.reshape(dw_fine, (num_steps, block_size))
    return jnp.sum(reshaped, axis=1)
