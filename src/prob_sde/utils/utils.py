"""
Utility helpers for the prob_sde package.

Exported objects
----------------
insert
    Insert an initial value at the beginning of an array along a given axis.
time_grid
    Build a uniform time grid with inclusive endpoints.
split_key
    Split a JAX PRNG key into a requested number of keys.
"""

import jax.numpy as jnp
from jax import random


def insert(
        trajectory: jnp.ndarray,
        axis: int,
        value: jnp.ndarray) -> jnp.ndarray:
    """
    Insert a value at the beginning of an array along the given axis.

    Parameters
    ----------
    trajectory : ndarray
        Array of shape (N, ...) to prepend to.
    axis : int
        Axis along which to insert (typically 0).
    value : ndarray
        Value to insert; must broadcast with trajectory shape except on axis.

    Returns
    -------
    ndarray
        New array with value prepended along axis, shape (N+1, ...).
    """
    return jnp.concatenate(
        [jnp.expand_dims(value, axis=axis), trajectory], axis=axis)


def time_grid(t0: float, t1: float, num_steps: int) -> jnp.ndarray:
    """
    Build a uniform time grid from t0 to t1 (inclusive endpoints).

    Parameters
    ----------
    t0 : float
        Start time.
    t1 : float
        End time.
    num_steps : int
        Number of intervals (number of points is num_steps + 1).

    Returns
    -------
    ndarray
        Array of shape (num_steps + 1,) with values from t0 to t1.
    """
    return jnp.linspace(t0, t1, num_steps + 1)


def split_key(key, num: int):
    """
    Split a JAX PRNG key into num keys.

    Parameters
    ----------
    key : jax.Array
        JAX random key.
    num : int
        Number of keys to produce.

    Returns
    -------
    ndarray
        Array of keys of shape (num,) (or a single key if num is 1).
    """
    return random.split(key, num)
