"""Scalar Benes SDE dynamics for benchmark scripts.

Exports
-------
drift
    Drift ``tanh(x)`` (time argument ignored).
diffusion
    Constant unit diffusion (time argument ignored).
"""
import jax.numpy as jnp


def drift(x, _t):
    """Benes drift: tanh(x)."""
    return jnp.tanh(x)


def diffusion(_x, _t):
    """Benes diffusion: constant one."""
    return jnp.array(1.0)