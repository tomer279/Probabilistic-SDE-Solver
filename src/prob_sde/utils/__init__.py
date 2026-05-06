"""General package utility helpers.

Exports:
- insert: insert values into an array at selected indices.
- time_grid: construct a regular time grid.
- split_key: split a JAX PRNG key into multiple keys.
"""

from .utils import insert, split_key, time_grid

__all__ = ["insert", "time_grid", "split_key"]