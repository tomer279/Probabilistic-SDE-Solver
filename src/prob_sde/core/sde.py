"""
Core SDE specification container used by integrators and simulators.

Exported objects
----------------
SDESpec
    Immutable dataclass that bundles drift, diffusion, initial state, and a
    Brownian-approximation factory. The class also provides from_args() as a
    convenience constructor.

Conventions
-----------
- drift(x, t) returns the deterministic drift term with shape compatible
  with x.
- diffusion(t) returns the diffusion term (scalar/vector/matrix).
- x0 is the initial state at t=0.
- bm_factory() returns (get_coeffs, eval_fn), where:
  - get_coeffs(key, delta) samples coefficients for one interval.
  - eval_fn(t, delta, *coeffs) evaluates the Brownian approximation on
    [0, delta].

Algorithm mapping
-----------------
- This module defines the interface used to inject Algorithm-1 components
  through bm_factory().
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SDESpec:
    """
    Specification of an SDE: drift, diffusion, initial state, and Brownian-motion factory.

    Attributes
    ----------
    drift : callable
        (x, t) -> drift vector, same shape as x.
    diffusion : callable
        (x, t) -> diffusion matrix or vector; for scalar SDE, (x, t) -> scalar.
    x0 : ndarray
        Initial state.
    bm_factory : callable
        () -> (get_coeffs, eval_fn). get_coeffs(key, delta) returns coeffs;
        eval_fn(t, delta, *coeffs) is the BM approximation on [0, delta].

    Methods
    -------
    from_args : classmethod
        Build an SDESpec from four positional arguments (drift, diffusion, x0, bm_factory).
    """

    drift: callable
    diffusion: callable
    x0: object
    bm_factory: callable

    @classmethod
    def from_args(cls, drift, diffusion, x0, bm_factory):
        """
        Build an SDESpec from the four SDE components.

        Parameters
        ----------
        drift : callable
            (x, t) -> drift.
        diffusion : callable
            (t) -> diffusion.
        x0 : ndarray
            Initial state.
        bm_factory : callable
            () -> (get_coeffs, eval_fn).

        Returns
        -------
        SDESpec
            Instance with the given drift, diffusion, x0, and bm_factory.
        """
        return cls(drift=drift, diffusion=diffusion, x0=x0, bm_factory=bm_factory)
