"""
State-space prior models for probabilistic ODE filtering.

This module provides prior dynamics used by the Gaussian ODE filter in
:mod:`prob_sde.ode_filter`. Priors define linear state transitions, process-noise
covariances, and initialization helpers for latent filtering states.

Exported objects
----------------
IWPPrior
    Generic scalar Integrated Wiener Process prior of configurable order.
IWP2Prior
    Backward-compatible wrapper for ``IWPPrior(order=2)``.
IWP3Prior
    Backward-compatible wrapper for ``IWPPrior(order=3)``.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import factorial

class IWPPrior:
    """
    Generic Integrated Wiener Process prior for scalar ODE filtering.
    This class defines a linear-Gaussian state-space prior whose latent state is
    ordered as ``[x, x_dot, x_ddot, ..., x^(order-1)]`` with dimension ``order``.
    The highest derivative is driven by white noise.

    Attributes
    ----------
    order : int
        Latent state dimension.
    diffusion : float
        Diffusion scale of the driving white noise.
    measurement_noise : float
        Observation-noise variance used by the ODE filter update.

    Methods
    -------
    state_dimension()
        Return latent state dimension.
    transition_matrix(dt)
        Return transition matrix ``F(dt)``.
    process_covariance(dt)
        Return process covariance ``Q(dt)``.
    initial_mean(x0, derivatives=())
        Build initial latent mean from value and derivatives.
    initial_mean_from_vector_field(vector_field, x0, t0)
        Build initial latent mean by differentiating the ODE vector field.
    initial_covariance(scale=1.0)
        Build diagonal initial covariance.
    """

    def __init__(
        self,
        order: int,
        diffusion: float = 1.0,
        measurement_noise: float = 1e-6,
    ):
        """
        Initialize an Integrated Wiener Process prior.

        Parameters
        ----------
        order : int
            Latent state dimension; must be at least 1.
        diffusion : float, optional
            Diffusion coefficient that scales the driving white noise.
        measurement_noise : float, optional
            Observation-noise variance used by the ODE filter update.

        Raises
        ------
        ValueError
            If ``order < 1``.
        """
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = int(order)
        self.diffusion = diffusion
        self.measurement_noise = measurement_noise

    def state_dimension(self) -> int:
        """
        Return latent state dimension.

        Returns
        -------
        int
            Latent state dimension.
        """
        return self.order

    def transition_matrix(self, dt: float) -> jnp.ndarray:
        """
        State transition matrix F for step dt: mean_new = F @ mean.
        Closed form:
        ``F[i, j] = 1(j >= i) * dt^(j-i) / (j-i)!``
        
        Parameters
        ----------
        dt : float
            Step size.
        
        Returns
        -------
        ndarray
            Shape ``(order, order)`` upper-triangular transition matrix.
        """
        n = self.order

        # Create column vector (i) and row vector (j) of (0,...,n-1) in matrix form
        i = jnp.arange(n)[:, None]
        j = jnp.arange(n)[None, :]

        # p_(ij) = j-i
        p = j - i
        upper = p >= 0 # All upper diagonal elements (including) are True.
        p_safe = jnp.where(upper,p, 0) # Zero values for all lower diagonal elements
        transition_matrix = jnp.where(
            upper,
            (dt ** p_safe / factorial(p_safe)),
            0.0
        )
        return transition_matrix

    def process_covariance(self, dt: float) -> jnp.ndarray:
        """
        Compute the discrete-time process covariance ``Q(dt)``.

        Parameters
        ----------
        dt : float
            Integration step size.

        Returns
        -------
        jnp.ndarray
            Process covariance matrix of shape ``(order, order)``.

        Notes
        -----
        For indices ``i, j`` in ``{0, ..., order-1}`` with ``n = order``:
        ``Q[i, j] = diffusion^2 * dt^(2n-1-i-j) /
        ((2n-1-i-j) * (n-1-i)! * (n-1-j)!)``.
        """
        n = self.order
        sigma2 = self.diffusion ** 2

        i = jnp.arange(n)[:, None]
        j = jnp.arange(n)[None, :]

        exponent = 2 * n - 1 - i - j
        fi = factorial(n-1-i)
        fj = factorial(n-1-j)

        denom = exponent * fi * fj
        covariance_matrix = sigma2 * (dt ** exponent) / denom
        return covariance_matrix

    def initial_mean(
            self,
            x0: float,
            derivatives: tuple[float, ...] = ()) -> jnp.ndarray:
        """
        Build the initial latent mean from position and derivatives.

        Parameters
        ----------
        x0 : float
            Initial ODE state value.
        derivatives : tuple[float, ...], optional
            Derivatives at the initial time, ordered as
            ``(x_dot, x_ddot, ..., x^(order-1))``.
            If empty, all missing derivatives are set to zero.
            If provided, length must be exactly ``order - 1``.

        Returns
        -------
        jnp.ndarray
            Initial mean vector of shape ``(order,)``.

        Raises
        ------
        ValueError
            If ``len(derivatives)`` is neither ``0`` nor ``order - 1``.
        """
        expected = self.order - 1
        if len(derivatives) not in (0, expected):
            raise ValueError(
                f"Expected derivatives length 0 or {expected}, got {len(derivatives)}"
            )
        if len(derivatives) == 0:
            derivatives = tuple(0.0 for _ in range(expected))
        values = [jnp.asarray(x0)]
        values.extend(jnp.asarray(val) for val in derivatives)
        return jnp.array(values)

    def initial_mean_from_vector_field(
            self,
            vector_field,
            x0: float,
            t0: float) -> jnp.ndarray:
        """
        Build the initial mean by differentiating the scalar vector field.

        This computes
        ``[x0, x'(t0), x''(t0), ..., x^(order-1)(t0)]``
        for scalar ODEs ``x' = f(x, t)`` via recursive total derivatives.

        Parameters
        ----------
        vector_field : callable
            Scalar ODE right-hand side with signature ``vector_field(x, t)``.
            Must return a scalar value compatible with ``jax.grad``.
        x0 : float
            Initial ODE state value.
        t0 : float
            Initial time.

        Returns
        -------
        jnp.ndarray
            Initial mean vector of shape ``(order,)``.

        Notes
        -----
        Recursion:
        ``g1(x,t) = f(x,t)``
        ``g_{k+1}(x,t) = (dg_k/dx)(x,t) * f(x,t) + (dg_k/dt)(x,t)``.
        """
        x0_arr = jnp.asarray(x0)
        t0_arr = jnp.asarray(t0)
        derivatives = []
        g = vector_field

        for _ in range(1, self.order):
            g_val = g(x0_arr, t0_arr)
            derivatives.append(g_val)
            dg_dx = jax.grad(g, argnums=0)
            dg_dt = jax.grad(g, argnums=1)

            def next_g(x, t, dg_dx=dg_dx, dg_dt=dg_dt, vector_field=vector_field):
                """Return total derivative d/dt g(x(t), t)."""
                return dg_dx(x, t) * vector_field(x, t) + dg_dt(x, t)

            g = next_g

        return jnp.array([x0_arr, *derivatives])

    def initial_covariance(self, scale: float = 0.0) -> jnp.ndarray:
        """
        Build a diagonal initial covariance matrix.

        Parameters
        ----------
        scale : float, optional
            Variance level for each latent component.

        Returns
        -------
        jnp.ndarray
            Diagonal covariance matrix of shape ``(order, order)``.

        Notes
        -----
        Setting ``scale=0.0`` yields deterministic initialization.
        """
        return scale * jnp.eye(self.order)


class IWP2Prior(IWPPrior):
    """
    Backward-compatible order-2 IWP prior.

    Public methods inherited from ``IWPPrior``.
    """
    def __init__(self, diffusion: float = 1.0, measurement_noise: float = 1e-6):
        """
        Initialize an order-2 integrated Wiener process prior.

        Parameters
        ----------
        diffusion : float, optional
            Diffusion coefficient that scales the driving white noise.

        measurement_noise : float, optional
            Observation-noise variance used by the ODE filter update.
        """
        super().__init__(
            order=2,
            diffusion=diffusion,
            measurement_noise=measurement_noise,
        )


    def mean_from_position_velocity(
            self, x0: float, x0_dot: float = 0.0) -> jnp.ndarray:
        """
        Build ``[x0, x0_dot]`` for convenience.

        Parameters
        ----------
        x0 : float
            Initial position.
        x0_dot : float, optional
            Initial first derivative.

        Returns
        -------
        jnp.ndarray
            Initial mean vector of shape ``(2,)``.
        """
        return super().initial_mean(x0=x0, derivatives=(x0_dot,))


class IWP3Prior(IWPPrior):
    """
    Backward-compatible order-3 IWP prior.

    Public methods inherited from ``IWPPrior``.
    """

    def __init__(self, diffusion: float = 1.0, measurement_noise: float = 1e-6):
        """
        Parameters
        ----------
        diffusion : float
            Diffusion coefficient scaling the driving white noise.
        measurement_noise : float
            Observation noise variance for the ODE filter.
        """
        super().__init__(
            order=3,
            diffusion=diffusion,
            measurement_noise=measurement_noise,
        )


    def mean_from_position_velocity_acceleration(
        self,
        x0: float,
        x0_dot: float = 0.0,
        x0_ddot: float = 0.0,
    ) -> jnp.ndarray:
        """
        Build ``[x0, x0_dot, x0_ddot]`` for convenience.

        Parameters
        ----------
        x0 : float
            Initial position.
        x0_dot : float, optional
            Initial first derivative.
        x0_ddot : float, optional
            Initial second derivative.

        Returns
        -------
        jnp.ndarray
            Initial mean vector of shape ``(3,)``.
        """
        return super().initial_mean(x0=x0, derivatives=(x0_dot, x0_ddot))
