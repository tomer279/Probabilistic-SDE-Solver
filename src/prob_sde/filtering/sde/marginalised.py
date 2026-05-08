"""
Algorithm 4 Marginalised Gaussian SDE Filter.

This module implements Algorithm 4 from Le Fay, Sarkka, and Corenflos (2025)
using an augmented Gaussian state

    Y_k = [Y_k^(0), Y_k^(1), Y_k^(2), Y_k^(3)]^T
        = [X_t, Xdot_t, B_{k*delta,(k+1)*delta}, I_{k*delta,(k+1)*delta}]^T.

Exported objects
----------------
MarginalisedConfig
    Immutable configuration for Algorithm 4 runs.
solve_sde_marginalised
    Run Algorithm 4 on a uniform grid and return one sampled trajectory.
solve_sde_marginalised_batch
    Run Algorithm 4 for a batch of independent PRNG keys and return stacked trajectories.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from prob_sde.core.prior_models import IWP2Prior
from prob_sde.core.sde import SDESpec

from .position_sampling import PositionSamplingConfig, select_posterior_position


@dataclass(frozen=True)
class _RunState:
    """Bundle immutable data used during one Algorithm-4 solve."""
    key: object
    ts: jnp.ndarray
    x0: jnp.ndarray
    mean0: jnp.ndarray
    cov0: jnp.ndarray
    ctx: object


def _initialize_run(key, sde, config):
    """Prepare grid, initial state, and immutable context for scan."""
    prior = IWP2Prior(diffusion=config.prior_diffusion, measurement_noise=0.0)
    ctx = _StepContext(sde=sde, cfg=config, prior=prior)

    ts = config.time_grid(t0=0.0)
    x0 = _as_scalar(sde.x0, "sde.x0")
    mean0, cov0 = _initial_augmented_moments(x0, ts[0], ctx)

    return _RunState(
        key=key,
        ts=ts,
        x0=x0,
        mean0=mean0,
        cov0=cov0,
        ctx=ctx,
    )


def _pack_uncertainty(mean0, cov0, means, covs):
    """Prepend initial moments to per-step latent uncertainty arrays."""
    means_out = jnp.concatenate([mean0[None, :], means], axis=0)
    covs_out = jnp.concatenate([cov0[None, :, :], covs], axis=0)
    return means_out, covs_out


@dataclass(frozen=True)
class MarginalisedConfig:
    """Configuration for Algorithm 4.

    Instance variables
    ------------------
    delta : float
        Positive uniform step size.
    num_steps : int
        Number of integration steps.
    sample_posterior_position : bool
        If True, sample X_{t_{k+1}} from N(m[0], P[0,0]); otherwise use m[0].
    use_ekf1 : bool
        If True, use EKF1 Jacobian term for d(mu)/dx in the measurement model.
    variance_floor : float
        Non-negative lower bound for posterior variance during sampling.
    prior_diffusion : float
        Diffusion parameter for the IWP(2) prior used in A(delta), Q(delta).
    return_uncertainty : bool
        If True, also return latent means and covariances.

    Public methods
    --------------
    validate()
        Validate configuration values.
    time_grid(t0=0.0)
        Build the inclusive uniform grid [t0, ..., t0 + num_steps * delta].
    from_args(...)
        Convenience constructor mirroring dataclass fields.
    """

    delta: float
    num_steps: int
    sample_posterior_position: bool = True
    use_ekf1: bool = True
    variance_floor: float = 1e-12
    prior_diffusion: float = 1.0
    return_uncertainty: bool = False

    def validate(self) -> None:
        """Validate parameter constraints for Algorithm 4."""
        if self.delta <= 0.0:
            raise ValueError("delta must be positive.")
        if self.num_steps < 1:
            raise ValueError("num_steps must be at least 1.")
        if self.variance_floor < 0.0:
            raise ValueError("variance_floor must be non-negative.")
        if self.prior_diffusion <= 0.0:
            raise ValueError("prior_diffusion must be positive.")

    def time_grid(self, t0: float = 0.0) -> jnp.ndarray:
        """Return the inclusive time grid for this configuration."""
        t1 = t0 + self.num_steps * self.delta
        return jnp.linspace(t0, t1, self.num_steps + 1)


@dataclass(frozen=True)
class _StepContext:
    """Internal immutable context for one full Algorithm-4 run.

    Instance variables
    ------------------
    sde : SDESpec
        SDE model with drift and diffusion.
    cfg : MarginalisedConfig
        Runtime configuration.
    prior : IWP2Prior
        Order-2 IWP prior used to build A(delta) and Q(delta).

    Public methods
    --------------
    sigma(t, x_ref)
        Evaluate sigma(t); x_ref is ignored mathematically if sigma is time-only.
    drift_x(x, t)
        Evaluate mu(x, t) as scalar.
    """

    sde: SDESpec
    cfg: MarginalisedConfig
    prior: IWP2Prior

    def sigma(self, t: float) -> jnp.ndarray:
        """Return sigma(t) from diffusion(x, t), assuming time-only dependence."""
        return _as_scalar(self.sde.diffusion(0.0, t), "diffusion(x, t)")

    def drift_x(self, x: jnp.ndarray, t: float) -> jnp.ndarray:
        """Return scalar mu(x, t)."""
        return _as_scalar(self.sde.drift(x, t), "drift(x, t)")


def _as_scalar(value, name: str) -> jnp.ndarray:
    """Convert input to scalar array and validate scalar shape."""
    arr = jnp.asarray(value)
    if arr.size != 1:
        raise ValueError(name + " must be scalar for this Algorithm-4 implementation.")
    return jnp.squeeze(arr)


def _initial_augmented_moments(
        x_k: jnp.ndarray,
        t_k: float,
        ctx: _StepContext,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build Equation (16)-style initial moments for Y_k(t_k)."""
    delta = ctx.cfg.delta
    x_s = _as_scalar(x_k, "x_k")
    mu_k = ctx.drift_x(x_s, t_k)
    sigma_k = ctx.sigma(t_k)

    mean = jnp.array([x_s, mu_k, 0.0, 0.0])

    cov = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, (4.0 / delta) * (sigma_k**2), sigma_k, -(jnp.sqrt(6.0) / 2.0) * sigma_k],
            [0.0, sigma_k, delta, 0.0],
            [0.0, -(jnp.sqrt(6.0) / 2.0) * sigma_k, 0.0, delta / 2.0],
        ]
    )
    return mean, cov


def _predict_augmented(
        mean: jnp.ndarray,
        cov: jnp.ndarray,
        ctx: _StepContext,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Predict Y_k(t_{k+1}) from Y_k(t_k) using Abar and Qbar:
        Y_k(t_(k+1)) | Y_k(t_k) ~ N(Abar(delta) Y_k(t_k), Qbar(delta))
    where Abar and Qbar are given by:
        Abar = diag(A, 1, 1),  Qbar = Diag(Q, 0, 0)
    where A, Q are the transition matrix and process covariance related to IWP.
    """
    delta = ctx.cfg.delta
    a = ctx.prior.transition_matrix(delta)
    q = ctx.prior.process_covariance(delta)

    a_bar = jnp.eye(4)
    a_bar = a_bar.at[:2, :2].set(a)

    q_bar = jnp.zeros((4, 4))
    q_bar = q_bar.at[:2, :2].set(q)

    mean_pred = a_bar @ mean
    cov_pred = a_bar @ cov @ a_bar.T + q_bar
    return mean_pred, cov_pred


def _measurement_terms(
        mean_pred: jnp.ndarray,
        t_k: float,
        t_eval: float,
        ctx: _StepContext,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build measurement residual h and Jacobian row H at t_eval."""
    delta = ctx.cfg.delta
    x_pred = mean_pred[0]

    sigma_t = ctx.sigma(t_eval)
    mu_t = ctx.drift_x(x_pred, t_eval)

    c_b = sigma_t / delta
    c_i = (sigma_t * jnp.sqrt(6.0) / delta) * (2.0 * (t_eval - t_k) / delta - 1.0)

    # f_bar(Y, t) = mu(Y0, t) + sigma(t) * (Y2 + sqrt(6)/delta * (...) * Y3)
    f_bar = mu_t + c_b * mean_pred[2] + c_i * mean_pred[3]
    h_pred = mean_pred[1] - f_bar

    if ctx.cfg.use_ekf1:
        dmu_dx = _as_scalar(jax.jacfwd(ctx.sde.drift, argnums=0)(x_pred, t_eval), "dmu/dx")
        h0 = -dmu_dx
    else:
        h0 = 0.0

    h_row = jnp.array([[h0, 1.0, -c_b, -c_i]])
    return h_pred, h_row


def _ekf_update(
        mean_pred: jnp.ndarray,
        cov_pred: jnp.ndarray,
        h_pred: jnp.ndarray,
        h_row: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply one scalar EKF update with z=0 and R=0 (paper setting)."""
    innovation = -h_pred
    s_val = jnp.squeeze(h_row @ cov_pred @ h_row.T)
    k_gain = (cov_pred @ h_row.T) / s_val

    mean_new = mean_pred + jnp.ravel(k_gain * innovation)
    cov_new = (jnp.eye(cov_pred.shape[0]) - k_gain @ h_row) @ cov_pred
    cov_new = 0.5 * (cov_new + cov_new.T)
    return mean_new, cov_new


def _next_position(
    key: jax.Array,
    mean: jnp.ndarray,
    cov: jnp.ndarray,
    ctx: _StepContext,
) -> jnp.ndarray:
    """Select X_{t_{k+1}} via shared posterior sampler.

    Builds `PositionSamplingConfig` from `MarginalisedConfig` and delegates
    deterministic-or-sampled position extraction to
    `select_posterior_position`.
    """
    cfg = PositionSamplingConfig(
        sample_posterior_position=ctx.cfg.sample_posterior_position,
        variance_floor=ctx.cfg.variance_floor,
        require_key_when_sampling=False,
    )
    return select_posterior_position(mean=mean, cov=cov, cfg=cfg, key=key)


def solve_sde_marginalised(
        key: jax.Array,
        sde: SDESpec,
        config: MarginalisedConfig,
        ode_integrator=None,
):
    """Run Algorithm 4 marginalised Gaussian SDE filter.

    Parameters
    ----------
    key : jax.Array
        PRNG key for trajectory sampling.
    sde : SDESpec
        SDE specification. This implementation assumes scalar state and scalar
        drift/diffusion outputs.
    config : MarginalisedConfig
        Algorithm-4 configuration.
    ode_integrator : object, optional
        Included for API compatibility; unused by this implementation.

    Returns
    -------
    ts : jnp.ndarray
        Time grid of shape (num_steps + 1,).
    trajectory : jnp.ndarray
        Sampled path [X_{t0}, ..., X_{tK}] from Algorithm 4.
    uncertainty : tuple[jnp.ndarray, jnp.ndarray], optional
        Returned only when config.return_uncertainty=True:
        - means: shape (num_steps + 1, 4)
        - covs: shape (num_steps + 1, 4, 4)
    """
    _ = ode_integrator
    config.validate()

    run = _initialize_run(key, sde, config)
    (_, _), (xs, means, covs) = jax.lax.scan(
        lambda carry, t_k: _scan_step(carry, t_k, run.ctx),
        (run.key, run.x0),
        run.ts[:-1],
    )

    trajectory = jnp.concatenate([jnp.asarray([run.x0]), xs], axis=0)

    if not config.return_uncertainty:
        return run.ts, trajectory

    return run.ts, trajectory, _pack_uncertainty(run.mean0, run.cov0, means, covs)


def _scan_step(
        carry: tuple[jax.Array, jnp.ndarray],
        t_k: float,
        ctx: _StepContext):
    """Advance one stochastic step of Algorithm 4 inside `jax.lax.scan`.
    
    For the current sampled path value X_{t_k}, this function:
    1. Computes posterior augmented moments at t_{k+1} via `_posterior_at_step`.
    2. Draws (or deterministically selects) X_{t_{k+1}} from the posterior
       position marginal according to `ctx.cfg.sample_posterior_position`.
    3. Returns updated scan carry and emitted outputs for trajectory and
       optional uncertainty stacking.
    
    Parameters
    ----------
    carry : tuple[jax.Array, jnp.ndarray]
        `(key_k, x_k)` where `key_k` is the PRNG state and `x_k` is X_{t_k}.
    t_k : float
        Start time of the current interval.
    ctx : _StepContext
        Immutable run context containing model, configuration, and prior.
    
    Returns
    -------
    tuple
        `((key_next, x_k1), (x_k1, mean_k1, cov_k1))` where:
        - `key_next` is the propagated PRNG key,
        - `x_k1` is the next sampled path value,
        - `mean_k1`, `cov_k1` are posterior augmented moments at t_{k+1}.
    """
    key_k, x_k = carry
    key_next, key_sample = jax.random.split(key_k, 2)

    mean_new, cov_new = _posterior_at_step(x_k, t_k, ctx)
    x_k1 = _next_position(key_sample, mean_new, cov_new, ctx)

    return (key_next, x_k1), (x_k1, mean_new, cov_new)


def _posterior_at_step(
        x_k: jnp.ndarray,
        t_k: float,
        ctx: _StepContext
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute posterior moments for one Algorithm-4 interval update.
    
    This helper performs the deterministic Gaussian update on one interval
    [t_k, t_{k+1}] for the augmented state
        Y = [Y^(0), Y^(1), Y^(2), Y^(3)]^T by:

    1. Building Equation-(16)-style initial moments for Y_k(t_k) from x_k.
    2. Predicting to t_{k+1} with Abar(delta), Qbar(delta).
    3. Applying the scalar residual update
       z_k(t_{k+1}) = Y^(1)(t_{k+1}) - fbar_k(Y(t_{k+1}), t_{k+1}) = 0
       using EKF0 or EKF1 Jacobian according to ctx.cfg.use_ekf1.
    
    Parameters
    ----------
    x_k : jnp.ndarray
        Discrete path sample X_{t_k} used to initialize Y_k^(0)(t_k).
    t_k : float
        Interval start time.
    ctx : _StepContext
        Immutable run context containing the SDE model, configuration, and prior.
    
    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Posterior (mean, covariance) of the augmented state at t_{k+1}.
    """
    mean_init, cov_init = _initial_augmented_moments(x_k, t_k, ctx)
    mean_pred, cov_pred = _predict_augmented(mean_init, cov_init, ctx)

    t_k1 = t_k + ctx.cfg.delta
    h_pred, h_row = _measurement_terms(mean_pred, t_k, t_k1, ctx)
    return _ekf_update(mean_pred, cov_pred, h_pred, h_row)


def solve_sde_marginalised_batch(
        keys: jax.Array,
        sde: SDESpec,
        config: MarginalisedConfig,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run Algorithm 4 independently for a batch of PRNG keys.

    This is the batched counterpart to `solve_sde_marginalised`. It applies the
    same marginalised Gaussian SDE filter independently to each PRNG key in
    `keys` and stacks the resulting sampled trajectories.

    The function is intended for Monte Carlo aggregation workloads where many
    independent Algorithm-4 paths are needed on the same SDE, grid, and
    configuration. It is semantically equivalent to calling
    `solve_sde_marginalised(key_i, sde, config)` once per key and stacking the
    trajectory outputs, but it can be substantially faster because the per-key
    work is expressed as a single batched JAX program.

    Parameters
    ----------
    keys : jax.Array
        PRNG keys with leading shape `(num_samples, ...)`. Each key produces one
        independent trajectory.
    sde : SDESpec
        Scalar SDE specification consumed by Algorithm 4.
    config : MarginalisedConfig
        Algorithm-4 configuration shared by every trajectory. The grid, EKF mode,
        prior diffusion, and sampling policy are identical for all samples.

    Returns
    -------
    ts : jnp.ndarray
        Time grid of shape `(num_steps + 1,)`.
    trajectories : jnp.ndarray
        Sampled trajectories with shape `(num_samples, num_steps + 1)`, where
        `trajectories[i]` corresponds to `keys[i]`.

    Raises
    ------
    ValueError
        If the configuration is invalid, or if unsupported options such as
        batched uncertainty output are requested.

    Notes
    -----
    This function does not average or compute variances. Callers such as
    `solve_marginalised` are responsible for Monte Carlo aggregation.
    """
    config.validate()

    if config.return_uncertainty:
        raise ValueError("Batched marginalised solve does not return uncertainty yet.")

    def solve_one(key_i):
        _ts, traj = solve_sde_marginalised(key_i, sde, config)
        return traj

    ts = config.time_grid(t0=0.0)
    trajectories = jax.vmap(solve_one)(keys)
    return ts, trajectories
