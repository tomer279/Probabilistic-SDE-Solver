"""
Single-step sanity check: pure Brownian (drift 0, diffusion 1).

True ODE on [0, delta]:  dX/dt = beta'(t),  X(0) = 0
  =>  X(delta) - X(0) = beta(delta) - beta(0) = w_delta  (with w0 = 0).

Compare w_delta to the Gaussian SDE filter's posterior position increment.
"""

import jax
import jax.numpy as jnp

from prob_sde import brownian_and_parabolic_coeffs
from prob_sde import SDESpec, IWP2Prior
from prob_sde.filtering.sde.gaussian_sde_filter import (
    GaussianSDEFilter, GaussianSDEFilterConfig
)


def drift_zero(_x, _t):
    return jnp.array(0.0)


def diffusion_one(_x, _t):
    return jnp.array(1.0)


def integration_increment_parabolic(delta, coeffs):
    """Integral of beta'(t) on [0, delta] == w_delta for the parabolic model."""
    _w0, w_delta, _i_delta = coeffs
    return jnp.asarray(w_delta)


def single_step_increment_error(
    key,
    delta_coarse: float = 0.01,
    measurement_noise: float = 1e-6,
    variance_floor: float = 1e-12,
    initial_cov_scale: float = 1e-8,
):
    delta_fine = delta_coarse * delta_coarse
    t_final = delta_coarse

    _times, _w, _dw_fine, dw_coarse, coeffs_list, _eval_fn, _block = (
        brownian_and_parabolic_coeffs(key, t_final, delta_fine, delta_coarse)
    )

    coeffs0 = coeffs_list[0]
    w_delta_true = integration_increment_parabolic(delta_coarse, coeffs0)

    sde = SDESpec.from_args(
        drift_zero,
        diffusion_one,
        jnp.array(0.0),
        bm_factory=None,
    )
    prior = IWP2Prior(1.0)
    cfg = GaussianSDEFilterConfig(
        measurement_noise=measurement_noise,
        sample_posterior_position=False,
        variance_floor=variance_floor,
        initial_cov_scale=initial_cov_scale,
        return_beta_coeffs=False,
    )
    solver = GaussianSDEFilter.from_parabolic(prior, sde, cfg)

    state0 = solver.initialize(jnp.array(0.0), t0=0.0)
    key_step = jax.random.fold_in(key, 4242)
    state1 = solver.step_with_coeffs(key_step, state0, delta_coarse, coeffs0)

    dx_mean = state1.mean[0] - state0.mean[0]
    dx_state_x = state1.x - state0.x

    err_mean = float(dx_mean - w_delta_true)
    err_x = float(dx_state_x - w_delta_true)

    return {
        "delta": delta_coarse,
        "w_delta_true": float(w_delta_true),
        "dx_posterior_mean_0": float(dx_mean),
        "dx_state_x": float(dx_state_x),
        "abs_err_mean_vs_w_delta": abs(err_mean),
        "abs_err_x_vs_w_delta": abs(err_x),
    }


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    out = single_step_increment_error(key)
    for k, v in out.items():
        print(k, "=", v)