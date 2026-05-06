"""
EKF0 ODE filter demo: prediction vs update step (Figure 38.1-style).

Demonstrates the ODE filter (Algorithm 2) with IWP-2 prior in a 2×2 layout:
- Left column: prediction step (prior only over [0, t1]).
- Right column: update step (multiple EKF0 steps; filter mean trajectory).
Rows: x(t) and x'(t) with  solution, samples, mean, and ±2σ bands.
The update column shows the filter mean (and ±2σ) at each step so the mean
tracks the true solution after corrections.

Exported / run as script
------------------------
main()
    Run demo and save figure (or print fallback).
"""
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import jax
import jax.numpy as jnp
from prob_sde.ode_filter import (
    ODEFilterState,
    ode_filter_init,
    ode_filter_advance,
    ode_filter_step,
    _update_step,
)
from prob_sde.prior_models import IWP2Prior
from prob_sde.prior_models import IWP3Prior

class _IntegrationConfig:
    """Holds dt, t_grid, num_samples, num_steps for the demo integration."""

    def __init__(self, dt, t_grid, num_samples, num_steps):
        self.dt = dt
        self.t_grid = t_grid
        self.num_samples = num_samples
        self.num_steps = num_steps

    def num_grid_points(self):
        """Return the number of grid points in t_grid."""
        return len(self.t_grid)

    def dt_sub(self):
        """Return the sub-step size (dt / num_steps)."""
        return self.dt / self.num_steps

def _vector_field(x, t):
    """ODE RHS: dx/dt = (-x**3)/2 (time-dependent only)."""
    return -(x**3)/2


def _true_solution(t, x0=1.0):
    """Exact x(t) for dx/dt = (-x**3)/2, x(0)=x0. Solution: 1/sqrt(t+1)."""
    return 1/jnp.sqrt(t+1)


def _filter_trajectory(state0, prior, dt, num_steps):
    """
    Run num_steps EKF0 advances over [0, dt]; return step times, means, stds.

    Uses global time in each step so that vector_field(x, t) is evaluated at
    the correct t. Returns step_times (num_steps+1,), filter_mean (num_steps+1, 2),
    filter_std (num_steps+1, 2).
    """
    dt_sub = dt / num_steps
    step_times = [0.0]
    step_means = [state0.mean]
    step_stds = [jnp.sqrt(jnp.diag(state0.cov))]
    state = state0
    for _ in range(num_steps):
        t_current = step_times[-1]
        mean_new, cov_new = ode_filter_step(
            state.mean, state.cov, _vector_field, (t_current, dt_sub), prior
        )
        state = ODEFilterState(mean=mean_new, cov=cov_new)
        step_times.append(t_current + dt_sub)
        step_means.append(state.mean)
        step_stds.append(jnp.sqrt(jnp.diag(state.cov)))
    return (
        jnp.array(step_times),
        jnp.stack(step_means),
        jnp.stack(step_stds),
    )

def ensemble_filter_trajectory(key, state0, prior, config):
    """
    Run ensemble EKF0 trajectory over config.dt with config.num_steps.
    """
    dt_sub = config.dt / config.num_steps
    eps = 1e-8
    dynamics = _ensemble_dynamics(prior, dt_sub, eps)
    keys = jax.random.split(key, config.num_steps + 1)
    ensemble = jax.random.multivariate_normal(
        keys[0], state0.mean, state0.cov, (config.num_samples,)
    )
    storage = _init_trajectory_storage(state0, ensemble)
    for step_index, step_key in enumerate(keys[1:]):
        t_end = (step_index + 1) * dt_sub
        ensemble, post_mean, post_cov = _ensemble_single_step(
            step_key, ensemble, dynamics, t_end
        )
        storage["times"].append(t_end)
        storage["means"].append(post_mean)
        storage["stds"].append(jnp.sqrt(jnp.diag(post_cov)))
        storage["ensembles"].append(ensemble)
    return (
        jnp.array(storage["times"]),
        jnp.stack(storage["means"]),
        jnp.stack(storage["stds"]),
        jnp.stack(storage["ensembles"]),
    )


def _sample_mean_cov(samples, eps):
    """Return sample mean/covariance with diagonal jitter."""
    mean = jnp.mean(samples, axis=0)
    centered = samples - mean
    cov = (centered.T @ centered) / max(samples.shape[0] - 1, 1)
    cov = cov + eps * jnp.eye(samples.shape[1])
    return mean, cov


def _ensemble_single_step(step_key, ensemble, dynamics, t_end):
    """
    Run one predict-update-resample step.
    Returns (new_ensemble, post_mean, post_cov).
    """
    member_keys, resample_key = jax.random.split(step_key, 2)
    sample_keys = jax.random.split(member_keys, ensemble.shape[0])
    pred_ensemble = jnp.array(
        [
            dynamics["f_mat"] @ ensemble[i]
            + jax.random.multivariate_normal(
                sample_keys[i],
                jnp.zeros(ensemble.shape[1]),
                dynamics["q_mat"],
            )
            for i in range(ensemble.shape[0])
        ]
    )
    mean_pred, cov_pred = _sample_mean_cov(pred_ensemble, dynamics["eps"])
    post_mean, post_cov = _update_step(
        mean_pred,
        cov_pred,
        _vector_field,
        dynamics["measurement_noise"],
        t_end,
    )
    new_ensemble = jax.random.multivariate_normal(
        resample_key,
        post_mean,
        post_cov + dynamics["eps"] * jnp.eye(ensemble.shape[1]),
        (ensemble.shape[0],),
    )
    return new_ensemble, post_mean, post_cov

def _ensemble_dynamics(prior, dt_sub, eps):
    """Return reusable dynamics terms for one sub-step."""
    return {
        "f_mat": prior.transition_matrix(dt_sub),
        "q_mat": prior.process_covariance(dt_sub),
        "measurement_noise": getattr(prior, "measurement_noise", 1e-6),
        "eps": eps,
    }

def _init_trajectory_storage(state0, ensemble0):
    """Initialize trajectory storage lists."""
    return {
        "times": [0.0],
        "means": [state0.mean],
        "stds": [jnp.sqrt(jnp.diag(state0.cov))],
        "ensembles": [ensemble0],
    }

def compute_prior_data(key, state0, prior, config):
    """
    Compute prior samples and prior mean/std along config.t_grid.

    Returns (pred_samples, pred_mean, pred_std).
    """
    pred_samples = _sample_prior_trajectories(
        key, state0, prior, config.t_grid, config.num_samples
    )
    pred_mean, pred_std = _prior_mean_cov_along_grid(
        state0.mean, state0.cov, prior, config.t_grid
    )
    return (pred_samples, pred_mean, pred_std)


def _sample_prior_trajectories(key, state0, prior, t_grid, num_samples=30):
    """
    Sample num_samples prior trajectories over t_grid using IWP-2 dynamics.

    For each trajectory: draw (x0, x'0) ~ N(mean0, cov0), then propagate
    with state_next = F @ state + N(0, Q) per sub-step. Returns shape
    (num_samples, len(t_grid), 2) for (x, x').
    """
    mean0 = state0.mean
    cov0 = state0.cov
    keys = jax.random.split(key, num_samples)
    n = len(t_grid)
    dt = (t_grid[-1] - t_grid[0]) / max(1, n - 1)

    def one_trajectory(key_i):
        state = jax.random.multivariate_normal(key_i, mean0, cov0)
        states = [state]
        sub_keys = jax.random.split(key_i, n - 1)
        n_state = mean0.shape[0]
        for i in range(n - 1):
            f = prior.transition_matrix(dt)
            q = prior.process_covariance(dt)
            state = f @ state + jax.random.multivariate_normal(
                sub_keys[i], jnp.zeros(n_state), q
            )
            states.append(state)
        return jnp.stack(states)

    return jax.vmap(one_trajectory)(keys)

def _prior_mean_cov_along_grid(
        mean0,
        cov0,
        prior,
        t_grid):
    """
    Prior mean and covariance at each time in t_grid (prediction only).

    Returns mean_arr (n, 2), std_arr (n, 2) for x and x' (marginal std).
    """
    n = len(t_grid)
    dt = (t_grid[-1] - t_grid[0]) / max(1, n - 1)
    means = [mean0]
    covs = [cov0]
    for _ in range(n - 1):
        f = prior.transition_matrix(dt)
        q = prior.process_covariance(dt)
        m = f @ means[-1]
        c = f @ covs[-1] @ f.T + q
        means.append(m)
        covs.append(c)
    mean_arr = jnp.stack(means)
    cov_arr = jnp.stack(covs)
    std_arr = jnp.sqrt(jnp.array([jnp.diag(c) for c in cov_arr]))
    return mean_arr, std_arr

def run_prediction_and_update(key, x0, prior, config):
    """
    Run prediction-only (prior over [0, dt]) and multi-step EKF0 over [0, dt].
    config : _IntegrationConfig
        dt, t_grid, num_samples, num_steps.
    """
    state0 = build_initial_state(prior, x0)
    key_pred, key_filter, key_post = jax.random.split(key, 3)
    pred_data = compute_prior_data(key_pred, state0, prior, config)
    filter_data = ensemble_filter_trajectory(key_filter, state0, prior, config)
    post_data = endpoint_posterior_samples(
        key_post, state0, prior, config.dt, config.num_samples
    )
    true_data = true_curves(config.t_grid, x0)
    return {
        "t_grid": config.t_grid,
        "true_x": true_data[0],
        "true_xd": true_data[1],
        "true_xdd": true_data[2],
        "pred_samples": pred_data[0],
        "pred_mean": pred_data[1],
        "pred_std": pred_data[2],
        "step_times": filter_data[0],
        "filter_mean": filter_data[1],
        "filter_std": filter_data[2],
        "post_step_samples": filter_data[3],
        "post_mean": post_data[0],
        "post_cov": post_data[1],
        "post_samples": post_data[2],
    }

def build_initial_state(prior, x0, t0=0.0):
    """Create initial ODE filter state from x0 and prior."""
    if hasattr(prior, "initial_mean_from_vector_field"):
        mean0 = prior.initial_mean_from_vector_field(_vector_field, x0, t0)
    else:
        rhs0 = _vector_field(x0, t0)
        mean0 = prior.initial_mean(x0, derivatives=(rhs0,))
    cov0 = prior.initial_covariance(scale=1e-6)
    return ODEFilterState(mean=mean0, cov=cov0)

def endpoint_posterior_samples(key, state0, prior, dt, num_samples):
    """Compute endpoint posterior state and draw samples."""
    state1 = ode_filter_advance(state0, _vector_field, dt, prior)
    post_samples = jax.random.multivariate_normal(
        key, state1.mean, state1.cov, (num_samples,)
    )
    return state1.mean, state1.cov, post_samples

def true_curves(t_grid, x0):
    """Return true x(t) and x'(t) along t_grid."""
    true_x = _true_solution(t_grid, float(x0))
    true_xd = jax.vmap(_vector_field)(true_x, t_grid)
    true_xdd = 0.75 * (t_grid + 1.0) ** (-2.5)  # d2/dt2 of 1/sqrt(t+1)
    return true_x, true_xd, true_xdd

def _first_docline(func):
    """First non-empty line of func's docstring, or func name if missing."""
    doc = (func.__doc__ or "").strip()
    for line in doc.split("\n"):
        line = line.strip()
        if line:
            return line
    return func.__name__

def _draw_prediction_column(axes_col, data):
    """Draw prediction step subplots for x(t) and x'(t) on axes_col [top, bottom].

    Mean and ±2σ are from the IWP prior (extrapolation); mean tracks the true
    solution approximately over short intervals but is not exact, per book Fig. 38.1.
    """
    t = data["t_grid"]
    pred_mean = data["pred_mean"]
    pred_std = data["pred_std"]
    pred_samples = data["pred_samples"]
    for ax, true_vals, idx in [
        (axes_col[0], data["true_x"], 0),
        (axes_col[1], data["true_xd"], 1),
        (axes_col[2], data["true_xdd"], 2)
    ]:
        for i in range(pred_samples.shape[0]):
            ax.plot(t, pred_samples[i, :, idx], color="gray", linewidth=0.6, alpha=0.8)
        ax.plot(t, true_vals, "k:", label="true solution", linewidth=1.2)
        ax.plot(t, pred_mean[:, idx], "k-", label="mean", linewidth=1.2)
        ax.plot(t, pred_mean[:, idx] - 2 * pred_std[:, idx], "k--", linewidth=0.9)
        ax.plot(t, pred_mean[:, idx] + 2 * pred_std[:, idx], "k--", linewidth=0.9, label="±2σ")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)


def _draw_update_column(axes_col, data):
    """Draw update step subplots for x(t) and x'(t) on axes_col [top, bottom]."""
    band_bounds = _posterior_band_bounds(data["t_grid"])
    _draw_update_axis(axes_col[0], 0, data["true_x"], data, band_bounds)
    _draw_update_axis(axes_col[1], 1, data["true_xd"], data, band_bounds)
    _draw_update_axis(axes_col[2], 2, data["true_xdd"], data, band_bounds)

def _posterior_band_bounds(t_grid):
    """Return (t1, x_left, x_right) for the endpoint posterior band."""
    t1 = float(t_grid[-1])
    t_range = float(t_grid[-1] - t_grid[0]) if len(t_grid) > 1 else 1.0
    half_width = 0.015 * t_range
    return t1, t1 - half_width, t1 + half_width

def _draw_update_axis(ax, idx, true_vals, data, band_bounds):
    """Draw one update subplot (either x or x')."""
    t1, x_left, x_right = band_bounds
    step_times = data["step_times"]
    post_step_samples = data["post_step_samples"]
    filter_mean = data["filter_mean"]
    filter_std = data["filter_std"]
    post_mean = data["post_mean"]
    post_std = jnp.sqrt(jnp.diag(data["post_cov"]))

    for i in range(post_step_samples.shape[1]):
        ax.plot(
            step_times,
            post_step_samples[:, i, idx],
            color="gray",
            linewidth=0.6,
            alpha=0.8,
        )

    ax.plot(data["t_grid"], true_vals, "k:", linewidth=1.2, label="true solution")
    ax.plot(step_times, filter_mean[:, idx], "k-", linewidth=1.2, label="mean")
    ax.plot(step_times, filter_mean[:, idx] - 2 * filter_std[:, idx], "k--", linewidth=0.9)
    ax.plot(
        step_times,
        filter_mean[:, idx] + 2 * filter_std[:, idx],
        "k--",
        linewidth=0.9,
        label="±2σ",
    )
    ax.axvline(t1, color="red", alpha=0.4, linestyle="-", linewidth=1)
    ax.scatter([t1], [post_mean[idx]], color="black", s=30, zorder=5, label="posterior mean")
    ax.fill_between(
        [x_left, x_right],
        [post_mean[idx] - 2 * post_std[idx], post_mean[idx] - 2 * post_std[idx]],
        [post_mean[idx] + 2 * post_std[idx], post_mean[idx] + 2 * post_std[idx]],
        alpha=0.25,
        color="blue",
        label="posterior ±2σ",
    )
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)

def _plot_ekf0_figure(data, output_path="ode_filter_ekf0_demo.png"):
    """Plot 2×2 grid: prediction step | update step, rows x(t) and x'(t)."""
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(10, 8))
    fig.text(
        0.5, 0.98,
        "EKF0 with 2-times IWP prior: prediction step | update step",
        ha="center", fontsize=11, transform=fig.transFigure
    )
    desc = f"{_first_docline(_vector_field)}  →  {_first_docline(_true_solution)}"
    fig.text(0.5, 0.94, desc, ha="center", fontsize=9, transform=fig.transFigure)
    for col, title in enumerate(["prediction step", "update step"]):
        axes[0, col].set_title(title)
    axes[0, 0].set_ylabel("x(t)")
    axes[1, 0].set_ylabel("x'(t)")
    axes[2, 0].set_ylabel("x''(t)")
    axes[2, 0].set_xlabel("time t")
    axes[2, 1].set_xlabel("time t")

    _draw_prediction_column([axes[0, 0], axes[1, 0], axes[2, 0]], data)
    _draw_update_column([axes[0, 1], axes[1, 1], axes[2, 1]], data)

    plt.tight_layout(rect=(0, 0, 1, 0.90))
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()
    print("Saved", output_path)


def main():
    """Run EKF0 demo: build data and plot or print fallback."""
    key = jax.random.PRNGKey(42)
    x0 = jnp.array(1.0)
    dt = 1
    n_grid = 500
    t_grid = jnp.linspace(0.0, dt, n_grid)
    prior = IWP3Prior(0.5)
    num_samples = 50
    num_steps = 3

    config = _IntegrationConfig(
        dt=dt,
        t_grid=t_grid,
        num_samples=num_samples,
        num_steps=num_steps,
    )
    data = run_prediction_and_update(key, x0, prior, config)

    if plt is not None:
        _plot_ekf0_figure(data)
    else:
        print("post_mean at t1:", data["post_mean"])
        print("Install matplotlib to generate the figure.")


if __name__ == "__main__":
    main()
