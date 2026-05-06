"""
Algorithm-1 Brownian approximation accuracy demo.

Exported functions
------------------
generate_reference_brownian_path
    Simulate a fine-grid Brownian motion sample path on [0, horizon].
build_piecewise_linear_from_reference
    Build the corresponding piecewise-linear Algorithm-1 approximation.
build_piecewise_parabolic_from_reference
    Build the corresponding piecewise-parabolic approximation using interval
    bridge area.
compute_error_metrics
    Compute RMSE and max-absolute error.
main
    Run the demo and produce a comparison plot.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from prob_sde.brownian.brownian import (
    piecewise_linear_brownian,
    piecewise_parabolic_brownian
)

def generate_key():
    seed = int(np.random.SeedSequence().generate_state(1)[0])
    return jax.random.PRNGKey(seed)

def generate_reference_brownian_path(key, horizon, fine_steps):
    """Return a fine-grid Brownian sample path."""
    delta_fine = horizon / fine_steps
    increments = jax.random.normal(key, (fine_steps,)) * jnp.sqrt(delta_fine)
    values = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(increments)])
    times = jnp.linspace(0.0, horizon, fine_steps + 1)
    return np.asarray(times), np.asarray(values)


def _coarse_stride(total_steps, coarse_steps):
    """Return stride size and validate divisibility."""
    if total_steps % coarse_steps != 0:
        raise ValueError("coarse_steps must divide fine_steps exactly.")
    return total_steps // coarse_steps


def _interval_bridge_area(interval_times, interval_values, left_value, right_value):
    """Approximate integral of Brownian-bridge part over one interval."""
    delta = interval_times[-1] - interval_times[0]
    u = (interval_times - interval_times[0]) / delta
    linear_part = left_value + (right_value - left_value) * u
    bridge_part = interval_values - linear_part
    return np.trapezoid(bridge_part, interval_times)


def _estimate_i_delta_from_area(area, delta):
    """Map bridge area to i_delta coefficient for parabolic basis."""
    return -(np.sqrt(6.0) / delta) * area


def _local_segment(eval_fn, times, left_idx, right_idx, coeffs):
    """Evaluate one local Algorithm-1 segment on a fine subgrid."""
    local_times = times[left_idx:right_idx + 1]
    delta = local_times[-1] - local_times[0]
    local_t = local_times - local_times[0]
    return np.asarray(eval_fn(jnp.asarray(local_t), delta, *coeffs))


def _insert_segment(approx, segment, left_idx, right_idx, is_first):
    """Insert segment into global array with endpoint de-duplication."""
    if is_first:
        approx[left_idx:right_idx + 1] = segment
        return
    approx[left_idx + 1:right_idx + 1] = segment[1:]


def _build_piecewise(times, coarse_steps, eval_fn, coeffs_fn):
    """Build a global piecewise approximation from local coefficients."""
    total_steps = len(times) - 1
    stride = _coarse_stride(total_steps, coarse_steps)
    approx = np.zeros(total_steps + 1)
    level = coeffs_fn.initial_level

    for k in range(coarse_steps):
        left_idx = k * stride
        right_idx = (k + 1) * stride
        coeffs, increment = coeffs_fn(k, left_idx, right_idx)
        local_eval = _local_segment(eval_fn, times, left_idx, right_idx, coeffs)
        segment = level + local_eval
        _insert_segment(approx, segment, left_idx, right_idx, is_first=k == 0)
        level = level + increment

    return approx


def _linear_coeffs_factory(coarse_values):
    """Return coefficient callback for piecewise-linear approximation."""

    def coeffs_fn(k, _left_idx, _right_idx):
        left_value = coarse_values[k]
        right_value = coarse_values[k + 1]
        w_delta = right_value - left_value
        return (0.0, w_delta), w_delta

    coeffs_fn.initial_level = coarse_values[0]
    return coeffs_fn


def _parabolic_coeffs_factory(times, values, coarse_values):
    """Return coefficient callback for piecewise-parabolic approximation."""

    def coeffs_fn(k, left_idx, right_idx):
        left_value = coarse_values[k]
        right_value = coarse_values[k + 1]
        interval_times = times[left_idx:right_idx + 1]
        interval_values = values[left_idx:right_idx + 1]
        delta = interval_times[-1] - interval_times[0]
        area = _interval_bridge_area(
            interval_times,
            interval_values,
            left_value,
            right_value,
        )
        w_delta = right_value - left_value
        i_delta = _estimate_i_delta_from_area(area, delta)
        return (0.0, w_delta, i_delta), w_delta

    coeffs_fn.initial_level = coarse_values[0]
    return coeffs_fn


def build_piecewise_linear_from_reference(times, values, coarse_steps):
    """Construct piecewise-linear Algorithm-1 approximation."""
    _, eval_fn = piecewise_linear_brownian()
    total_steps = len(times) - 1
    stride = _coarse_stride(total_steps, coarse_steps)
    coarse_values = values[::stride]
    coeffs_fn = _linear_coeffs_factory(coarse_values)
    return _build_piecewise(times, coarse_steps, eval_fn, coeffs_fn)


def build_piecewise_parabolic_from_reference(times, values, coarse_steps):
    """Construct piecewise-parabolic Algorithm-1 approximation."""
    _, eval_fn = piecewise_parabolic_brownian()
    total_steps = len(times) - 1
    stride = _coarse_stride(total_steps, coarse_steps)
    coarse_values = values[::stride]
    coeffs_fn = _parabolic_coeffs_factory(times, values, coarse_values)
    return _build_piecewise(times, coarse_steps, eval_fn, coeffs_fn)


def compute_error_metrics(reference, approximation):
    """Return RMSE and max absolute error."""
    error = approximation - reference
    rmse = float(np.sqrt(np.mean(error * error)))
    max_abs = float(np.max(np.abs(error)))
    return rmse, max_abs

def plot_results(
        axes,
        times,
        reference,
        linear_approx,
        parabolic_approx):

    axes[0].plot(
        times,
        reference,
        color="tab:blue",
        linewidth=0.9,
        label="Brownian path",
    )
    axes[0].plot(
        times,
        linear_approx,
        color="tab:orange",
        linewidth=1.2,
        label="Algorithm 1 linear",
    )
    axes[0].plot(
        times,
        parabolic_approx,
        color="tab:green",
        linewidth=1.2,
        label="Algorithm 1 parabolic",
    )
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("W(t)")
    axes[0].set_title("Brownian path and corresponding Algorithm-1 approximations")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        times,
        np.abs(reference - linear_approx),
        color="tab:orange",
        linewidth=1.1,
        label="|error| linear",
    )
    axes[1].plot(
        times,
        np.abs(reference - parabolic_approx),
        color="tab:green",
        linewidth=1.1,
        label="|error| parabolic",
    )
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("Absolute error")
    axes[1].set_title("Pathwise error")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

def main():
    """Run one accuracy demo and save the figure."""
    if plt is None:
        print("Install matplotlib to generate the plot.")
        return

    key = generate_key()
    t_final = 1.0
    fine_steps = 4000
    coarse_steps = 40

    times, reference = generate_reference_brownian_path(key, t_final, fine_steps)
    linear_approx = build_piecewise_linear_from_reference(
        times,
        reference,
        coarse_steps,
    )
    parabolic_approx = build_piecewise_parabolic_from_reference(
        times,
        reference,
        coarse_steps,
    )

    linear_rmse, linear_max = compute_error_metrics(reference, linear_approx)
    parabolic_rmse, parabolic_max = compute_error_metrics(reference, parabolic_approx)

    print("Linear RMSE:", linear_rmse, "Linear max-abs:", linear_max)
    print("Parabolic RMSE:", parabolic_rmse, "Parabolic max-abs:", parabolic_max)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    plot_results(axes, times, reference, linear_approx, parabolic_approx)

    fig.tight_layout()
    output_path = "brownian_algorithm1_accuracy.png"
    fig.savefig(output_path, dpi=150)
    plt.show()
    plt.close(fig)
    print("Saved " + output_path)


if __name__ == "__main__":
    main()
