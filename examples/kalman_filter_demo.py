"""
Simulate the Gaussian random walk from Sarkka Example 6.4
and plot signal and measurements (Figure 6.1 style).
"""
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()

def simulate_random_walk(num_steps, process_var, meas_var, x0=0.0):
    """
    Simulate state x_k and measurements y_k for the Gaussian random walk.
    x_k = x_{k-1} + q_{k-1},  q_{k-1} ~ N(0, Q)
    y_k = x_k + r_k,          r_k ~ N(0, R)
    Parameters
    ----------
    num_steps : int
        Number of time steps (e.g. 101 for k = 0, ..., 100).
    process_var : float
        Process noise variance Q.
    meas_var : float
        Measurement noise variance R.
    x0 : float
        Initial state x_0.
    rng : np.random.Generator or None
        Random number generator for reproducibility.
    Returns
    -------
    signal : ndarray, shape (num_steps,)
        True state x_0, x_1, ..., x_{num_steps-1}.
    measurements : ndarray, shape (num_steps,)
        Noisy measurements y_0, y_1, ..., y_{num_steps-1}.
    """
    signal = np.empty(num_steps)
    signal[0] = x0
    for k in range(1, num_steps):
        signal[k] = signal[k - 1] + rng.normal(0, np.sqrt(process_var))
    measurements = signal + rng.normal(0, np.sqrt(meas_var), size=num_steps)
    return signal, measurements

def kalman_filter_random_walk(measurements, process_var, meas_var, m0=None, P0=None):
    """
    Scalar Kalman filter for the Gaussian random walk (Sarkka Example 6.7, Eq. 6.31).

    State model: x_k = x_{k-1} + q_{k-1}, q_{k-1} ~ N(0, Q).
    Observation: y_k = x_k + r_k, r_k ~ N(0, R).
    Returns posterior mean m_k and variance P_k at each time step.

    Parameters
    ----------
    measurements : ndarray, shape (N,)
        Observed values y_0, y_1, ..., y_{N-1}.
    process_var : float
        Process noise variance Q.
    meas_var : float
        Measurement noise variance R.
    m0 : float or None
        Initial state mean. If None, use measurements[0].
    P0 : float or None
        Initial state variance. If None, use meas_var (R).

    Returns
    -------
    m : ndarray, shape (N,)
        Posterior mean m_k (filter estimate).
    P : ndarray, shape (N,)
        Posterior variance P_k.
    """
    num_steps = len(measurements)
    if m0 is None:
        m0 = float(measurements[0])
    if P0 is None:
        P0 = meas_var
    m = np.empty(num_steps)
    P = np.empty(num_steps)
    m[0] = m0
    P[0] = P0
    for k in range(1, num_steps):
        m_pred = m[k - 1]
        P_pred = P[k - 1] + process_var
        v = measurements[k] - m_pred
        S = P_pred + meas_var
        K = P_pred / S
        m[k] = m_pred + K * v
        P[k] = P_pred - K * P_pred
    return m, P

def plot_signal_and_measurements(time_steps, signal, measurements, title=None):
    """
    Plot true signal (gray line) and measurements (open circles), Figure 6.1 style.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(time_steps, signal, color="gray", linewidth=2.5, label="Signal")
    plt.plot(
        time_steps,
        measurements,
        linestyle="",
        marker="o",
        markerfacecolor="none",
        markeredgecolor="black",
        markersize=4,
        label="Measurement",
    )
    plt.xlabel("Time step k")
    plt.ylabel("$x_k$")
    if title is not None:
        plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    
def plot_signal_measurements_and_filter(
    time_steps, signal, measurements, m, P, title=None
):
    """
    Plot signal, measurements, Kalman filter estimate, and 95% quantiles (Figure 6.4 style).
    """
    plt.figure(figsize=(8, 4))
    plt.plot(time_steps, signal, color="gray", linewidth=2.5, label="Signal")
    plt.plot(
        time_steps,
        measurements,
        linestyle="",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=4,
        label="Measurement",
    )
    plt.plot(time_steps, m, color="black", linewidth=2.5, label="Filter estimate")
    std = np.sqrt(np.maximum(P, 0.0))
    plt.plot(
        time_steps,
        m - 1.96 * std,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="95% quantiles",
    )
    plt.plot(time_steps, m + 1.96 * std, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Time step k")
    plt.ylabel("$x_k$")
    if title is not None:
        plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

# Example: reproduce Example 6.4 with Q = R = 1, 101 steps (k = 0, ..., 100)
# Reproduce Example 6.4 / 6.7 with Q = R = 1, 101 steps (k = 0, ..., 100)
Q = 1.0
R = 1.0
num_steps = 101
rng = np.random.default_rng(42)  # fixed seed so both figures use same data

signal, measurements = simulate_random_walk(num_steps, Q, R, x0=0.0)
time_steps = np.arange(num_steps)

# Figure 6.1 style: signal and measurements only
plot_signal_and_measurements(
    time_steps,
    signal,
    measurements,
    title="Simulated signal and measurements from the Gaussian random walk model (Example 6.4)",
)

# Kalman filter (Example 6.7, Eq. 6.31) then Figure 6.4 style
m, P = kalman_filter_random_walk(measurements, Q, R)
plot_signal_measurements_and_filter(
    time_steps,
    signal,
    measurements,
    m,
    P,
    title="Signal, measurements, and result of Kalman filtering of the Gaussian random walk (Example 6.7)",
)
