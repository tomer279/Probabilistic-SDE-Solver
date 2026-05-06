"""
2D car tracking with Kalman filter (Sarkka Example 6.8).
Simulates constant-velocity model, noisy position measurements, and plots
true trajectory, measurements, and filter estimate.
"""
import numpy as np
import matplotlib.pyplot as plt


def build_car_model(dt, q1c, q2c, sigma1, sigma2):
    """
    Build A, Q, H, R for the 2D constant-velocity car model (Example 6.8).

    Parameters
    ----------
    dt : float
        Time step Δt.
    q1c, q2c : float
        Process noise spectral densities (x and y).
    sigma1, sigma2 : float
        Measurement noise standard deviations (x and y).

    Returns
    -------
    A : ndarray, shape (4, 4)
        State transition matrix.
    Q : ndarray, shape (4, 4)
        Process noise covariance.
    H : ndarray, shape (2, 4)
        Observation matrix.
    R : ndarray, shape (2, 2)
        Measurement noise covariance.
    """
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    Q = np.array([
        [q1c * dt**3 / 3, 0, q1c * dt**2 / 2, 0],
        [0, q2c * dt**3 / 3, 0, q2c * dt**2 / 2],
        [q1c * dt**2 / 2, 0, q1c * dt, 0],
        [0, q2c * dt**2 / 2, 0, q2c * dt],
    ], dtype=float)
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=float)
    R = np.diag([sigma1**2, sigma2**2])
    return A, Q, H, R


def simulate_car_trajectory(num_steps, A, Q, H, R, x0, rng=None):
    """
    Simulate true state trajectory and noisy position measurements.

    Parameters
    ----------
    num_steps : int
        Number of time steps (including k=0).
    A, Q, H, R : ndarray
        Model matrices from build_car_model.
    x0 : ndarray, shape (4,)
        Initial state [pos_x, pos_y, vel_x, vel_y].
    rng : np.random.Generator or None
        Random generator for reproducibility.

    Returns
    -------
    states : ndarray, shape (num_steps, 4)
        True states x_0, ..., x_{num_steps-1}.
    measurements : ndarray, shape (num_steps, 2)
        Noisy position observations y_0, ..., y_{num_steps-1}.
    """
    if rng is None:
        rng = np.random.default_rng()
    states = np.empty((num_steps, 4))
    states[0] = x0
    for k in range(1, num_steps):
        q = rng.multivariate_normal(np.zeros(4), Q)
        states[k] = A @ states[k - 1] + q
    # y_k = H x_k + r_k
    noise = rng.multivariate_normal(np.zeros(2), R, size=num_steps)
    measurements = (states @ H.T) + noise
    return states, measurements


def kalman_filter_car(measurements, A, Q, H, R, m0=None, P0=None):
    """
    Run Kalman filter for the 2D car model (predict + update each step).

    Parameters
    ----------
    measurements : ndarray, shape (N, 2)
        Position observations at each time step.
    A, Q, H, R : ndarray
        Model matrices.

    Returns
    -------
    m : ndarray, shape (N, 4)
        Posterior state mean at each step.
    P : ndarray, shape (N, 4, 4)
        Posterior state covariance at each step.
    """
    num_steps = len(measurements)
    m = np.empty((num_steps, 4))
    P = np.empty((num_steps, 4, 4))
    if m0 is None:
        m0 = np.array([measurements[0, 0], measurements[0, 1], 0.0, 0.0])
    if P0 is None:
        P0 = np.diag([R[0, 0], R[1, 1], 1.0, 1.0])  # position uncertainty from R, velocity large
    m[0] = m0
    P[0] = P0
    for k in range(1, num_steps):
        m_pred = A @ m[k - 1]
        P_pred = A @ P[k - 1] @ A.T + Q
        y_k = measurements[k]
        v = y_k - (H @ m_pred)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        m[k] = m_pred + K @ v
        P[k] = P_pred - K @ S @ K.T
    return m, P


def plot_car_tracking_2d(states, measurements, m, title=None):
    """
    Plot true trajectory, measurements, and filter estimate in the (x_1, x_2) plane (Figure 6.x style).
    """
    plt.figure(figsize=(8, 6))
    plt.plot(
        states[:, 0], states[:, 1],
        color="gray", linewidth=2.5, label="True trajectory",
    )
    plt.plot(
        measurements[:, 0], measurements[:, 1],
        linestyle="", marker="o", markerfacecolor="none", markeredgecolor="black",
        markersize=4, label="Measurement",
    )
    plt.plot(
        m[:, 0], m[:, 1],
        color="black", linewidth=2.5, label="Filter estimate",
    )
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    if title is not None:
        plt.title(title)
    plt.legend(loc="best")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


# --- Example: book parameters and single run ---
if __name__ == "__main__":
    dt = 1.0 / 10
    sigma1, sigma2 = 0.5, 0.5
    q1c, q2c = 1.0, 1.0
    A, Q, H, R = build_car_model(dt, q1c, q2c, sigma1, sigma2)

    num_steps = 101  # e.g. k = 0, ..., 100
    x0 = np.array([-1.0, 0.0, 0.5, -0.2])  # initial position and velocity (tune for nice path)
    rng = np.random.default_rng()

    states, measurements = simulate_car_trajectory(num_steps, A, Q, H, R, x0, rng=rng)
    m, P = kalman_filter_car(measurements, A, Q, H, R)

    plot_car_tracking_2d(
        states, measurements, m,
        title="Kalman filter for car tracking (Example 6.8)",
    )