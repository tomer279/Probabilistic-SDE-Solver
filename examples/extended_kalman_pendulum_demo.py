"""
Pendulum model and Jacobians for EKF (additive noise).

Exports:
    f : state transition f_{k-1}(x_{k-1})
    h : measurement map h(x_k)
    Q : process noise covariance Q_{k-1}
    J_f : Jacobian of f
    J_h : Jacobian of h
"""

import numpy as np


def f(x, dt, g):
    """
    Dynamic model (state transition).

    x_k = f(x_{k-1}) + q_{k-1}, with
    f(x) = [ x_1 + x_2*dt ; x_2 - g*sin(x_1)*dt ].

    Parameters
    ----------
    x : array-like, shape (2,)
        State (angle x_1, angular velocity x_2).
    dt : float
        Time step Δt_{k-1}.
    g : float
        Gravity-related constant.

    Returns
    -------
    ndarray, shape (2,)
        f(x).
    """
    x = np.asarray(x, dtype=float)
    x1, x2 = x[0], x[1]
    out = np.array([
        x1 + x2 * dt,
        x2 - g * np.sin(x1) * dt,
    ])
    return out


def h(x):
    """
    Measurement model.

    y_k = h(x_k) + r_k, with h(x) = sin(x_1).

    Parameters
    ----------
    x : array-like, shape (2,)
        State (angle x_1, angular velocity x_2).

    Returns
    -------
    float
        h(x).
    """
    x = np.asarray(x, dtype=float)
    return float(np.sin(x[0]))


def Q(dt, qc):
    """
    Process noise covariance Q_{k-1}.

    Q = q^c * [[ dt^3/3, dt^2/2 ], [ dt^2/2, dt ]].

    Parameters
    ----------
    dt : float
        Time step Δt_{k-1}.
    qc : float
        Spectral density q^c of the continuous-time process noise.

    Returns
    -------
    ndarray, shape (2, 2)
        Covariance matrix.
    """
    dt2 = dt * dt
    dt3 = dt2 * dt
    return qc * np.array([
        [dt3 / 3.0, dt2 / 2.0],
        [dt2 / 2.0, dt],
    ])


def J_f(x, dt, g):
    """
    Jacobian of the dynamic model f with respect to the state.

    J_f = [[ 1, dt ], [ -g*cos(x_1)*dt, 1 ]].

    Parameters
    ----------
    x : array-like, shape (2,)
        State (angle x_1, angular velocity x_2).
    dt : float
        Time step.
    g : float
        Gravity-related constant.

    Returns
    -------
    ndarray, shape (2, 2)
        Jacobian matrix.
    """
    x = np.asarray(x, dtype=float)
    x1 = x[0]
    return np.array([
        [1.0, dt],
        [-g * np.cos(x1) * dt, 1.0],
    ])


def J_h(x):
    """
    Jacobian of the measurement model h with respect to the state.

    J_h = ( cos(x_1), 0 ).  Returned as shape (1, 2) for matrix use.

    Parameters
    ----------
    x : array-like, shape (2,)
        State (angle x_1, angular velocity x_2).

    Returns
    -------
    ndarray, shape (1, 2)
        Jacobian row vector.
    """
    x = np.asarray(x, dtype=float)
    return np.array([[np.cos(x[0]), 0.0]])


def ekf_predict(m, P, dt, g, qc):
    """
    EKF prediction step.

    m_bar = f(m),  P_bar = J_f(m) P J_f(m)^T + Q.

    Parameters
    ----------
    m : array-like, shape (2,)
        Posterior mean at k-1.
    P : array-like, shape (2, 2)
        Posterior covariance at k-1.
    dt : float
        Time step.
    g : float
        Gravity-related constant.
    qc : float
        Process noise spectral density.

    Returns
    -------
    m_bar : ndarray, shape (2,)
        Predicted mean.
    P_bar : ndarray, shape (2, 2)
        Predicted covariance.
    """
    m = np.asarray(m, dtype=float)
    P = np.asarray(P, dtype=float)
    m_bar = f(m, dt, g)
    J = J_f(m, dt, g)
    P_bar = J @ P @ J.T + Q(dt, qc)
    return m_bar, P_bar


def ekf_update(m_bar, P_bar, y_k, R):
    """
    EKF update step (additive measurement noise).

    v = y_k - h(m_bar),  S = J_h P_bar J_h^T + R,
    K = P_bar J_h^T S^{-1},  m = m_bar + K v,  P = P_bar - K S K^T.

    Parameters
    ----------
    m_bar : array-like, shape (2,)
        Predicted mean.
    P_bar : array-like, shape (2, 2)
        Predicted covariance.
    y_k : float
        Measurement at current step.
    R : float
        Measurement noise variance (scalar).

    Returns
    -------
    m : ndarray, shape (2,)
        Posterior mean.
    P : ndarray, shape (2, 2)
        Posterior covariance.
    """
    m_bar = np.asarray(m_bar, dtype=float)
    P_bar = np.asarray(P_bar, dtype=float)
    jh = J_h(m_bar)
    v = y_k - h(m_bar)
    S = (jh @ P_bar @ jh.T)[0, 0] + R
    S_inv = 1.0 / S
    K = (P_bar @ jh.T) * S_inv
    m = m_bar + K.ravel() * v
    P = P_bar - np.outer(K, K) * S
    return m, P


def run_ekf(t, y, dt, g, qc, R, m0=None, P0=None):
    """
    Run EKF over time: predict then update at each step.

    Parameters
    ----------
    t : array-like, shape (N,)
        Time grid (used for length; can use uniform dt).
    y : array-like, shape (N,)
        Measurements y_k.
    dt : float
        Time step.
    g : float
        Gravity-related constant.
    qc : float
        Process noise spectral density.
    R : float
        Measurement noise variance.
    m0 : array-like, shape (2,), optional
        Prior mean. Default (0, 0).
    P0 : array-like, shape (2, 2), optional
        Prior covariance. Default I.

    Returns
    -------
    m_arr : ndarray, shape (N, 2)
        Posterior means m_0, m_1, ..., m_{N-1}.
    P_arr : ndarray, shape (N, 2, 2)
        Posterior covariances.
    """
    n = len(y)
    m_arr = np.zeros((n, 2))
    P_arr = np.zeros((n, 2, 2))
    if m0 is None:
        m0 = np.zeros(2)
    if P0 is None:
        P0 = np.eye(2)
    m_arr[0] = m0
    P_arr[0] = P0
    for k in range(1, n):
        m_bar, P_bar = ekf_predict(m_arr[k - 1], P_arr[k - 1], dt, g, qc)
        m_arr[k], P_arr[k] = ekf_update(m_bar, P_bar, y[k], R)
    return m_arr, P_arr


def simulate_pendulum(g, dt, T, R, x1_0, x2_0, rng=None):
    """Simulate pendulum state and measurements. Returns t, x1_true, x2_true, y."""
    if rng is None:
        rng = np.random.default_rng()
    n_steps = int(round(T / dt)) + 1
    t = np.linspace(0, T, n_steps)
    x1_true = np.zeros(n_steps)
    x2_true = np.zeros(n_steps)
    x1_true[0], x2_true[0] = x1_0, x2_0
    for k in range(1, n_steps):
        x1_true[k] = x1_true[k - 1] + x2_true[k - 1] * dt
        x2_true[k] = x2_true[k - 1] - g * np.sin(x1_true[k - 1]) * dt
    y = np.sin(x1_true) + np.sqrt(R) * rng.standard_normal(n_steps)
    return t, x1_true, x2_true, y


def main():
    import matplotlib.pyplot as plt
    g, dt, T, R = 9.81, 0.025, 4.5, 0.1
    qc = 1
    x1_0, x2_0 = 0.5, 0.0
    rng = np.random.default_rng()
    t, x1_true, x2_true, y = simulate_pendulum(g, dt, T, R, x1_0, x2_0, rng=rng)
    m0 = np.zeros(2)
    P0 = np.eye(2)
    m_arr, P_arr = run_ekf(t, y, dt, g, qc, R, m0=m0, P0=P0)
    true_measurement = np.sin(x1_true)
    ekf_measurement = np.sin(m_arr[:, 0])
    plt.figure()
    plt.plot(t, true_measurement, color="gray", linewidth=2, label="True (sin x_1,k)")
    plt.scatter(
        t, y, s=3, c="black", label="Measurements y_k", zorder=3)
    plt.plot(t, ekf_measurement, color="C0", linewidth=1.5, label="EKF estimate (sin m_1,k)")
    plt.ylim(-3, 5)
    plt.xlabel("Time t")
    plt.ylabel("Measurement y_k (sin of angle)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()