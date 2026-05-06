
import numpy as np
import matplotlib.pyplot as plt

# Radar positions to match the reference figure (sensors as black outlined triangles)
RADAR_POSITIONS = np.array([
    [-1.5, 0.5],   # Sensor 1
    [1.0, 1.0],    # Sensor 2
    [-0.5, -1.5],  # Sensor 3
    [1.0, -1.0],   # Sensor 4
])

# State indices
IDX_X1, IDX_X2, IDX_S, IDX_PHI, IDX_OMEGA = 0, 1, 2, 3, 4
STATE_DIM = 5


def polar_turn_transition(state: np.ndarray, dt: float) -> np.ndarray:
    """
    State transition x_k = f(x_{k-1}) for the polar coordinated turn model.
    state = (x1, x2, s, phi, omega). Handles omega ≈ 0 with constant-velocity limit.
    """
    x1, x2, s, phi, omega = state
    omega_abs = np.abs(omega)
    if omega_abs < 1e-10:
        # Straight line: x += s*dt*(cos(phi), sin(phi))
        dx = s * dt * np.array([np.cos(phi), np.sin(phi)])
        return np.array([
            x1 + dx[0],
            x2 + dx[1],
            s,
            phi,
            omega,
        ])
    half_angle = 0.5 * dt * omega
    factor = (2.0 * s / omega) * np.sin(half_angle)
    mid_angle = phi + half_angle
    return np.array([
        x1 + factor * np.cos(mid_angle),
        x2 + factor * np.sin(mid_angle),
        s,
        phi + omega * dt,
        omega,
    ])


def polar_turn_jacobian_f(state: np.ndarray, dt: float) -> np.ndarray:
    """
    Jacobian F = d f / d x (5x5) for the polar coordinated turn model.
    """
    x1, x2, s, phi, omega = state
    F = np.zeros((STATE_DIM, STATE_DIM))
    omega_abs = np.abs(omega)
    if omega_abs < 1e-10:
        # Straight-line Jacobian
        F[0, 0] = F[1, 1] = F[2, 2] = F[4, 4] = 1.0
        F[0, 2] = dt * np.cos(phi)
        F[0, 3] = -s * dt * np.sin(phi)
        F[1, 2] = dt * np.sin(phi)
        F[1, 3] = s * dt * np.cos(phi)
        F[3, 3] = 1.0
        return F
    half = 0.5 * dt * omega
    mid = phi + half
    sin_half = np.sin(half)
    cos_half = np.cos(half)
    A = (2.0 * s / omega) * sin_half
    # Position rows
    F[0, 0] = F[1, 1] = 1.0
    F[0, 2] = (2.0 / omega) * sin_half * np.cos(mid)
    F[0, 3] = -A * np.sin(mid)
    F[1, 2] = (2.0 / omega) * sin_half * np.sin(mid)
    F[1, 3] = A * np.cos(mid)
    # d(x1_k)/d(omega) and d(x2_k)/d(omega)
    dU_dom = -2.0 * s / (omega * omega)
    dV_dom = 0.5 * dt * cos_half
    cos_mid = np.cos(mid)
    sin_mid = np.sin(mid)
    F[0, 4] = dU_dom * sin_half * cos_mid + A * (dV_dom * cos_mid + sin_half * (-sin_mid) * 0.5 * dt)
    F[1, 4] = dU_dom * sin_half * sin_mid + A * (dV_dom * sin_mid + sin_half * cos_mid * 0.5 * dt)
    # s, phi, omega
    F[2, 2] = F[4, 4] = 1.0
    F[3, 3] = 1.0
    F[3, 4] = dt
    return F


def process_noise_covariance(dt: float, qs_c: float, qom_c: float) -> np.ndarray:
    """
    Process noise covariance Q_{k-1} (5x5) for the polar turn model.
    qs_c: continuous-time spectral density for translational motion.
    qom_c: continuous-time spectral density for rotational motion.
    """
    Q = np.zeros((STATE_DIM, STATE_DIM))
    dt2 = dt * dt
    dt3 = dt2 * dt
    Q[0, 0] = Q[1, 1] = qs_c * dt3 / 3.0
    Q[2, 2] = qs_c * dt
    Q[3, 3] = qom_c * dt3 / 3.0
    Q[3, 4] = Q[4, 3] = qom_c * dt2 / 2.0
    Q[4, 4] = qom_c * dt
    return Q


def range_measurement(state: np.ndarray, radar_positions: np.ndarray) -> np.ndarray:
    """
    Deterministic measurement h(x_k): 4 range values to the four radars.
    radar_positions: (4, 2), each row (s_i,x, s_i,y).
    """
    x1, x2 = state[IDX_X1], state[IDX_X2]
    dx = radar_positions[:, 0] - x1  # (4,)
    dy = radar_positions[:, 1] - x2
    return np.sqrt(dx * dx + dy * dy)


def range_measurement_jacobian_h(state: np.ndarray, radar_positions: np.ndarray) -> np.ndarray:
    """
    Jacobian H = d h / d x (4x5). Only x1 and x2 affect ranges.
    """
    x1, x2 = state[IDX_X1], state[IDX_X2]
    dx = radar_positions[:, 0] - x1
    dy = radar_positions[:, 1] - x2
    d = np.sqrt(dx * dx + dy * dy)
    # Avoid division by zero for radar exactly on target
    d = np.where(d < 1e-12, 1e-12, d)
    dh_dx1 = -dx / d
    dh_dx2 = -dy / d
    H = np.zeros((4, STATE_DIM))
    H[:, IDX_X1] = dh_dx1
    H[:, IDX_X2] = dh_dx2
    return H

def simulate_step(
    state: np.ndarray,
    dt: float,
    qs_c: float,
    qom_c: float,
    radar_positions: np.ndarray,
    R_chol: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate one time step: state -> next state, and noisy measurement y.
    R_chol: lower Cholesky of R so R = R_chol @ R_chol.T.
    """
    Q = process_noise_covariance(dt, qs_c, qom_c)
    q = rng.multivariate_normal(np.zeros(STATE_DIM), Q)
    next_state = polar_turn_transition(state, dt) + q
    h = range_measurement(next_state, radar_positions)
    r = R_chol @ rng.standard_normal(4)
    y = h + r
    return next_state, y


def simulate_trajectory(
    initial_state: np.ndarray,
    n_steps: int,
    dt: float,
    qs_c: float,
    qom_c: float,
    radar_positions: np.ndarray,
    R_chol: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate trajectory that follows the dynamic model:
    x_k = f(x_{k-1}) + q_{k-1}, with q_{k-1} ~ N(0, Q_{k-1}).
    Returns (true_states, measurements), where true_states has shape (n_steps+1, 5).
    """
    state = np.array(initial_state, dtype=float)
    true_states = [state.copy()]
    measurements = []

    for _ in range(n_steps):
        next_state, y = simulate_step(
            state, dt, qs_c, qom_c, radar_positions, R_chol, rng
        )
        true_states.append(next_state)
        measurements.append(y)
        state = next_state

    return np.array(true_states), np.array(measurements)

def ekf_predict(
    m_prev: np.ndarray,
    P_prev: np.ndarray,
    dt: float,
    qs_c: float,
    qom_c: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    EKF prediction step.
    Returns predicted state m_k^- and predicted covariance P_k^-.
    """
    J_f = polar_turn_jacobian_f(m_prev, dt)
    m_pred = polar_turn_transition(m_prev, dt)
    Q = process_noise_covariance(dt, qs_c, qom_c)
    P_pred = J_f @ P_prev @ J_f.T + Q
    return m_pred, P_pred

def ekf_update(
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    y_k: np.ndarray,
    radar_positions: np.ndarray,
    R_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    EKF update step.
    Returns posterior state m_k and posterior covariance P_k.
    """
    h_pred = range_measurement(m_pred, radar_positions)
    v_k = y_k - h_pred
    J_h = range_measurement_jacobian_h(m_pred, radar_positions)
    S_k = J_h @ P_pred @ J_h.T + R_k
    K_k = P_pred @ J_h.T @ np.linalg.solve(S_k, np.eye(S_k.shape[0]))
    m_post = m_pred + K_k @ v_k
    P_post = P_pred - K_k @ S_k @ K_k.T
    return m_post, P_post


def run_ekf(
    m_0: np.ndarray,
    P_0: np.ndarray,
    measurements: np.ndarray,
    dt: float,
    qs_c: float,
    qom_c: float,
    radar_positions: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run EKF over all measurements.
    measurements: (n_steps, n_radars), one row per time step k=1,...,n_steps.
    Returns (ekf_means, ekf_covariances), where ekf_means has shape (n_steps+1, 5)
    with ekf_means[0] = m_0 and ekf_means[k] = posterior at step k.
    """
    n_steps = measurements.shape[0]
    ekf_means = np.zeros((n_steps + 1, STATE_DIM))
    ekf_covariances = np.zeros((n_steps + 1, STATE_DIM, STATE_DIM))
    ekf_means[0] = m_0
    ekf_covariances[0] = P_0

    m = m_0.copy()
    P = P_0.copy()
    for k in range(n_steps):
        m_pred, P_pred = ekf_predict(m, P, dt, qs_c, qom_c)
        m, P = ekf_update(m_pred, P_pred, measurements[k], radar_positions, R)
        ekf_means[k + 1] = m
        ekf_covariances[k + 1] = P

    return ekf_means, ekf_covariances

def plot_trajectory_and_sensors(
    true_states: np.ndarray,
    ekf_states: np.ndarray | None = None,
    radar_positions: np.ndarray = RADAR_POSITIONS,
) -> None:
    """
    Plot true trajectory, optional EKF estimate, and sensor positions
    in the style of the reference figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    x1_true = true_states[:, IDX_X1]
    x2_true = true_states[:, IDX_X2]
    ax.plot(x1_true, x2_true, color="gray", linewidth=2.5, label="True trajectory")
    if ekf_states is not None:
        ax.plot(
            ekf_states[:, IDX_X1],
            ekf_states[:, IDX_X2],
            color="black",
            linewidth=1.0,
            label="EKF estimate",
        )
    else:
        ax.plot(
            x1_true,
            x2_true,
            color="black",
            linewidth=1.0,
            label="EKF estimate",
        )
    ax.scatter(
        radar_positions[:, 0],
        radar_positions[:, 1],
        marker="^",
        s=80,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Sensors",
        zorder=5,
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(-2.0, 1.5)
    ax.set_ylim(-2.0, 1.5)
    ax.set_aspect("equal")
    ax.grid(True, color="lightgray")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    rng = np.random.default_rng()
    initial_state = np.array([-1.0, -1.0, 2.0, 2.0, -3.0])
    n_steps = 140
    dt = 0.05
    qs_c = 1.0
    qom_c = 1.0
    R = 0.01**2 * np.eye(4)

    true_states, measurements = simulate_trajectory(
        initial_state,
        n_steps=n_steps,
        dt=dt,
        qs_c=qs_c,
        qom_c=qom_c,
        radar_positions=RADAR_POSITIONS,
        R_chol=np.linalg.cholesky(R),
        rng=rng,
    )

    # EKF: initial guess (e.g. perturbed true initial state or just use true)
    m_0 = initial_state.copy()
    P_0 = np.diag([0.1, 0.1, 0.05, 0.05, 0.05])  # initial uncertainty

    ekf_means, ekf_covariances = run_ekf(
        m_0, P_0, measurements, dt, qs_c, qom_c, RADAR_POSITIONS, R
    )

    plot_trajectory_and_sensors(true_states, ekf_states=ekf_means)