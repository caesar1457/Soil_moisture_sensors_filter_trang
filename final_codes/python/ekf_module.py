import numpy as np
import math

class EKF:
    """
    Adaptive EKF for estimating sensor moisture and its rate of change.
    State vector: [moisture, moisture_rate].
    """
    def __init__(self, dt=1.0, x_init=0.0, P_init=1.0, base_Q=None, R=1.0):
        self.dt = dt
        self.x = np.array([x_init, 0.0], dtype=float)  # Initial state
        self.P = np.eye(2) * P_init  # Initial covariance matrix
        # Default base process noise covariance if not provided
        self.base_Q = np.array([[0.01, 0.0], [0.0, 0.01]]) if base_Q is None else base_Q
        self.Q = self.base_Q.copy()  # Adaptive process noise covariance
        self.R = R  # Measurement noise variance

    def predict(self, dt=None):
        dt = dt or self.dt
        # State transition matrix: moisture evolves with its rate
        F = np.array([[1, dt],
                      [0, 1]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z, dt=None):
        dt = dt or self.dt
        self.predict(dt)  # Prediction step

        H = np.array([[1, 0]])  # Measurement matrix (only moisture is measured)
        z_pred = H @ self.x
        y = np.array([z]) - z_pred  # Innovation (measurement residual)

        # Adaptively adjust process noise based on the innovation magnitude
        alpha = 0.1  # Tuning factor
        adaptive_factor = 1.0 + alpha * abs(y[0])
        self.Q = self.base_Q * adaptive_factor

        S = H @ self.P @ H.T + self.R  # Innovation covariance
        K = self.P @ H.T / S  # Kalman gain

        # State and covariance update
        self.x += (K.flatten() * y)
        self.P = (np.eye(2) - K @ H) @ self.P

        return self.x[0]  # Return current moisture estimate

    def get_confidence_interval(self):
        """
        Return the 3σ confidence interval for the moisture estimate,
        using the (0,0) element of the covariance matrix.
        """
        sigma = math.sqrt(self.P[0, 0])
        return self.x[0] - 3 * sigma, self.x[0] + 3 * sigma
