# ekf_module.py
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
        self.base_Q = np.array([[0.01, 0.01], [0.01, 0.01]]) if base_Q is None else base_Q
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
        # First, perform prediction using the current Q
        self.predict(dt)
        
        H = np.array([[1, 0]])
        z_pred = H @ self.x
        y = np.array([z]) - z_pred  # Measurement residual (innovation)

        # Increase alpha so that the adaptation becomes more pronounced when the innovation is large
        alpha = 0.5  # For example, increase the tuning factor to 0.5
        adaptive_factor = 1.0 + alpha * abs(y[0])
        
        # Update the process noise for the next prediction
        self.Q = self.base_Q * adaptive_factor
        
        S = H @ self.P @ H.T + self.R  # Innovation covariance
        K = self.P @ H.T / S          # Kalman gain

        # State update
        self.x += (K.flatten() * y)
        # Covariance update (standard update)
        self.P = (np.eye(2) - K @ H) @ self.P

        # If the measurement residual exceeds a certain threshold, directly inflate the current covariance
        threshold = 5.0  # This threshold can be adjusted according to the actual situation
        if abs(y[0]) > threshold:
            self.P *= adaptive_factor  # Directly scale up P so that the confidence interval becomes wider

        return self.x[0]  # Return the current moisture estimate

    def get_confidence_interval(self):
        """
        Return the 3σ confidence interval for the moisture estimate,
        using the (0,0) element of the covariance matrix.
        """
        sigma = math.sqrt(self.P[0, 0])
        return self.x[0] - 3 * sigma, self.x[0] + 3 * sigma
