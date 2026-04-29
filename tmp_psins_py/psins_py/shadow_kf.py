import numpy as np

class EnhancedKalmanFilter:
    """
    A unified 15-state (or custom dimension) Extended Kalman Filter that supports 
    standard high-frequency physical updates and asynchronous low-frequency shadow updates.
    """
    def __init__(self, initial_state: np.ndarray, initial_P: np.ndarray, Q: np.ndarray, 
                 H_speed: np.ndarray, R_speed: np.ndarray):
        self.X = initial_state
        self.P = initial_P
        self.Q = Q
        
        # Standard observation matrices for physical state
        self.H_speed = H_speed
        self.R_speed = R_speed

    def time_update(self, Phi: np.ndarray, Gamma: np.ndarray = None):
        """Standard time update (Prediction Step)"""
        self.X = Phi @ self.X
        if Gamma is None:
            self.P = Phi @ self.P @ Phi.T + self.Q
        else:
            self.P = Phi @ self.P @ Phi.T + Gamma @ self.Q @ Gamma.T

    def physical_update(self, z_speed: np.ndarray) -> np.ndarray:
        """
        High-frequency physical loop update (Assumes velocity error v=0)
        Returns:
            innovation (np.ndarray): The residual of this update, used for LLM feature extraction.
        """
        z_pred = self.H_speed @ self.X
        innovation = z_speed - z_pred
        
        S = self.H_speed @ self.P @ self.H_speed.T + self.R_speed
        
        # Solving Kalman Gain dynamically protecting against singularity
        try:
            K = self.P @ self.H_speed.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ self.H_speed.T @ np.linalg.pinv(S)
        
        self.X = self.X + K @ innovation
        self.P = (np.eye(len(self.X)) - K @ self.H_speed) @ self.P
        
        # Enforce P matrix symmetry
        self.P = (self.P + self.P.T) * 0.5
        return innovation

    def shadow_update(self, z_llm: float, target_idx: int, R_llm: float):
        """
        Asynchronous Shadow Injection capability (Low-Frequency).
        Instead of recomputing the full coupled H_total, this uses sequential filtering principle,
        dynamically applying a scalar LLM prediction observation to the respective state axis.
        
        Args:
            z_llm: The predicted absolute error value (e.g. LLM infers installation angle error is X)
            target_idx: The state index to which z_llm applies
            R_llm: The mapped observation noise variance based on LLM Confidence
        """
        # Formulate sparse H_llm based strictly targeting the designated index
        H_llm = np.zeros((1, len(self.X)))
        H_llm[0, target_idx] = 1.0
        
        z_pred = H_llm @ self.X
        innovation = z_llm - z_pred[0]
        
        S = H_llm @ self.P @ H_llm.T + R_llm
        S_scalar = S[0, 0] # It's a 1x1 scalar
        
        # Optional basic Chi-Square bounds-check (reject if extremely out of distribution)
        # chi2 = (innovation**2) / S_scalar
        # if chi2 > 9.0: # Roughly 3-sigma
        #    return False
            
        K = self.P @ H_llm.T * (1.0 / S_scalar) 
        
        self.X = self.X + K.flatten() * innovation
        self.P = (np.eye(len(self.X)) - K @ H_llm) @ self.P
        
        self.P = (self.P + self.P.T) * 0.5
        return True
