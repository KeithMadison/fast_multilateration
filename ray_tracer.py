from dataclasses import dataclass
import warnings
import numpy as np

# Suppress warnings (sometimes there is overflow in _calculate_z_coord, etc.)
warnings.filterwarnings("ignore", category=RuntimeWarning)

@dataclass
class RayTracer:
    medium_model: np.ndarray
    SPEED_OF_LIGHT = 299792458  # Speed of light in m/s

    def _calculate_z_coord(self, x: np.ndarray, launch_angle: np.ndarray, x0: np.ndarray, z0: np.ndarray) -> np.ndarray:
        """Calculate the z-coordinate based on launch angle and other parameters."""
        A, B, C = self.medium_model

        exp_Cz0 = np.exp(C * z0)
        cos_launch_angle = np.cos(launch_angle)
        beta = (A - B * exp_Cz0) * cos_launch_angle
        
        sqrt_A2_beta2 = np.sqrt(A**2 - beta**2)
        K = C * sqrt_A2_beta2 / beta

        term1 = A**2 - beta**2
        term2 = A * B * exp_Cz0
        sqrt_term = np.sqrt(term1 + 2 * term2 + B**2 * exp_Cz0**2)

        # Precompute for efficiency
        log_arg = term1 + term2 + sqrt_A2_beta2 * sqrt_term
        t = x0 + (- C * z0 + np.log(log_arg)) / K

        exp_Kx = np.exp(K * x)
        exp_Kt = np.exp(K * t)
        log_term_num = 2 * term1 * exp_Kx * exp_Kt
        log_term_den = beta**2 * B**2 * exp_Kx**2 - 2 * A * B * exp_Kt + exp_Kt**2
        log_term = log_term_num / log_term_den

        return (1 / C) * np.log(log_term)

    def _find_launch_angle(self, init_points: np.ndarray, term_points: np.ndarray, num_steps: int = 350) -> np.ndarray:
        """Find the optimal launch angle."""
        # Project init_points and term_points into 2D while preserving z-coordinates
        init_proj = np.copy(init_points)
        term_proj = np.copy(term_points)

        for i in range(init_points.shape[0]):
            x0, y0, z0 = init_points[i]
            x1, y1, z1 = term_points[i]

            # Calculate angle theta
            delta_x = x1 - x0
            delta_y = y1 - y0
            theta = np.arctan2(delta_y, delta_x)  # Angle in radians

            # Projected x-coordinates
            init_proj[i, 0] = np.hypot(delta_x, delta_y) * np.cos(theta)
            term_proj[i, 0] = np.hypot(delta_x, delta_y) * np.cos(theta)
            init_proj[i, 1] = 0  # Set y-coordinate to 0 for projection
            term_proj[i, 1] = 0  # Set y-coordinate to 0 for projection

            # Keep z-coordinates unchanged
            init_proj[i, 2] = z0
            term_proj[i, 2] = z1

        # Calculate distances in the projected space
        x0 = np.hypot(init_proj[:, 0], init_proj[:, 1])
        x1 = np.hypot(term_proj[:, 0], term_proj[:, 1])
        
        # Coarse search followed by a fine search
        launch_angles = np.linspace(-np.pi, np.pi, num_steps)
        best_angles = np.zeros(init_points.shape[0])

        for i in range(init_points.shape[0]):
            objective_values = (self._calculate_z_coord(x1[i], launch_angles, x0[i], init_proj[i, 2]) - term_proj[i, 2])**2
            best_idx = np.nanargmin(objective_values)
            best_angle = launch_angles[best_idx]
            best_angles[i] = best_angle

        return best_angles

    def transit_time(self, init_points: np.ndarray, term_points: np.ndarray) -> np.ndarray:
        """Calculate the transit time."""
        A, B, C = self.medium_model

        # Vectorized launch angle search
        launch_angles = self._find_launch_angle(init_points, term_points)

        print(launch_angles)

        exp_Cz_init = np.exp(C * init_points[:, 2])
        beta = np.abs((A - B * exp_Cz_init) * np.cos(launch_angles))
        sqrt_A2_beta2 = np.sqrt(A**2 - beta**2)
        K = C * sqrt_A2_beta2 / beta

        def time_expression(z: np.ndarray, beta: np.ndarray, K: np.ndarray) -> np.ndarray:
            exp_Cz = np.exp(C * z)
            t = np.sqrt((A + B * exp_Cz - beta) / (A + B * exp_Cz + beta))
            alpha = np.sqrt((A - beta) / (A + beta))
            log_expr = np.log(np.abs((t - alpha) / (t + alpha)))
            return A * np.sqrt((C**2 + K**2) / (C**2 * K**2)) * log_expr

        time_diff = time_expression(term_points[:, 2], beta, K) - time_expression(init_points[:, 2], beta, K)

        return time_diff / self.SPEED_OF_LIGHT

# Example usage
if __name__ == "__main__":
    ray_tracer = RayTracer(np.array([1.78, 0.454, 0.0132]))

    # Generate random init_points and term_points
    num_points = 1000
    init_points = np.array([[8.4,7.03,-500]])
    term_points = np.array([[0,0,-250]])

    import time

    start_time = time.time()
    transit_times = ray_tracer.transit_time(init_points, term_points)
    end_time = time.time()

    # Print results
    print("Transit times:", transit_times)
    print(f"Elapsed time: {end_time - start_time:.6f} seconds")
