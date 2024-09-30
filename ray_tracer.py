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

    def _find_launch_angle(self, init_points: np.ndarray, term_points: np.ndarray, num_steps: int = 1000) -> np.ndarray:
        """Find the optimal launch angle."""
        # Project init_points and term_points into 2D while preserving z-coordinates
        init_proj = np.copy(init_points)
        term_proj = np.copy(term_points)

        # Calculate differences
        delta_x = term_points[:, 0] - init_points[:, 0]
        delta_y = term_points[:, 1] - init_points[:, 1]

        # Calculate angle theta for all points
        theta = np.arctan2(delta_y, delta_x)  # Angle in radians

        # Projected x-coordinates
        hypotenuse = np.hypot(delta_x, delta_y)
        init_proj[:, 0] = hypotenuse * np.cos(theta)
        term_proj[:, 0] = hypotenuse * np.cos(theta)
        init_proj[:, 1] = 0  # Set y-coordinate to 0 for projection
        term_proj[:, 1] = 0  # Set y-coordinate to 0 for projection

        # Keep z-coordinates unchanged
        init_proj[:, 2] = init_points[:, 2]
        term_proj[:, 2] = term_points[:, 2]

        # Calculate distances in the projected space
        x0 = np.hypot(init_proj[:, 0], init_proj[:, 1])
        x1 = np.hypot(term_proj[:, 0], term_proj[:, 1])

        # Create launch angles
        launch_angles = np.linspace(-np.pi, np.pi, num_steps)

        # Preallocate array for z-coordinates for all launch angles
        z_coords = np.zeros((num_steps, init_points.shape[0]))

        # Vectorized calculation of z-coordinates for all launch angles
        for i, angle in enumerate(launch_angles):
            z_coords[i, :] = self._calculate_z_coord(x1, angle, x0, init_proj[:, 2])

        # Calculate objective values for all points and launch angles
        objective_values = (z_coords - term_proj[:, 2].reshape(1, -1))**2

        # Find the best launch angle index for each point
        best_idx = np.argmin(objective_values, axis=0)

        # Get best angles directly
        best_angles = launch_angles[best_idx]

        return best_angles

    def transit_time(self, init_points: np.ndarray, term_points: np.ndarray) -> np.ndarray:
        """Calculate the transit time."""
        A, B, C = self.medium_model

        # Vectorized launch angle search
        launch_angles = self._find_launch_angle(init_points, term_points)

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
