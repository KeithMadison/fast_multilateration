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
        t = (sqrt_A2_beta2 * C * x0 - beta * C * z0 + beta * np.log(log_arg)) / (sqrt_A2_beta2 * C)

        exp_Kx = np.exp(K * x)
        exp_Kt = np.exp(K * t)
        log_term_num = 2 * term1 * np.exp(K * (t + x))
        log_term_den = beta**2 * B**2 * exp_Kx**2 - 2 * A * B * exp_Kt + exp_Kt**2
        log_term = log_term_num / log_term_den

        return (1 / C) * np.log(log_term)

    def _find_launch_angle(self, init_points: np.ndarray, term_points: np.ndarray, num_steps: int = 1000) -> np.ndarray:
        """Find the optimal launch angle."""
        x0 = np.hypot(init_points[:, 0], init_points[:, 1])
        x1 = np.hypot(term_points[:, 0], term_points[:, 1])
        
        # Coarse search with a fine search
        launch_angles = np.linspace(-np.pi, np.pi, num_steps)

        # Precompute z_coords for all launch angles
        z_coords = self._calculate_z_coord(x1[:, np.newaxis], launch_angles, x0[:, np.newaxis], init_points[:, 2, np.newaxis])

        term_z = term_points[:, 2][:, np.newaxis]
        objective_values = (z_coords - term_z)**2

        # Find the best angles
        best_indices = np.nanargmin(objective_values, axis=1)
        best_angles = launch_angles[best_indices]

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
    num_points = 50000
    init_points = np.random.uniform(low=-50, high=50, size=(num_points, 3))
    term_points = np.random.uniform(low=-50, high=50, size=(num_points, 3))

    import time

    start_time = time.time()
    transit_times = ray_tracer.transit_time(init_points, term_points)
    end_time = time.time()

    # Print results
    print("Transit times:", transit_times)
    print(f"Elapsed time: {end_time - start_time:.6f} seconds")
