from dataclasses import dataclass
import warnings
import numpy as np

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

@dataclass
class RayTracer:
    ice_model: np.ndarray
    SPEED_OF_LIGHT = 299792458  # Speed of light in m/s

    def _calculate_z_coord(self, x: np.ndarray, launch_angle: np.ndarray, x0: np.ndarray, z0: np.ndarray) -> np.ndarray:
        """Calculate the z-coordinate based on launch angle and other parameters."""
        A, B, C = self.ice_model

        exp_Cz0 = np.exp(C * z0)
        cos_launch_angle = np.cos(launch_angle)  # Compute cosine once
        beta = (A - B * exp_Cz0) * cos_launch_angle
        
        sqrt_A2_beta2 = np.sqrt(A**2 - beta**2)
        K = C * sqrt_A2_beta2 / beta

        term1 = A**2 - beta**2
        term2 = A * B * exp_Cz0
        sqrt_term = np.sqrt(term1 + 2 * term2 + B**2 * exp_Cz0**2)

        # Precompute terms for efficiency
        log_arg = term1 + term2 + sqrt_A2_beta2 * sqrt_term

        # Calculate t
        t = (sqrt_A2_beta2 * C * x0 - beta * C * z0 + beta * np.log(log_arg)) / (sqrt_A2_beta2 * C)

        # Calculate log_term for return value
        exp_Kx = np.exp(K * x)
        exp_Kt = np.exp(K * t)
        log_term_num = 2 * term1 * np.exp(K * (t + x))
        log_term_den = beta**2 * B**2 * exp_Kx**2 - 2 * A * B * exp_Kt + exp_Kt**2
        log_term = log_term_num / log_term_den

        return (1 / C) * np.log(log_term)

    def _find_launch_angle(self, init_points: np.ndarray, term_points: np.ndarray, num_steps: int = 5000) -> np.ndarray:
        """Find the optimal launch angle that minimizes the difference in z-coordinates."""
        x0 = np.hypot(init_points[:, 0], init_points[:, 1])
        x1 = np.hypot(term_points[:, 0], term_points[:, 1])

        # Generate fewer launch angles to reduce computation
        launch_angles = np.linspace(-np.pi, np.pi, num_steps)
        
        # Initialize a list to hold the best launch angles for each point
        best_angles = np.zeros(init_points.shape[0])

        # For each pair of init and term points, calculate the best launch angle
        for i in range(init_points.shape[0]):
            # Use squared error to compare z-coordinates
            objective_values = (self._calculate_z_coord(x1[i], launch_angles, x0[i], init_points[i, 2]) - term_points[i, 2])**2
            best_angles[i] = launch_angles[np.nanargmin(objective_values)]

        return best_angles

    def transit_time(self, init_points: np.ndarray, term_points: np.ndarray) -> np.ndarray:
        """Calculate the transit time between multiple points in the medium."""
        A, B, C = self.ice_model

        # Vectorized computation of launch angles for all init_points and term_points
        launch_angles = self._find_launch_angle(init_points, term_points)

        # Precompute beta for all points at once
        exp_Cz_init = np.exp(C * init_points[:, 2])
        beta = np.abs((A - B * exp_Cz_init) * np.cos(launch_angles))

        # Precompute constants K for all points
        sqrt_A2_beta2 = np.sqrt(A**2 - beta**2)
        K = C * sqrt_A2_beta2 / beta

        def time_expression(z: np.ndarray, beta: np.ndarray, K: np.ndarray) -> np.ndarray:
            exp_Cz = np.exp(C * z)
            t = np.sqrt((A + B * exp_Cz - beta) / (A + B * exp_Cz + beta))
            alpha = np.sqrt((A - beta) / (A + beta))
            log_expr = np.log(np.abs((t - alpha) / (t + alpha)))
            return A * np.sqrt((C**2 + K**2) / (C**2 * K**2)) * log_expr

        # Compute transit time in a vectorized way
        time_diff = time_expression(term_points[:, 2], beta, K) - time_expression(init_points[:, 2], beta, K)

        return time_diff / self.SPEED_OF_LIGHT

# Example usage
if __name__ == "__main__":
    ray_tracer = RayTracer(np.array([1.78, 0.454, 0.0132]))

    # Generate random init_points and term_points
    num_points = 100000 #* 1000 * 1000  # Number of points to generate
    init_points = np.random.uniform(low=-50, high=50, size=(num_points, 3))  # Random initial points
    term_points = np.random.uniform(low=-50, high=50, size=(num_points, 3))  # Random terminal points

    import time

    start_time = time.time()  # Start timing
    transit_times = ray_tracer.transit_time(init_points, term_points)
    end_time = time.time()    # End timing

    # Print results
    print("Transit times:", transit_times)
    print(f"Elapsed time: {end_time - start_time:.6f} seconds")
