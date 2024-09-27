from dataclasses import dataclass
import numpy as np

@dataclass
class RayTracer:
    ice_model: np.ndarray
    SPEED_OF_LIGHT = 299792458  # Speed of light in m/s

    def _calculate_z_coord(self, x: float, launch_angle: float, x0: float, z0: float) -> float:
        """Calculate the z-coordinate based on launch angle and other parameters."""
        A, B, C = self.ice_model

        exp_Cz0 = np.exp(C * z0)
        beta = (A - B * exp_Cz0) * np.cos(launch_angle)
        
        sqrt_A2_beta2 = np.sqrt(A**2 - beta**2)
        K = C * sqrt_A2_beta2 / beta

        term1 = A**2 - beta**2
        term2 = A * B * exp_Cz0
        sqrt_term = np.sqrt(term1 + 2 * term2 + B**2 * exp_Cz0**2)

        # Calculate t
        t = (sqrt_A2_beta2 * C * x0 - beta * C * z0 +
             beta * np.log(term1 + term2 + sqrt_A2_beta2 * sqrt_term)) / (sqrt_A2_beta2 * C)

        # Calculate log_term for return value
        exp_Kx = np.exp(K * x)
        exp_Kt = np.exp(K * t)
        log_term = (2 * term1 * np.exp(K * (t + x))) / (beta**2 * B**2 * exp_Kx**2 - 2 * A * B * exp_Kt + exp_Kt**2)

        return (1 / C) * np.log(log_term)

    def _find_launch_angle(self, init_point: np.ndarray, term_point: np.ndarray, num_steps: int = 10000) -> float:
        """Find the optimal launch angle that minimizes the difference in z-coordinates."""
        x0 = np.hypot(init_point[0], init_point[1])
        x1 = np.hypot(term_point[0], term_point[1])

        # Generate launch angles and calculate objective values
        launch_angles = np.linspace(-np.pi, np.pi, num_steps)
        objective_values = (self._calculate_z_coord(x1, launch_angles, x0, init_point[2]) - term_point[2])**2

        # Return the launch angle that minimizes the objective value
        return launch_angles[np.argmin(objective_values)]

    def transit_time(self, init_point: np.ndarray, term_point: np.ndarray) -> float:
        """Calculate the transit time between two points in the medium."""
        A, B, C = self.ice_model
        launch_angle = self._find_launch_angle(init_point, term_point)

        beta = np.abs((A - B * np.exp(C * init_point[2])) * np.cos(launch_angle))
        K = C * np.sqrt(A**2 - beta**2) / beta

        def time_expression(z: float) -> float:
            exp_Cz = np.exp(C * z)
            t = np.sqrt((A + B * exp_Cz - beta) / (A + B * exp_Cz + beta))
            alpha = np.sqrt((A - beta) / (A + beta))
            log_expr = np.log(np.abs((t - alpha) / (t + alpha)))
            return A * np.sqrt((C**2 + K**2) / (C**2 * K**2)) * log_expr

        # Calculate and return the transit time
        return (time_expression(term_point[2]) - time_expression(init_point[2])) / self.SPEED_OF_LIGHT

# Example usage
if __name__ == "__main__":
    ray_tracer = RayTracer(np.array([1.78, 0.454, 0.0132]))

    import time

    start_time = time.time()  # Start timing
    transit_time = ray_tracer.transit_time(np.array([-100, -20, -500]), np.array([-30, -10, -250]))
    end_time = time.time()    # End timing

    # Print results
    print("Transit time:", transit_time)
    print(f"Elapsed time: {end_time - start_time:.6f} seconds")
