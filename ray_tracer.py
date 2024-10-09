from dataclasses import dataclass
import warnings
import numpy as np

# Suppress warnings (sometimes there is overflow in _calculate_z_coord, etc., none of which are concerning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

@dataclass
class RayTracer:
	medium_model: np.ndarray
	SPEED_OF_LIGHT = 299792458  # Speed of light in m/s

	def _calculate_z_coord(self, x: np.ndarray, launch_angle: np.ndarray, x0: np.ndarray, z0: np.ndarray) -> np.ndarray:
		"""Calculate the z-coordinate based on launch angle and other parameters."""
		A, B, C = self.medium_model

		exp_Cz0 = np.exp(C * z0)
		beta = (A - B * exp_Cz0) * np.cos(launch_angle)

		K = C * np.sqrt(A**2 - beta**2) / beta

		log_arg = (A**2 - beta**2) + (A * B * exp_Cz0) + np.sqrt((A**2 - beta**2) + 2 * (A * B * exp_Cz0) + B**2 * exp_Cz0**2) * np.sqrt(A**2 - beta**2)
		t = x0 + (-C * z0 + np.log(log_arg)) / K

		exp_Kx = np.exp(K * x)
		exp_Kt = np.exp(K * t)

		log_term_num = 2 * (A**2 - beta**2) * exp_Kx * exp_Kt
		log_term_den = beta**2 * B**2 * exp_Kx**2 - 2 * A * B * exp_Kt + exp_Kt**2

		return (1 / C) * np.log(log_term_num / log_term_den)

	def _find_launch_angle(self, init_points: np.ndarray, term_points: np.ndarray, num_steps: int = 1000) -> np.ndarray:
		"""Find the optimal launch angle."""
		# Calculate angle theta and hypotenuse for all points
		delta_x = term_points[:, 0] - init_points[:, 0]
		delta_y = term_points[:, 1] - init_points[:, 1]
		x1 = np.hypot(delta_x, delta_y)
		x0 = 0.0
 
		# Create launch angles and preallocate z-coordinates
		launch_angles = np.linspace(-np.pi, np.pi, num_steps)
		z_coords = np.array([self._calculate_z_coord(x1, angle, x0, init_points[:, 2]) for angle in launch_angles])

		# Calculate objective values and find best launch angle indices
		objective_values = (z_coords - term_points[:, 2].reshape(1, -1)) ** 2
		best_idx = np.argmin(objective_values, axis=0)

		return launch_angles[best_idx]

	def transit_time(self, init_points: np.ndarray, term_points: np.ndarray) -> np.ndarray:
		"""Calculate the transit time."""
		A, B, C = self.medium_model

		# Ensure terminal points are shallower than corresponding initial points
		mask = term_points[:, 2] <= init_points[:, 2]
		init_points[mask], term_points[mask] = term_points[mask].copy(), init_points[mask].copy()

		# Vectorized launch angle search
		launch_angles = self._find_launch_angle(init_points, term_points)

		exp_Cz_init = np.exp(C * init_points[:, 2], out=np.empty_like(init_points[:, 2]))
		beta = (A - B * exp_Cz_init) * np.cos(launch_angles)
		K = C * np.sqrt(A**2 - beta**2) / beta

		def time_expression(z: np.ndarray) -> np.ndarray:
			exp_Cz = np.exp(C * z, out=np.empty_like(z))
			t = np.sqrt((A + B * exp_Cz - beta) / (A + B * exp_Cz + beta))
			alpha = np.sqrt((A - beta) / (A + beta))
			log_expr = np.log(np.abs((t - alpha) / (t + alpha)))
			return A * np.sqrt((C**2 + K**2) / (C**2 * K**2)) * log_expr

		time_diff = time_expression(term_points[:, 2]) - time_expression(init_points[:, 2])

		return time_diff / self.SPEED_OF_LIGHT

import time

if __name__ == "__main__":
    ray_tracer = RayTracer(np.array([1.78, 0.454, 0.0132]))

    # Generate random init_points and term_points
    num_points = 10000
    # Generating random coordinates within a specified range
    init_points = np.random.uniform(low=-10.0, high=10.0, size=(num_points, 3))  # Change range as needed
    term_points = np.random.uniform(low=-10.0, high=10.0, size=(num_points, 3))  # Change range as needed

    start_time = time.time()
    transit_times = ray_tracer.transit_time(init_points, term_points)
    end_time = time.time()

    # Print results
    print("Transit times:", transit_times)
    print(f"Time taken to trace {num_points} rays: {end_time - start_time:.6f} seconds")
