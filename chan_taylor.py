import numpy as np
from dataclasses import dataclass

@dataclass
class ChanTaylorVertexer:

	def find(self, node_positions, distance_differences):
		# Noise covariance matrix
		noise_covariance = np.eye(len(node_positions[0]) - 1)
		
		squared_distances = np.sum(node_positions[:, 1:] ** 2, axis=0)
	
		Ga_matrix = np.hstack((-base_station_positions[:, 1:].T,
		(-distance_differences).reshape(-1, 1)))

		h_vector = 0.5 * (distance_differences ** 2 - squared_distances)

		Ga_pseudo_inverse_Q = np.linalg.pinv(Ga_matrix.T @ np.linalg.pinv(noise_covariance) @ Ga_matrix)
		initial_estimate = Ga_pseudo_inverse_Q @ Ga_matrix.T @ np.linalg.pinv(noise_covariance) @ h_vector

		distances_from_estimate = np.eye(base_station_count - 1)
		for i in range(base_station_count - 1):
			distances_from_estimate[i, i] = np.sqrt(
				(base_station_positions[0, i + 1] - initial_estimate[0]) ** 2 +
				(base_station_positions[1, i + 1] - initial_estimate[1]) ** 2 +
				(base_station_positions[2, i + 1] - initial_estimate[2]) ** 2
			)

		FI_matrix = distances_from_estimate @ noise_covariance @ distances_from_estimate
		FI_pseudo_inverse_Ga = np.linalg.pinv(Ga_matrix.T @ np.linalg.pinv(FI_matrix) @ Ga_matrix)
		refined_estimate = FI_pseudo_inverse_Ga @ Ga_matrix.T @ np.linalg.pinv(FI_matrix) @ h_vector

		covariance_refined_estimate = np.linalg.pinv(Ga_matrix.T @ np.linalg.pinv(FI_matrix) @ Ga_matrix)
		sB_matrix = np.eye(4)
		np.fill_diagonal(sB_matrix, refined_estimate)
		sFI_matrix = 4 * sB_matrix @ covariance_refined_estimate @ sB_matrix

		sGa_matrix = np.array([
			[1, 0, 0],
			[0, 1, 0],
			[0, 0, 1],
			[1, 1, 1]
		])

		sh_vector = np.array([refined_estimate[0] ** 2, refined_estimate[1] ** 2, refined_estimate[2] ** 2, refined_estimate[3] ** 2])
		sFI_pseudo_inverse_sGa = np.linalg.pinv(sGa_matrix.T @ np.linalg.pinv(sFI_matrix) @ sGa_matrix)
		final_estimate_squared = sFI_pseudo_inverse_sGa @ sGa_matrix.T @ np.linalg.pinv(sFI_matrix) @ sh_vector
		estimated_position = np.sqrt(np.abs(final_estimate_squared))
		estimated_position[2] = -estimated_position[2]
		
		return estimated_position
