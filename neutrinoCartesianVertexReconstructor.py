import numpy as np
from scipy.optimize import differential_evolution
from NuRadioReco.modules.neutrinoVertexReconstructor.ray_tracer import RayTracer
# Don't forget to inherit from modules.base

class neutrinoCartesianVertexReconstructor:
	def __init__(self):
		self.__detector = None
		self.__channel_ids = None
		self.__station_id = None
		self.__output_path = None
		self.__channel_positions = []
		self._best_objective_value = float('inf')
		self._smoothed_objective_value = None
		self._best_solution = None
		self._alpha = 0.025 # Smoothing factor for EMA
		self.ray_tracer = RayTracer(np.array([1.78, 0.454, 0.0132]))

	def _objective(self, vertex_coord_guess, measured_times):
		x, y, z = vertex_coord_guess

		# Compute predicted arrival times for vertex coordinate guess
		predicted_times = self.ray_tracer.transit_time(
			np.full_like(self.__channel_positions, vertex_coord_guess),
			self.__channel_positions
		)

		# Construct pairwise arrival time difference matrices
		measured_time_diff = np.abs(measured_times[:, np.newaxis] - measured_times)
		predicted_time_diff = np.abs(predicted_times[:, np.newaxis] - predicted_times)

		# Create a mask for the upper triangular part (excluding the diagonal)
		# The matrix is antisymmetric
		mask = np.triu_indices(measured_time_diff.shape[0], k=1)

		errors = measured_time_diff[mask] - predicted_time_diff[mask]
		total_error = np.nansum(np.abs(errors))

		# Apply Exponential Moving Average (EMA) smoothing to the objective value
		if total_error < self._best_objective_value:
			self._best_objective_value = total_error
			self._best_solution = vertex_coord_guess

		objective_value = np.log10(total_error + 1)

		if self._smoothed_objective_value is None:
			self._smoothed_objective_value = objective_value
		else:
			self._smoothed_objective_value = (
				self._alpha * objective_value + (1 - self._alpha) * self._smoothed_objective_value
			)

		return self._smoothed_objective_value * 2

	def _compute_arrival_times(self, station):
		# Will compute the arrival times relative to reference channel 0 
		reference_channel = station.get_channel(self.__channel_ids[0])
		reference_trace = reference_channel.get_trace()
		
		# This seems wrong sometimes? Not certain what the units are
		sampling_rate = reference_channel.get_sampling_rate()

		measured_times = [(np.argmax(np.correlate(reference_trace, station.get_channel(cid).get_trace(), mode='full'))
				 - len(reference_trace) + 1) / sampling_rate for cid in self.__channel_ids]

		return np.array(measured_times)

	def begin(self, station_id, channel_ids, detector, output_path=None):
		self.__detector = detector
		self.__channel_ids = channel_ids
		self.__station_id = station_id
		self.__output_path = output_path

		for cid in channel_ids:
			channel_position = detector.get_relative_position(station_id, cid, mode='channel')
			self.__channel_positions.append(channel_position)

		self.__channel_positions = np.array(self.__channel_positions)

	def run(self, event, station):
		bounds = [(-1000, 1000), (-1000, 1000), (-1000, 0)]
		measured_times = self._compute_arrival_times(station)

		result = differential_evolution(
			self._objective,
			bounds=bounds,
			args=(measured_times,),
			strategy='rand1bin',
			maxiter=500,
			tol=1e-6,
			popsize=5,
			mutation=(1.59, 1.99),
			recombination=0.7,
#			disp=True,
#			workers=-1
		)

		print(self._best_solution)

		if result.success or self._best_solution is not None:
			return self._best_solution
		else:
			raise ValueError("Optimization failed without finding a valid vertex solution.")
