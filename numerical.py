from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, shgo, differential_evolution
from ray_tracer import RayTracer
from random import randrange

@dataclass
class Optimizer:
    ice_model: np.ndarray
    antenna_coordinates: np.ndarray

    def __post_init__(self):
        self.ray_tracer = RayTracer(self.ice_model)

    def _objective(self, coords: np.ndarray, measured_times: np.ndarray) -> float:
        """Objective function to minimize the error between measured and predicted transit times."""
        x, y, z = coords
        predicted_times = np.array([
            self.ray_tracer.transit_time([x, y, z], [nx, ny, nz])
            for nx, ny, nz in self.antenna_coordinates
        ])

        # Compute time difference matrices
        measured_time_diff = np.abs(measured_times[:, None] - measured_times)
        predicted_time_diff = np.abs(predicted_times[:, None] - predicted_times)

        # Calculate error matrix and fill the diagonal with zeros
        error_matrix = np.abs(measured_time_diff - predicted_time_diff)
        np.fill_diagonal(error_matrix, 0)

        # Return the maximum error
        return np.nanmax(error_matrix)

    def solve(self, measured_times: np.ndarray, initial_guess: np.ndarray) -> np.ndarray:
        """Solve the optimization problem to find the coordinates that minimize the objective function."""
        bounds = [(-1000, 1000), (-1000, 1000), (-1000, 0)]

        # Use differential evolution to minimize the objective function
        result = differential_evolution(
            self._objective,
            bounds=bounds,
            args=(measured_times,),
            strategy='best1bin',  # The strategy to use; can be customized
            maxiter=5000,  # Maximum number of iterations
            tol=1e-7,  # Tolerance for convergence
            popsize=15,  # Population size
            mutation=(0.5, 1),  # Mutation factor
            recombination=0.7,  # Recombination constant
            disp=True  # Set to True to display convergence messages
        )

        if result.success:
            return result.x  # Return the optimized coordinates
        else:
            raise ValueError("Optimization failed: " + result.message)

# Example usage
if __name__ == "__main__":
    ice_model = np.array([1.78, 0.454, 0.0132])

    ray_tracer = RayTracer(ice_model)

    antenna_coordinates = np.array([[randrange(-100,0) for _ in range(3)] for _ in range(10)])
    x, y, z = [randrange(-1000,0) for _ in range(3)]
    measured_times = np.array([ray_tracer.transit_time([x, y, z], [nx, ny, nz]) + 10 for nx, ny, nz in antenna_coordinates])

    optimizer = Optimizer(ice_model, antenna_coordinates)
    initial_guess = np.array([0.0, 0.0, 0.0])  # Example initial guess for (x, y, z)

    try:
        optimal_coordinates = optimizer.solve(measured_times, initial_guess)
        print("Optimal Coordinates (x, y, z):", optimal_coordinates)
        print("True Coordinates (x, y, z):", x, y, z)
    except ValueError as e:
        print(e)
