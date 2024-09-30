from dataclasses import dataclass
import numpy as np
from scipy.optimize import differential_evolution
from ray_tracer import RayTracer
from scipy.ndimage import gaussian_filter1d
from random import randrange

@dataclass
class Optimizer:
    ice_model: np.ndarray  # Parameters used by my RayTracer class to model complex medium
    antenna_coordinates: np.ndarray

    def __post_init__(self):
        self.best_solution = None
        self.best_objective_value = float('inf')
        self.ray_tracer = RayTracer(self.ice_model)

    def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        """Create a Gaussian kernel."""
        kernel = np.fromfunction(
            lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * 
                       np.exp(-0.5 * ((x - (size - 1) / 2) / sigma) ** 2),
            (size,)
        )
        return kernel / np.sum(kernel)  # Normalize the kernel

    def _objective(self, coords: np.ndarray, measured_times: np.ndarray) -> float:
        """Objective function to minimize the error between measured and predicted transit times."""
        x, y, z = coords
        antenna_coords = self.antenna_coordinates

        # Use in-place operations for array manipulation
        num_antennas = antenna_coords.shape[0]
    
        predicted_times = self.ray_tracer.transit_time(
            np.full_like(antenna_coords, coords), antenna_coords
        )

        # Precompute and reuse arrays for time differences
        measured_diff = np.abs(measured_times[:, np.newaxis] - measured_times)
        predicted_diff = np.abs(predicted_times[:, np.newaxis] - predicted_times)

        # Create a mask for the upper triangular part (excluding the diagonal)
        mask = np.triu_indices(measured_diff.shape[0], k=1)

        errors = measured_diff[mask] - predicted_diff[mask]
        sigma = 2.0
        smoothed_errors = gaussian_filter1d(errors, sigma=sigma)
        total_error = np.nansum(np.abs(smoothed_errors))

        # Update the best solution found
        if total_error < self.best_objective_value:
            self.best_objective_value = total_error
            self.best_solution = coords

        return np.log10(total_error + 1)

    def solve(self, measured_times: np.ndarray) -> np.ndarray:
        """Solve the optimization problem to find coordinates minimizing the objective function."""
        bounds = [(-1000, 1000), (-1000, 1000), (-1000, 0)]

        # Use differential evolution to minimize the objective function
        result = differential_evolution(
            self._objective,
            bounds=bounds,
            args=(measured_times,),
            strategy='rand1bin',  # Strategy for optimization
            maxiter=500,  # Maximum number of iterations
            tol=1e-6,  # Tolerance for convergence
            popsize=5,  # Population size
            mutation=(1.59, 1.99),  # Mutation factor
            recombination=0.7,  # Recombination constant
            workers=-1,
            disp=True  # Display convergence messages
        )

        # Return the best solution found, even if maxiter was reached
        if result.success or self.best_solution is not None:
            return self.best_solution, self.best_objective_value
        else:
            raise ValueError("Optimization failed without finding a valid solution.")

# Example usage
if __name__ == "__main__":
    # Define the ice model
    ice_model = np.array([1.78, 0.454, 0.0132])
    
    # Initialize RayTracer with the ice model
    ray_tracer = RayTracer(ice_model)

    # Generate random antenna coordinates
    antenna_coordinates = np.array([[randrange(-100, 0) for _ in range(3)] for _ in range(10)])

    # Generate random true coordinates and measured transit times
    x, y, z = [randrange(-1000, 0) for _ in range(3)]
    measured_times = np.array([
        np.random.normal(ray_tracer.transit_time(np.array([[x, y, z]]), np.array([[nx, ny, nz]]))[0] + 10, 0)
        for nx, ny, nz in antenna_coordinates
    ])

    # Initialize the optimizer
    optimizer = Optimizer(ice_model, antenna_coordinates)

    # Attempt to find optimal coordinates
    try:
        optimal_coordinates = optimizer.solve(measured_times)
        print("Optimal Coordinates (x, y, z):", optimal_coordinates)
        print("True Coordinates (x, y, z):", x, y, z)
    except ValueError as e:
        print(e)
