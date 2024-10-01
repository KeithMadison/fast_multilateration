from dataclasses import dataclass
import numpy as np
from scipy.optimize import differential_evolution
from ray_tracer import RayTracer
import matplotlib.pyplot as plt
from random import randrange

@dataclass
class Optimizer:
    ice_model: np.ndarray  # Parameters used by my RayTracer class to model complex medium
    antenna_coordinates: np.ndarray

    def __post_init__(self):
        self.best_solution = None
        self.best_objective_value = float('inf')
        self.ray_tracer = RayTracer(self.ice_model)
        self.smoothed_objective_value = None  # For storing the smoothed value
        self.alpha = 0.025  # Smoothing factor for EMA (tune as necessary)

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
        smoothed_errors = errors
        total_error = np.nansum(np.abs(smoothed_errors))

        # Update the best solution found
        if total_error < self.best_objective_value:
            self.best_objective_value = total_error
            self.best_solution = coords

        # Apply EMA smoothing to the objective value
        objective_value = np.log10(total_error + 1)
        if self.smoothed_objective_value is None:
            self.smoothed_objective_value = objective_value  # Initialize with the first value
        else:
            self.smoothed_objective_value = (
                self.alpha * objective_value + (1 - self.alpha) * self.smoothed_objective_value
            )

        # Return the smoothed objective value
        return self.smoothed_objective_value*2

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

    def plot_objective(self, measured_times: np.ndarray, x: float, y: float, z_true: float):
        """Plot the objective function values as a function of z."""
        z_values = np.linspace(-1000, 0, 1000)  # Adjust range as necessary
        objective_values = []

        for z in z_values:
            coords = np.array([x, y, z])
            objective_value = self._objective(coords, measured_times)
            objective_values.append(objective_value)

        plt.figure(figsize=(10, 6))
        plt.plot(z_values, objective_values, label='Objective Function', color='blue')
        plt.scatter(z_true, self._objective(np.array([x, y, z_true]), measured_times), color='red', label='True z Value', zorder=5)
        plt.xlabel('z Value')
        plt.ylabel('Objective Function Value (log scale)')
        plt.title('Objective Function as a Function of z')
        plt.legend()
        plt.grid()
        plt.show()

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
        optimizer.plot_objective(measured_times, x, y, z)
    except ValueError as e:
        print(e)
