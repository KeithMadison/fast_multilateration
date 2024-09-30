from dataclasses import dataclass
import numpy as np
from fast_ray_tracer import RayTracer
from random import randrange
from concurrent.futures import ProcessPoolExecutor, as_completed

@dataclass
class Optimizer:
    ice_model: np.ndarray  # Parameters used by my RayTracer class to model complex medium
    antenna_coordinates: np.ndarray

    def __post_init__(self):
        self.ray_tracer = RayTracer(self.ice_model)

    def _objective(self, coords: np.ndarray, measured_times: np.ndarray) -> float:
        """Objective function to minimize the error between measured and predicted transit times."""
        # Broadcast coords to match the shape of antenna_coordinates
        predicted_times = self.ray_tracer.transit_time(np.array([coords] * len(self.antenna_coordinates)), self.antenna_coordinates)

        # Compute time difference matrices
        measured_time_diff = np.abs(measured_times[:, None] - measured_times)
        predicted_time_diff = np.abs(predicted_times[:, None] - predicted_times)

        # Calculate error matrix and fill the diagonal with zeros
        error_matrix = np.abs(measured_time_diff - predicted_time_diff)
        np.fill_diagonal(error_matrix, 0)

        # Return the maximum error
        return np.nanmax(error_matrix)

    def _random_grid_search(self, measured_times: np.ndarray, num_samples: int) -> tuple:
        """Perform a random grid search and return the best coordinates and the associated error."""
        bounds = np.array([[-1000, 1000], [-1000, 1000], [-1000, 0]])
        
        # Generate random coordinates in a vectorized way
        random_coords = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_samples, 3))
        
        # Evaluate the objective function for all random coordinates
        errors = np.array([self._objective(coords, measured_times) for coords in random_coords])

        # Find the index of the best coordinates
        best_index = np.argmin(errors)
        best_coords = random_coords[best_index]
        min_error = errors[best_index]
        
        return best_coords, min_error

    def solve(self, measured_times: np.ndarray, num_samples: int, num_workers: int) -> np.ndarray:
        """Solve the optimization problem to find coordinates minimizing the objective function."""
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for _ in range(num_workers):
                futures.append(executor.submit(self._random_grid_search, measured_times, num_samples))
            
            best_coords = None
            min_error = float('inf')
            
            for future in as_completed(futures):
                coords, error = future.result()
                if error < min_error:
                    min_error = error
                    best_coords = coords

        return best_coords

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
        np.random.normal(ray_tracer.transit_time(np.array([[x, y, z]]), np.array([[nx, ny, nz]]))[0] + 10, 1e-12)
        for nx, ny, nz in antenna_coordinates
    ])

    # Initialize the optimizer
    optimizer = Optimizer(ice_model, antenna_coordinates)

    # Attempt to find optimal coordinates
    num_samples_per_worker = 10000  # Number of random samples per worker
    num_workers = 10  # Number of parallel workers

    optimal_coordinates = optimizer.solve(measured_times, num_samples_per_worker, num_workers)
    print("Optimal Coordinates (x, y, z):", optimal_coordinates)
    print("True Coordinates (x, y, z):", x, y, z)

