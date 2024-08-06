import numpy as np
from scipy.optimize import minimize

class RayTracer:
    """A class to model and optimize ray tracing through an ice medium."""

    SPEED_OF_LIGHT = 299792458  # Speed of light in meters per second

    def __init__(self, ice_model: np.ndarray):
        """
        Initialize the RayTracer with a given ice model.

        Parameters:
        ice_model (np.ndarray): An array where the first element is the
                                refractive index at sea level, the second
                                is the attenuation coefficient, and the third
                                is the depth-dependent coefficient.
        """
        self.ice_model = ice_model

    def _refractive_index(self, z: float) -> float:
        """
        Compute the refractive index at depth z.

        Parameters:
        z (float): The depth in meters.

        Returns:
        float: The refractive index at depth z.
        """
        return self.ice_model[0] - self.ice_model[1] * np.exp(self.ice_model[2] * z)

    def _propagation_time(self, path: np.ndarray, x_start: float, x_end: float) -> float:
        """
        Calculate the total propagation time of a ray along a given path.

        Parameters:
        path (np.ndarray): The array of depth values representing the path.
        x_start (float): The starting distance from the origin.
        x_end (float): The ending distance from the origin.

        Returns:
        float: The total propagation time along the path.
        """
        z_points = np.concatenate(([path[0]], path, [path[-1]]))
        x_points = np.linspace(x_start, x_end, len(z_points))

        def integrand(i: int) -> float:
            """
            Compute the integrand for the propagation time calculation.

            Parameters:
            i (int): The index of the current segment.

            Returns:
            float: The integrand value for the segment.
            """
            dzdx = (z_points[i + 1] - z_points[i]) / (x_points[i + 1] - x_points[i])
            return (self._refractive_index((z_points[i + 1] + z_points[i]) / 2) / self.SPEED_OF_LIGHT) * np.sqrt(1 + dzdx**2)

        dx = x_points[1] - x_points[0]
        return np.sum([integrand(i) for i in range(len(z_points) - 1)]) * dx

    def find_optimal_path(self, start: np.ndarray, end: np.ndarray, num_points: int = 1000) -> tuple:
        """
        Find the optimal path that minimizes the propagation time from start to end.

        Parameters:
        start (np.ndarray): Array containing the starting coordinates [x, y, z].
        end (np.ndarray): Array containing the ending coordinates [x, y, z].
        num_points (int): Number of points to use in the initial path (default is 100).

        Returns:
        tuple: A tuple containing:
            - The optimized path (np.ndarray) with depth values.
            - The total propagation time (float) along the optimized path.
        """
        x_start = np.hypot(start[0], start[1])
        x_end = np.hypot(end[0], end[1])

        #z_start, z_end = start[2], end[2]
        #z_min, z_max = min(z_start, z_end), max(z_start, z_end)
        #random_points = np.random.uniform(z_min, z_max, num_points - 2)
        #init_path = np.sort(np.concatenate(([z_start], random_points, [z_end])))

        init_path = np.linspace(start[2], end[2], num_points)

        bounds = [(min(start[2], end[2]) - 10, max(start[2], end[2]) + 10)] * num_points

        # Define the objective function as a lambda
        objective = lambda path: self._propagation_time(path, x_start, x_end)

        options = {
            'maxiter': 2500,
            'disp': False,  # Do not display optimization progress
            'ftol': 1e-30
        }

        # Try a different optimization method
        result = minimize(objective, init_path, bounds=bounds, method='L-BFGS-B', options=options)

        optimized_path = np.concatenate(([start[2]], result.x, [end[2]]))
        total_time = self._propagation_time(result.x, x_start, x_end)

        return optimized_path, total_time
