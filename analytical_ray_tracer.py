import numpy as np
from dataclasses import dataclass

@dataclass
class AnalyticalRayTracer:
        ice_model: np.ndarray

        SPEED_OF_LIGHT = 299792458

        def _find_gamma():
                return 1

        def _find_delta():
                return 1

        def propagation_time(self, init_point, term_point):
                sqrt_C = np.sqrt(C)
                u_Li = np.exp(sqrt_C * L_i)

                term = 2 / np.sqrt(4 * gamma * delta - A**2)
                arg_Li = (2 * gamma * u_Li + A) * term
                arg_0 = (2 * gamma + A) * term

                return (1 / (c * sqrt_C)) * (term * (np.arctan(arg_Li) - np.arctan(arg_0)))

        def x_position(self, z_position, init_point, term_point):
                assert init_point[2] <= z_position <= terminal_point[0], \
                        f"z_position {z_position} is not between the initial z-coordinate {init_point[2]} and the terminal z-coordinate {term_point[0]}"

                A, B, C = self.ice_model

                numerator = A * gamma - B * np.exp(C * z_position) - np.sqrt(B**2 * np.exp(2 * C * z_position) - 4 * gamma * delta)
                denominator = gamma * np.sqrt(4 * gamma * delta - A**2)

                alpha = init_point[0]

                return alpha + (2 * beta / np.sqrt(C / (4 * gamma * delta - A**2))) * np.arctan(numerator / denominator)

myRayTracer = AnalyticalRayTracer(np.array([1.78, 1.78-1.323, 0.202]))
myRayTracer.x_position(-1500, np.array([50,0,-1000]), np.array([0,0,-10]))
