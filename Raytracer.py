import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass

@dataclass
class Raytracer:
    """
    A class for computing the propagation time of a ray
    through an anisotropic ice medium based on a given ice model.

    Attributes
    ----------
    ice_model : list
        A list of parameters defining the simple exponential ice model, of
        the form [A, B, C]
    """

    ice_model: list

    def _n(self, z):
        """
        Compute the refractive index of the ice at a given depth.

        Parameters
        ----------
        z : float
            The depth in the ice.

        Returns
        -------
        float
            The refractive index at depth z.
        """
        return self.ice_model[0] - self.ice_model[1] * np.exp(self.ice_model[2] * z)

    def _euler_lagrange_equations(self, t, y):
        """
        Compute the Euler-Lagrange equations for the ray propagation.

        Parameters
        ----------
        t : float
            The current time (not used in this function).
        y : list of floats
            The state vector [x, y, z, dx/dt, dy/dt, dz/dt], where (x, y, z)
            represents the position and (dx/dt, dy/dt, dz/dt) represents the
            velocity components.

        Returns
        -------
        list of floats
            The derivatives [dx/dt, dy/dt, dz/dt, dL/dx, dL/dy, dL/dz] for the
            position and Lagrangian components.
        """
        x, y, z, dxdt, dydt, dzdt = y

        # Compute the propagation speed
        propagation_speed = np.sqrt(dxdt**2 + dydt**2 + dzdt**2)

        # Compute the derivatives of the Lagrangian with respect to position
        dLdx = self._n(z) * dxdt / propagation_speed
        dLdy = self._n(z) * dydt / propagation_speed
        dLdz = self._n(z) * dzdt / propagation_speed

        return [dxdt, dydt, dzdt, dLdx, dLdy, dLdz]

    def get_propagation_time(self, init_point, final_point):
        """
        Compute the total propagation time of a ray from the initial point to the final point.

        Parameters
        ----------
        init_point : list of floats
            The initial position [x0, y0, z0].
        final_point : list of floats
            The final position [xf, yf, zf].

        Returns
        -------
        float
            The total propagation time of the ray from the initial point to the final point.
        """
        # Initial conditions: position and velocity
        initial_conditions = init_point + [f - i for i, f in zip(init_point, final_point)]
        
        # Time span for the integration
        t_span = (0, 1)

        # Solve the Euler-Lagrange equations using the RK45 method
        sol = solve_ivp(self._euler_lagrange_equations, t_span, initial_conditions, method='RK45', dense_output=True)

        # Generate time values and evaluate the solution at those times
        t_values = np.linspace(0, 1, 500)
        sol_values = sol.sol(t_values)
        x_sol, y_sol, z_sol = sol_values[0], sol_values[1], sol_values[2]

        # Compute gradients (differentials) for the position components
        dx = np.gradient(x_sol)
        dy = np.gradient(y_sol)
        dz = np.gradient(z_sol)
        
        # Compute the differential arc length
        ds = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Compute the optical path length using numerical integration
        optical_path_length = np.trapz(self._n(z_sol) * ds, t_values)
        
        # Compute the propagation time (assuming speed of light in vacuum)
        transit_time = optical_path_length / 299792458

        return transit_time
