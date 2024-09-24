# This uses my solution to the general exponential Euler-Lagrange system to construct a system of equations which can (in principle) be solved for an unknown emission coordinate
# (Doing so in practice has proven difficult due to small but frequent perturbations in the objective)

from ray_tracer import RayTracer
from random import randrange
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass

@dataclass
class TimeDifferenceSolver:
    z_i: np.ndarray  # Array of known z_i coordinates
    params: np.ndarray  # Array containing A, B, and C

    def _g(self, z0, beta_i, z_i, A, B, C):
        A, B, C = self.params

        beta = beta_i
        sqrt_term = np.sqrt(A**2 - beta**2)

        term_zi = np.sqrt((C**2 + (C * sqrt_term / beta)**2) / (C**2 * (C * sqrt_term / beta)**2))
        log_argument_zi = np.abs(
            (np.sqrt((A + B * np.exp(C * z_i) - beta) / (A + B * np.exp(C * z_i) + beta)) - np.sqrt((A - beta) / (A + beta))) /
            (np.sqrt((A + B * np.exp(C * z_i) - beta) / (A + B * np.exp(C * z_i) + beta)) + np.sqrt((A - beta) / (A + beta)))
        )
        log_zi = np.log(log_argument_zi)

        first_term_zi = A * term_zi * log_zi

        log_argument_z0 = np.abs(
            (np.sqrt((A + B * np.exp(C * z0) - beta) / (A + B * np.exp(C * z0) + beta)) - np.sqrt((A - beta) / (A + beta))) /
            (np.sqrt((A + B * np.exp(C * z0) - beta) / (A + B * np.exp(C * z0) + beta)) + np.sqrt((A - beta) / (A + beta)))
        )
        log_z0 = np.log(log_argument_z0)

        first_term_z0 = A * term_zi * log_z0

        result = first_term_zi - first_term_z0

        return result/299792458

    def objective_function(self, variables, t_i_values):
        # Extract z_0 and beta_i values from the optimization variable
        z0 = variables[0]
        beta_i = variables[1:]

        A, B, C = self.params
        n = len(self.z_i)

        # Compute the sum of squared differences for all pairs (i, j)
        error_sum = 0
        for i in range(n):
            for j in range(i + 1, n):
                g_i_value = self._g(z0, beta_i[i], self.z_i[i], A, B, C)
                g_j_value = self._g(z0, beta_i[j], self.z_i[j], A, B, C)
                error_sum += (g_i_value - g_j_value - (t_i_values[i] - t_i_values[j]))**2

        return error_sum

    def find(self, t_i_values):
        # Initial guess for z_0 and beta_i
        initial_guess = np.ones(len(self.z_i) + 1) * -1  # Starting with z_0 and beta_i all -1 (as z0 and zi < 0)

        # Minimize the objective function
        result = minimize(self.objective_function, initial_guess, args=(t_i_values,), method='L-BFGS-B')
        
        # Extract z_0 and beta_i values from the result
        z0_opt = result.x[0]
        beta_i_opt = result.x[1:]
        
        return z0_opt, beta_i_opt

t_i_examples = [[6.866262313050545e-07, 5.954638024811323e-07, 7.370951274348997e-07, 6.643872447828826e-07, 4.562073179501557e-07, 3.865749758904363e-07, 2.8658197543587016e-07, 6.939577942623097e-07, 3.7768732572207756e-07, 1.325045467564166e-07], [6.767497217196613e-06, 6.560908657371282e-06, 6.925210333973973e-06, 6.530254888400991e-06, 6.755516813206531e-06, 6.619582551729789e-06, 6.579166768050318e-06, 7.151171380072868e-06, 7.000049368578958e-06, 6.682370044729153e-06], [5.984702539514993e-06, 5.930014598549508e-06, 5.598447335939742e-06, 6.190116484206916e-06, 5.749205253437443e-06, 6.099216577640195e-06, 6.09800253359304e-06, 6.002179216351752e-06, 6.385774539401187e-06, 6.088005634055416e-06], [4.665259457709067e-06, 4.72830311317074e-06, 4.824128899105909e-06, 5.068049986598144e-06, 4.720203892443567e-06, 5.203220656040072e-06, 4.684312194021325e-06, 5.011060151019048e-06, 5.196928493759881e-06, 5.086752131673956e-06], [5.004986255762178e-06, 5.1259407121812555e-06, 5.577304001403536e-06, 4.945493904613105e-06, 5.3276368868484015e-06, 4.984916724948599e-06, 5.504421969627293e-06, 5.086412981784503e-06, 5.0244461591639915e-06, 5.09299975674606e-06]]

z_0_examples = [-87, -528, -679, -781, -877]

z_i_examples = [[-10, -22, -3, -13, -39, -47, -58, -9, -48, -74],
        [-34, -94, -29, -65, -11, -29, -80, -34, -86, -63],
        [-92, -22, -99, -84, -85, -62, -86, -19, -23, -70],
        [-22, -96, -86, -42, -96, -3, -83, -40, -21, -3],
        [-52, -26, -4, -63, -36, -89, -17, -100, -48, -86]]


beta_i_examples = [[1.6360167176504987, 1.6360167176504987, 1.6360167176504987, 1.6360167176504987, 1.6360167176504987, 1.6360167176504987, 1.6360167176504987, 1.6360167176504987, 1.6360167176504987, 1.6360167176504987],
           [1.6396928887135265, 1.6574727386987005, 1.6454327818483134, 1.6418398684695759, 1.6294595761205148, 1.6296247114861102, 1.6515037156116221, 1.65748863574386, 1.6715678081516174, 1.6483391958245628],
           [1.4662909546401228, 1.3834362335899673, 1.4220961871328066, 1.482251553825198, 1.4272665047376376, 1.4498234809191288, 1.4736817069254222, 1.3919204565138836, 1.4497458404733339, 1.4564473325785974],
           [0.5693154689394847, 0.9341213201145733, 0.9509296446353164, 0.9389788567461165, 0.9298062250965199, 0.8975872756256842, 0.8629097820901123, 0.9035389490524869, 0.9435090344661008, 0.831211938974932],
           [0.45239345553514915, 0.4303687193335445, 0.7425426901537331, 0.45295109015203266, 0.685269810976178, 0.6509174325017857, 0.7387590151232529, 0.7763085100137423, 0.44961840753325033, 0.7225205197927285]]


params = np.array([1.78, 0.454, 0.0132])

for i in range(len(z_i_examples)):

    solver = TimeDifferenceSolver(z_i_examples[i], params)
    z0_solution, beta_i_solution = solver.find(t_i_examples[i])


    print("Value of objective at 'true' solution:", solver.objective_function(np.insert(beta_i_examples[i], 0, z_0_examples[i], axis=0), t_i_examples[i]))

    print(f"True z0: {z_0_examples[i]}")
    print(f"True beta_i: {beta_i_examples[i]}")

    print(f"Optimal z0: {z0_solution}")
    print(f"Optimal beta_i: {beta_i_solution}")
