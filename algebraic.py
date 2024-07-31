from random import randrange
import numpy as np
from numpy import random
import math
from random import randrange, normalvariate as normal
from scipy.optimize import minimize

from dataclasses import dataclass
import numpy as np
from numpy.linalg import inv, norm
M = np.diag([1, 1, 1, -1]) # Minkowski matrix

@dataclass
class Vertexer:
    
	nodes: np.ndarray
        
	v = 3e8#(2.99792458e8/sol_2_m)*alpha
    
	def errFunc(self, point, times):
		error = 0    
		for n, t in zip(self.nodes, times):
			error += ((np.linalg.norm(n - point) / self.v) - t)**2
    
		return error
    
	def poly_errFunc(self, Lambda, oneA, invA):
		return abs(np.dot(oneA, oneA) * Lambda**2 + 2 * (np.dot(oneA, invA) - 1) * Lambda + np.dot(invA, invA))

	def find(self, times):
		def lorentzInner(v, w):
			return np.sum(v * (w @ M), axis=-1)

		A = np.append(self.nodes, times * self.v, axis = 1)
	
		times -= min(times)	

		b = 0.5 * lorentzInner(A, A)
		oneA = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, np.ones(len(self.nodes))))
		invA = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))

		initial_roots = np.roots([lorentzInner(oneA, oneA),
                                  2 * (lorentzInner(oneA, invA) - 1),
                                  lorentzInner(invA, invA)])

		refined_roots = []
		for root in initial_roots:
			root = np.real(root)
			print(root)
			res = minimize(self.poly_errFunc, x0=root, args=(oneA, invA), method='BFGS')
			refined_roots.append(res.x[0])
			print(res.x[0])

		refined_solutions = []

		for Lambda in refined_roots:
			X, Y, Z, T = np.linalg.solve(A.T @ A, A.T @ (Lambda * np.ones(len(self.nodes)) + b))
			refined_solutions.append(np.array([X, Y, Z]))

		best_solution = min(refined_solutions, key=lambda x: self.errFunc(x, times))
		return best_solution


# Pick nodes and source to be at random locations
nodes = [[randrange(100) for _ in range(3)] for _ in range(40)]
x, y, z = [randrange(1000) for _ in range(3)]

# Set velocity
c = 3e+8  # m/s
noise_level = 1e-8

# Generate simulated source
distances = [math.sqrt((x - nx)**2 + (y - ny)**2 + (z - nz)**2) / c for nx, ny, nz in nodes]
times = [normal(d, noise_level) for d in distances]
#times = time min(times)

# Print actual source coordinates
print('Actual:', x, y, z)

# Create Vertexer instance and find
myVertexer = Vertexer(np.array(nodes))
print('Reconstructed:',  myVertexer.find(np.array(times).reshape(-1, 1)))
