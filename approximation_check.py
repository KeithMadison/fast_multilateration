import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Define constants
A = 1.78
B = 0.454
C = 0.0132

def compute_differences(max_depth, min_depth):
	# Define the integral_result function
	def integral_result(z, A, B, C, beta, K):
		t = np.sqrt((A + B * np.exp(C * z) - beta) / (A + B * np.exp(C * z) + beta))
		alpha = np.sqrt((A - beta) / (A + beta))
		return A * np.sqrt((C ** 2 + K ** 2) / (C ** 2 * K ** 2)) * np.log(np.abs((t - alpha) / (t + alpha)))

	# Define the integrand function
	def integrand(z, A, B, C, beta, K):
		return 1 / np.sqrt(A ** 2 + 2 * A * B * np.exp(C * z) + B ** 2 * np.exp(2 * C * z) - beta ** 2)

	# Define the exact_integrand function
	def exact_integrand(z, A, B, C, K, beta):
		numerator = C ** 2 * (A - beta) * (A + beta)
		denominator = K ** 2 * (A - beta + B * np.exp(C * z)) * (A + beta + B * np.exp(C * z))
		return (A - B * np.exp(C * z)) * np.sqrt(numerator / denominator + 1)

	# Number of samples
	num_samples = 500000

	# Store the differences
	differences = []

	for _ in range(num_samples):
		# Generate random values
		theta = np.random.uniform(-np.pi/2, np.pi/2)
		z0 = np.random.uniform(max_depth, min_depth)
		z1 = np.random.uniform(max_depth, min_depth)

		# Calculate beta and K
		beta = (A - B * np.exp(C * z0)) * np.cos(theta)
		K = C * np.sqrt(A ** 2 - beta ** 2) / beta

		# Compute exact result
		exact_result, _ = quad(lambda z: exact_integrand(z, A, B, C, K, beta), z0, z1)

		# Compute numerical result
		result, _ = quad(lambda z: integrand(z, A, B, C, beta, K), z0, z1)
		r = A * np.sqrt((A ** 2 - beta ** 2) * (C ** 2 + K ** 2)) / np.abs(K)
		result *= r

		# Compute analytical result
		analytical_result = (integral_result(z1, A, B, C, beta, K) - integral_result(z0, A, B, C, beta, K))

		# Compute difference
		difference = abs(exact_result - analytical_result)
		differences.append((difference / 299792458) * 1e9)
	
	return differences

differences_1 = compute_differences(-1000, 0)
differences_2 = compute_differences(-1000, -25)
differences_3 = compute_differences(-1000, -50)
differences_4 = compute_differences(-1000, -100)

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the stacked histograms
ax.hist(differences_1, bins=5000, alpha=0.5, label='0 → 1km', color='blue', stacked=True)
ax.hist(differences_2, bins=5000, alpha=0.5, label='25m → 1km', color='orange', stacked=True)
ax.hist(differences_3, bins=5000, alpha=0.5, label='50m → 1km', color='green', stacked=True)
ax.hist(differences_4, bins=5000, alpha=0.5, label='100m → 1km', color='red', stacked=True)

# Add labels and title
ax.set_xlabel('Difference (ns)')
ax.set_ylabel('Frequency')
ax.set_title('5e+5 Coordinate Pairs')
ax.legend()

ax.set_xlim([0, 0.2])

# Show the plot
plt.show()
