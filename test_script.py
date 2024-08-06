# Pick nodes and source to be at random locations
nodes = [[randrange(100) for _ in range(3)] for _ in range(40)]
x, y, z = [randrange(1000) for _ in range(3)]

# Set velocity
c = 3e+8  # m/s
noise_level = 1e-9

# Generate simulated source
distances = [math.sqrt((x - nx)**2 + (y - ny)**2 + (z - nz)**2) / c for nx, ny, nz in nodes]
times = [normal(d, noise_level) for d in distances]

# Print actual source coordinates
print('Actual:', x, y, z)

# Create Vertexer instance and find
myVertexer = Vertexer(np.array(nodes))
myVertexer.find(np.array(times).reshape(-1, 1))
