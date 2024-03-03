import numpy as np
import matplotlib.pyplot as plt

num_darts = 100000

x_coords = np.random.uniform(0, 1, num_darts)
y_coords = np.random.uniform(0, 1, num_darts)
distance_squared = x_coords**2 + y_coords**2
darts_inside_circle = np.sum(distance_squared <= 1)
pi_estimate = 4 * (darts_inside_circle / num_darts)

print(f"Estimated value of π: {pi_estimate:.6f}")

# show the plot
plt.figure(figsize=(6, 6))
plt.scatter(x_coords, y_coords, c=distance_squared <= 1, cmap='viridis', s=1)
plt.show()