import numpy as np
import matplotlib.pyplot as plt

# Define parameters for spirals with random theta offsets
num_spirals = 5
colors = ['pink', 'purple']  # Alternating colors
line_width = 10

# Create a plot
plt.figure(figsize=(8, 8))

# Generate and plot each spiral
for i in range(num_spirals):
    theta_offset = np.random.uniform(0, 2 * np.pi)  # Random starting angle
    theta = np.linspace(theta_offset, theta_offset + 2 * np.pi, 500)  # Less than a full rotation
    random_growth = np.random.uniform(0.5, 1.2, size=500)  # Random growth rate for radius
    radius = (np.cumsum(random_growth) / 200) + i * 0.5  # More spacing between spirals
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    plt.plot(x, y, color=colors[i % 2], linewidth=line_width)

# Adjust plot aesthetics
plt.axis('off')  # Remove background grid and coordinates
#plt.title("Spirals with Random Growth and Theta Offsets", fontsize=16)
plt.savefig('spiral.png')

