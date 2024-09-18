import matplotlib.pyplot as plt
import numpy as np

# Data for the lines
bar_count = 3
blue_data = np.array([0.8, 0.2, 0.7])
red_data = np.array([0.2, 0.8, 0.3])

# X positions for the data points
x = np.arange(bar_count)

# Use the xkcd plotting style
plt.xkcd()

# Create a figure with a larger size
plt.figure(figsize=(10, 6))

# Plotting the lines
plt.plot(x, blue_data, color='blue', marker='o', linestyle='-', linewidth=6, label=r'True $\omega(\theta)$')
plt.plot(x, red_data, color='pink', marker='o', linestyle='-', linewidth=6, label=r'Approximate $\omega(\theta)$')

plt.legend(loc='upper right')

# Adding labels and title
plt.xlabel(r'$\theta$', fontsize=14)
plt.ylabel(r'$\omega(\theta)$', fontsize=14)
plt.title(r'Theoretical vs Observed $\omega(\theta$)', fontsize=16)

# Show the plot
plt.savefig('toy_line.png')
plt.show()

