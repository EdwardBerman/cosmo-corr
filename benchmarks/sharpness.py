import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 3, 1000)

# Define the function with two sigmoids facing opposite directions
def two_sigmoids(x, alpha, a=1, b=2):
    return (1 / (1 + np.exp(-alpha * (x - a)))) * (1 / (1 + np.exp(alpha * (x - b))))

# Define the alphas for the plot
alphas = [0.5, 1, 5, 10, 20]

# Create a plot for both sigmoids with different alpha values
plt.figure(figsize=(8, 6))

# Plot the function for different alpha values with a cool colormap
colors = plt.cm.cool(np.linspace(0, 1, len(alphas)))

# Plot both sigmoids facing opposite directions
for i, alpha in enumerate(alphas):
    y = two_sigmoids(x, alpha, a=1, b=2)
    plt.plot(x, y, color=colors[i], label=f'alpha={alpha}')

# Add labels, title, and legend
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Two Sigmoids Facing Opposite Directions for Different Alpha Values')
plt.legend()

# Show the plot
plt.show()

# Now plot the effect of flipping the sign of alpha on one of the sigmoids
def flipped_sigmoids(x, alpha, a=1, b=2):
    return (1 / (1 + np.exp(-alpha * (x - a)))) * (1 / (1 + np.exp(-alpha * (x - b))))

# Create a plot for the flipped sigmoid
plt.figure(figsize=(8, 6))

# Plot the function for different alpha values with the flipped sign
for i, alpha in enumerate(alphas):
    y_flipped = flipped_sigmoids(x, alpha, a=1, b=2)
    plt.plot(x, y_flipped, color=colors[i], label=f'alpha={alpha} (flipped)')

# Add labels, title, and legend
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Sharpness Sensitivity to Sigmoid Filter')
plt.legend()

# Show the plot
plt.savefig('flipped_sigmoids.png')

