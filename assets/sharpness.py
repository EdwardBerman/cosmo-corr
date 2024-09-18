import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans Mono'
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
plt.xlabel('x', fontsize=18)
plt.ylabel(r'$f(x) = \frac{1}{1 + e^{-\alpha(x - 1)}} \cdot \frac{1}{1 + e^{\alpha(x - 2)}}$', fontsize=18)
plt.title('Sharpness Sensitivity to Sigmoid Filter', fontsize=20)
plt.legend()

# Show the plot
plt.savefig('/home/eddieberman/research/mcclearygroup/AstroCorr/assets/two_sigmoids.png')
