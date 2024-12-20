import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np

# Data for the histogram
bar_count = 3
blue_data = np.array([8, 2, 7])
red_data = np.array([2, 8, 3])

# X positions for the bars
x = np.arange(bar_count)

# Use the xkcd plotting style
#plt.xkcd()
plt.rc('font', family='monospace')

plt.figure(figsize=(10, 6))

# Plotting the stacked bars
plt.bar(x, blue_data, color='blue', label='True Distribution')
plt.bar(x, red_data, bottom=blue_data, color='pink', label='Binned Approximate Distribution')

# Adding labels and title
plt.xlabel(r'$\theta$', fontsize=14)
plt.ylabel(r'$DD(\theta)$', fontsize=14)
plt.title(r'True vs Approximate $DD(\theta$)', fontsize=14)

# Adding a legend
plt.legend(loc='lower right')

# Adding arrows
plt.annotate('', xy=(0.5, 6), xytext=(0, 6),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('', xy=(1.5, 6), xytext=(2, 6),
             arrowprops=dict(facecolor='black', shrink=0.05))

def add_outlined_text(x, y, text, fontsize=20):
    t = plt.text(x, y, text, fontsize=fontsize, fontweight='bold', color='black',
                 verticalalignment='center', horizontalalignment='center')
    t.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])
    return t

add_outlined_text(0.25, 6.5, "Binning Error")
add_outlined_text(1.75, 6.5, "Binning Error")

#plt.text(0.0, 6.5, "Binning Error", fontsize=12, verticalalignment='center', fontweight='bold')
#plt.text(1.45, 6.5, "Binning Error", fontsize=12, verticalalignment='center', fontweight='bold')
# Show the plot
plt.savefig('toy.png')
