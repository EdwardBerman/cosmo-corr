import numpy as np
import matplotlib.pyplot as plt

# Define file names for rho statistics
file_mapping = {
    'rho1': {
        'bootstrap': ('distances_bootstrap.npy', 'rho1_means_bootstrap.npy', 'rho1_stds_bootstrap.npy'),
        'sample': ('rho1_distances_sample.npy', 'rho1_means_sample.npy', 'rho1_stds_sample.npy')
    },
    'rho2': {
        'bootstrap': ('distances2_bootstrap.npy', 'rho2_means_bootstrap.npy', 'rho2_stds_bootstrap.npy'),
        'sample': ('rho2_distances_sample.npy', 'rho2_means_sample.npy', 'rho2_stds_sample.npy')
    },
    'rho3': {
        'bootstrap': ('distances3_bootstrap.npy', 'rho3_means_bootstrap.npy', 'rho3_stds_bootstrap.npy'),
        'sample': ('rho3_distances_sample.npy', 'rho3_means_sample.npy', 'rho3_stds_sample.npy')
    },
    'rho4': {
        'bootstrap': ('distances4_bootstrap.npy', 'rho4_means_bootstrap.npy', 'rho4_stds_bootstrap.npy'),
        'sample': ('rho4_distances_sample.npy', 'rho4_means_sample.npy', 'rho4_stds_sample.npy')
    },
    'rho5': {
        'bootstrap': ('distances5_bootstrap.npy', 'rho5_means_bootstrap.npy', 'rho5_stds_bootstrap.npy'),
        'sample': ('rho5_distances_sample.npy', 'rho5_means_sample.npy', 'rho5_stds_sample.npy')
    }
}

for rho, datasets in file_mapping.items():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Initialize overall bounds for both sample and bootstrap
    x_min_global, x_max_global = np.inf, -np.inf
    y_min_global, y_max_global = np.inf, -np.inf

    # Compute global bounds for both sample and bootstrap
    for method, (distance_file, mean_file, std_file) in datasets.items():
        distances = np.load(distance_file)
        means = np.abs(np.load(mean_file))
        stds = np.load(std_file)

        x_min_global = min(x_min_global, np.min(distances))
        x_max_global = max(x_max_global, np.max(distances))
        y_min_global = min(y_min_global, np.min(means - stds))
        y_max_global = max(y_max_global, np.max(means + stds))

    # Ensure positive bounds for log scale
    x_min_global = max(x_min_global, 1e-10)
    y_min_global = max(y_min_global, 1e-10)

    # Add buffer for axis limits
    x_buffer = (x_max_global - x_min_global) * 0.05
    y_buffer = (y_max_global - y_min_global) * 0.05

    # Plot each method with the same bounds
    for ax, (method, (distance_file, mean_file, std_file)) in zip(axes, datasets.items()):
        distances = np.load(distance_file)
        means = np.abs(np.load(mean_file))
        stds = np.load(std_file)

        ax.errorbar(distances, means, yerr=stds, fmt='o', alpha=0.7)
        ax.set_yscale('log')
        ax.set_xlim(x_min_global - x_buffer, x_max_global + x_buffer)
        ax.set_ylim(y_min_global - y_buffer, y_max_global + y_buffer)
        ax.set_xlabel(r'$θ \ [\mathrm{arcmin}]$', fontsize=12, family='monospace')
        ax.set_title(rf'$\rho_{{{rho[-1]}}}$ {method.capitalize()}', fontsize=12, family='monospace')
        ax.grid(True, which="both", linestyle='--', linewidth=0.5)

    # Set shared ylabel
    axes[0].set_ylabel(r'$\log_{10}(|\xi(\theta)|)$', fontsize=12, family='monospace')
    plt.tight_layout()
    plt.savefig(f'{rho}_plot.png')
