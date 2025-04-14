import numpy as np
import corner
import matplotlib.pyplot as plt

param1_file = "/home/eddieberman/research/mcclearygroup/AstroCorr/emu/iaemu_predict/hmc_results/mu_cen_samples_set_one.npy"
param2_file = "/home/eddieberman/research/mcclearygroup/AstroCorr/emu/iaemu_predict/hmc_results/mu_sat_samples_set_one.npy"

param1_data = np.load(param1_file)  
param2_data = np.load(param2_file)

combined_data = np.column_stack([param1_data, param2_data])  # shape: (N, 2)

labels = [r"$\mu_{cen}$", r"$\mu_{sat}$"]

fig = corner.corner(
    combined_data,
    labels=labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    color='blue',
    title_kwargs={"fontsize": 12},
    label_kwargs={"fontname": "Monospace", "fontsize": 12},
)

plt.savefig("hmc_results/corner_plot_set_one.png")

param1_file = "/home/eddieberman/research/mcclearygroup/AstroCorr/emu/iaemu_predict/hmc_results/mu_cen_samples_set_two.npy"
param2_file = "/home/eddieberman/research/mcclearygroup/AstroCorr/emu/iaemu_predict/hmc_results/mu_sat_samples_set_two.npy"

param1_data = np.load(param1_file)  
param2_data = np.load(param2_file)

combined_data = np.column_stack([param1_data, param2_data])  # shape: (N, 2)

labels = [r"$\mu_{cen}$", r"$\mu_{sat}$"]

fig = corner.corner(
    combined_data,
    labels=labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    color='blue',
    title_kwargs={"fontsize": 12},
    label_kwargs={"fontname": "Monospace", "fontsize": 12},
)

plt.savefig("hmc_results/corner_plot_set_two.png")


param1_file = "/home/eddieberman/research/mcclearygroup/AstroCorr/emu/iaemu_predict/hmc_results/mu_cen_samples_set_three.npy"
param2_file = "/home/eddieberman/research/mcclearygroup/AstroCorr/emu/iaemu_predict/hmc_results/mu_sat_samples_set_three.npy"

param1_data = np.load(param1_file)  
param2_data = np.load(param2_file)

combined_data = np.column_stack([param1_data, param2_data])  # shape: (N, 2)

labels = [r"$\mu_{cen}$", r"$\mu_{sat}$"]

fig = corner.corner(
    combined_data,
    labels=labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    color='blue',
    title_kwargs={"fontsize": 12},
    label_kwargs={"fontname": "Monospace", "fontsize": 12},
)

plt.savefig("hmc_results/corner_plot_set_three.png")


