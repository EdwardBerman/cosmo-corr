import numpy as np
import corner
import matplotlib.pyplot as plt

log_M_min_file = "/home/eddieberman/research/mcclearygroup/AstroCorr/emu/iaemu_predict/hmc_results/log_M_min_samples.npy"
sigma_log_M_file = "/home/eddieberman/research/mcclearygroup/AstroCorr/emu/iaemu_predict/hmc_results/sigma_log_M_samples.npy"
log_M_0_file = "/home/eddieberman/research/mcclearygroup/AstroCorr/emu/iaemu_predict/hmc_results/log_M_0_samples.npy"
log_M_1_file = "/home/eddieberman/research/mcclearygroup/AstroCorr/emu/iaemu_predict/hmc_results/log_M_1_samples.npy"
alpha_file = "/home/eddieberman/research/mcclearygroup/AstroCorr/emu/iaemu_predict/hmc_results/alpha_samples.npy"

log_M_min_data = np.load(log_M_min_file)
sigma_log_M_data = np.load(sigma_log_M_file)
log_M_0_data = np.load(log_M_0_file)
log_M_1_data = np.load(log_M_1_file)
alpha_data = np.load(alpha_file)


combined_data = np.column_stack([log_M_min_data, sigma_log_M_data, log_M_0_data, log_M_1_data, alpha_data])  # shape: (N, 5)

labels = [r"$\log M_{min}$", r"$\sigma_{log M}$", r"$\log M_0$", r"$\log M_1$", r"$\alpha$"]

fig = corner.corner(
    combined_data,
    labels=labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    color='blue',
    title_kwargs={"fontsize": 12},
    label_kwargs={"fontname": "Monospace", "fontsize": 12},
)

plt.savefig("hmc_results/hod_corner_plot.png")
