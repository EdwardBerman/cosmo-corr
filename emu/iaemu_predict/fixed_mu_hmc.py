import os

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS

from iaemu_predict import Model
from model_utils import ResidualMultiTaskEncoderDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize your PyTorch model and move it to the GPU
model = Model(
    ResidualMultiTaskEncoderDecoder(),
    "/home/eddieberman/research/mcclearygroup/AstroCorr/emu/iaemu_predict/iaemu.pt",
    "/home/eddieberman/research/mcclearygroup/AstroCorr/emu/iaemu_predict/data/x_train_10x.npy",
    "/home/eddieberman/research/mcclearygroup/AstroCorr/emu/iaemu_predict/data/y_train_10x.npy",
)


def forward_model(hod, fixed_mu_cen, fixed_mu_sat):
    fixed_mu_cen = fixed_mu_cen.view(1, 1)  # Make it [1, 1]
    fixed_mu_sat = fixed_mu_sat.view(1, 1)  # Make it [1, 1]

    x = torch.cat(
        [
            fixed_mu_cen,
            fixed_mu_sat,
            hod
        ],
        dim=0,  # Stack vertically
    ).to(device)

    preds, aleos, elis = model.predict(x=x, predict="all", return_grad=True)
    xi, omega, eta = preds
    xi_al, omega_al, eta_al = aleos
    xi_ep, omega_ep, eta_ep = elis

    return xi, omega, eta

# Define the model function with observational data and covariances
def pyro_model():
    log_M_min = torch.tensor([11.35, 11.46, 11.60, 11.75, 12.02, 12.30, 12.79, 13.38, 14.22], device=device)
    sigma_log_M = torch.tensor([0.25, 0.24, 0.26, 0.28, 0.26, 0.21, 0.39, 0.51, 0.77], device=device)
    log_M_zero = torch.tensor([11.20, 10.59, 11.49, 11.69, 11.38, 11.84, 11.92, 13.94, 14.00], device=device)
    log_M_one = torch.tensor([12.40, 12.68, 12.83, 13.01, 13.31, 13.58, 13.94, 13.91, 14.69], device=device)
    alpha = torch.tensor([0.83, 0.97, 1.02, 1.06, 1.06, 1.12, 1.15, 1.04, 0.87], device=device)

    linear_fit_log_M_zero = torch.linalg.lstsq(torch.vstack([log_M_min, torch.ones_like(log_M_min)]).T, log_M_zero).solution
    rmse_log_M_zero = 20 * torch.sqrt(torch.mean((torch.matmul(torch.vstack([log_M_min, torch.ones_like(log_M_min)]).T, linear_fit_log_M_zero) - log_M_zero)**2))

    linear_fit_log_M_one = torch.linalg.lstsq(torch.vstack([log_M_min, torch.ones_like(log_M_min)]).T, log_M_one).solution
    rmse_log_M_one = 20 * torch.sqrt(torch.mean((torch.matmul(torch.vstack([log_M_min, torch.ones_like(log_M_min)]).T, linear_fit_log_M_one) - log_M_one)**2))

    linear_fit_alpha = torch.linalg.lstsq(torch.vstack([log_M_min, torch.ones_like(log_M_min)]).T, alpha).solution
    rmse_alpha = 20 * torch.sqrt(torch.mean((torch.matmul(torch.vstack([log_M_min, torch.ones_like(log_M_min)]).T, linear_fit_alpha) - alpha)**2))

    linear_fit_sigma_log_M = torch.linalg.lstsq(torch.vstack([log_M_min, torch.ones_like(log_M_min)]).T, sigma_log_M).solution
    rmse_sigma_log_M = 20 * torch.sqrt(torch.mean((torch.matmul(torch.vstack([log_M_min, torch.ones_like(log_M_min)]).T, linear_fit_sigma_log_M) - sigma_log_M)**2))

    # Sample mu_cen and mu_sat
    mu_cen = torch.tensor(1.0, dtype=torch.float32, device=device)
    mu_sat = torch.tensor(1.0, dtype=torch.float32, device=device)
    log_M_min = pyro.sample("log_M_min", dist.Uniform(11.0, 15.0)).to(device)

    sigma_log_M_lower_bound = torch.matmul(torch.vstack([log_M_min, torch.ones_like(log_M_min)]).T, linear_fit_sigma_log_M) - rmse_sigma_log_M
    sigma_log_M_upper_bound = torch.matmul(torch.vstack([log_M_min, torch.ones_like(log_M_min)]).T, linear_fit_sigma_log_M) + rmse_sigma_log_M
    sigma_log_M = pyro.sample("sigma_log_M", dist.Uniform(sigma_log_M_lower_bound, sigma_log_M_upper_bound)).to(device)

    log_M_0_lower_bound = torch.matmul(torch.vstack([log_M_min, torch.ones_like(log_M_min)]).T, linear_fit_log_M_zero) - rmse_log_M_zero
    log_M_0_upper_bound = torch.matmul(torch.vstack([log_M_min, torch.ones_like(log_M_min)]).T, linear_fit_log_M_zero) + rmse_log_M_zero
    log_M_0 = pyro.sample("log_M_0", dist.Uniform(log_M_0_lower_bound, log_M_0_upper_bound)).to(device)

    log_M_1_lower_bound = torch.matmul(torch.vstack([log_M_min, torch.ones_like(log_M_min)]).T, linear_fit_log_M_one) - rmse_log_M_one
    log_M_1_upper_bound = torch.matmul(torch.vstack([log_M_min, torch.ones_like(log_M_min)]).T, linear_fit_log_M_one) + rmse_log_M_one
    log_M_1 = pyro.sample("log_M_1", dist.Uniform(log_M_1_lower_bound, log_M_1_upper_bound)).to(device)

    alpha_lower_bound = torch.matmul(torch.vstack([log_M_min, torch.ones_like(log_M_min)]).T, linear_fit_alpha) - rmse_alpha
    alpha_upper_bound = torch.matmul(torch.vstack([log_M_min, torch.ones_like(log_M_min)]).T, linear_fit_alpha) + rmse_alpha
    alpha = pyro.sample("alpha", dist.Uniform(alpha_lower_bound, alpha_upper_bound)).to(device)

    log_M_min = log_M_min.unsqueeze(0) if log_M_min.dim() == 0 else log_M_min
    sigma_log_M = sigma_log_M.unsqueeze(0) if sigma_log_M.dim() == 0 else sigma_log_M
    log_M_0 = log_M_0.unsqueeze(0) if log_M_0.dim() == 0 else log_M_0
    log_M_1 = log_M_1.unsqueeze(0) if log_M_1.dim() == 0 else log_M_1
    alpha = alpha.unsqueeze(0) if alpha.dim() == 0 else alpha
    
    HOD_params = torch.stack([log_M_min, sigma_log_M, log_M_0, log_M_1, alpha])

    # Forward model predictions
    xi_pred, omega_pred, eta_pred = forward_model(HOD_params, mu_cen, mu_sat)

    # Define small prior variance for vanishing correlations
    zero_cov = torch.eye(omega_pred.shape[0], dtype=torch.float32).to(device) * 1e-3 + 1e-6 * torch.eye(omega_pred.shape[0]).to(device)

    # Likelihood to encourage omega and eta to vanish
    pyro.sample(
        "omega",
        dist.MultivariateNormal(omega_pred, covariance_matrix=zero_cov),
        obs=torch.zeros_like(omega_pred),
    )

# Run NUTS sampling with Pyro, passing the observational data and covariances
nuts_kernel = NUTS(pyro_model, step_size=0.005, adapt_step_size=True)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=1000)
mcmc.run()
samples = mcmc.get_samples()

# Access samples
log_M_min_samples = samples["log_M_min"]
sigma_log_M_samples = samples["sigma_log_M"]
log_M_0_samples = samples["log_M_0"]
log_M_1_samples = samples["log_M_1"]
alpha_samples = samples["alpha"]


save_dir = "hmc_results/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

np.save(f"{save_dir}log_M_min_samples.npy", log_M_min_samples)
np.save(f"{save_dir}sigma_log_M_samples.npy", sigma_log_M_samples)
np.save(f"{save_dir}log_M_0_samples.npy", log_M_0_samples)
np.save(f"{save_dir}log_M_1_samples.npy", log_M_1_samples)
np.save(f"{save_dir}alpha_samples.npy", alpha_samples)
