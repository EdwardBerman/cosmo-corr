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


def forward_model(fixed_hod, mu_cen, mu_sat):
    x = torch.cat(
        [
            mu_cen.unsqueeze(0),
            mu_sat.unsqueeze(0),
            fixed_hod
        ],
        dim=0,
    ).to(device)

    preds, aleos, elis = model.predict(x=x, predict="all", return_grad=True)
    xi, omega, eta = preds
    xi_al, omega_al, eta_al = aleos
    xi_ep, omega_ep, eta_ep = elis

    return xi, omega, eta, xi_al, omega_al, eta_al


# Define the model function with observational data and covariances
def pyro_model():
    # Fixed HOD parameters
    #fixed_HOD_params = torch.tensor([11.37, 0.26, 11.55, 12.35, 1.0], dtype=torch.float32, device=device)
    #fixed_HOD_params = torch.tensor([11.61, 0.26, 11.8, 12.6, 1.0], dtype=torch.float32, device=device)
    fixed_HOD_params = torch.tensor([11.93, 0.26, 12.05, 12.85, 1.0], dtype=torch.float32, device=device)

    # Sample mu_cen and mu_sat
    mu_cen = pyro.sample("mu_cen", dist.Uniform(-1, 1)).type_as(fixed_HOD_params)
    mu_sat = pyro.sample("mu_sat", dist.Uniform(-1, 1)).type_as(fixed_HOD_params)

    # Forward model predictions
    xi_pred, omega_pred, eta_pred, xi_al, omega_aleatoric, eta_al = forward_model(fixed_HOD_params, mu_cen, mu_sat)

    # Define small prior variance for vanishing correlations
    #zero_cov = torch.eye(omega_pred.shape[0], dtype=torch.float32).to(device) * 1e-3 + 1e-6 * torch.eye(omega_pred.shape[0]).to(device)
    omega_cov = (
        torch.diag(omega_aleatoric ** 2)
        + 1e-6 * torch.eye(omega_pred.shape[0]).to(device)  # Regularization
    )

    # Likelihood to encourage omega and eta to vanish
    pyro.sample(
        "omega",
        dist.MultivariateNormal(omega_pred, covariance_matrix=omega_cov),
        obs=torch.zeros_like(omega_pred),
    )

# Run NUTS sampling with Pyro, passing the observational data and covariances
nuts_kernel = NUTS(pyro_model, step_size=0.005, adapt_step_size=True)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=1000)
mcmc.run()
samples = mcmc.get_samples()

# Access samples
mu_cen_samples = samples["mu_cen"]
mu_sat_samples = samples["mu_sat"]

save_dir = "hmc_results/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

np.save(f"{save_dir}mu_cen_samples_set_three.npy", mu_cen_samples)
np.save(f"{save_dir}mu_sat_samples_set_three.npy", mu_sat_samples)
