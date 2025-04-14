import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib import rcParams

# Import your model and dataset utilities from the first snippet's module
from model_utils import IADatasetNew, ResidualMultiTaskEncoderDecoder
from torch.autograd import grad

class Predict():
    def __init__(self, 
                 model_class, 
                 model_checkpoint,
                 x_train_path,
                 y_train_path,
                 num_passes=1):

        # Initialize the Model class from the first snippet
        self.model_wrapper = model_class(
            ResidualMultiTaskEncoderDecoder(), 
            model_checkpoint,
            x_train_path,
            y_train_path
        )
        
        self.device = self.model_wrapper.device
        self.model = self.model_wrapper.model
        self.train_dataset = self.model_wrapper.train_dataset
        self.num_passes = num_passes

    def predict(self, input_sample, predict: str):
        """
        Use the model_wrapper's predict method for forward pass.
        This method returns means, aleatoric std, and epistemic std.
        """
        # Convert input to a torch tensor on device and with requires_grad=False
        x = torch.tensor(input_sample, dtype=torch.float32, requires_grad=False).to(self.device)
        means, aleo_std, epi_std = self.model_wrapper.predict(x, predict=predict, return_grad=False, num_passes=self.num_passes)
        
        # Convert to numpy for plotting if needed
        means = means.detach().cpu().numpy()
        aleo_std = aleo_std.detach().cpu().numpy()
        epi_std = epi_std.detach().cpu().numpy()
        
        return means, aleo_std, epi_std

    def compute_loss_and_gradient(self, input_sample, predict: str):
        """
        Compute a simple loss (e.g., sum of absolute values of the prediction)
        and the gradient of that loss w.r.t. the input parameters.
        """

        # Prepare input as torch tensor with gradient tracking
        x = torch.tensor(input_sample, dtype=torch.float32, requires_grad=True).to(self.device)

        self.model.eval()
        with torch.set_grad_enabled(True):
            # Forward pass using model.predict logic
            # We just need the raw predictions before inverse transforms,
            # so we replicate a simplified version of model.predict steps here.
            
            # Scale the input the same way model.predict does
            x_reshaped = x.reshape(1, -1)
            x_scaled = (x_reshaped - self.model_wrapper.input_scaler_mean) / self.model_wrapper.input_scaler_std

            if self.num_passes > 1:
                self.model.train()
                for module in self.model.modules():
                    if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
                        module.eval()
                preds = []
                for i in range(self.num_passes):
                    y_pred = self.model.forward_with_dropout(x_scaled)
                    preds.append(y_pred)
                y_pred = torch.stack(preds).to(self.device)
                y_pred_mean = torch.mean(y_pred, dim=0).to(self.device)
            else:
                self.model.eval()
                y_pred_mean = self.model(x_scaled)
            
            # Now choose the correct part of y_pred_mean for the desired prediction
            # Predictions structure: [batch, 3, 40], where each of the 3 tasks has 20 means + 20 std.
            if predict == "xi":
                y_pred_to_use = y_pred_mean[:, 0, :20]
            elif predict == "omega":
                y_pred_to_use = y_pred_mean[:, 1, :20]
            elif predict == "eta":
                y_pred_to_use = y_pred_mean[:, 2, :20]
            else:
                raise ValueError("predict must be one of ['xi', 'omega', 'eta']")

            # Define a loss (example: sum of absolute values)
            loss = torch.sum(torch.abs(y_pred_to_use))

        # Compute gradient w.r.t. input
        gradients = torch.autograd.grad(outputs=loss, inputs=x)[0]
        return loss.item(), gradients.detach().cpu().numpy()

    def gradient_descent(self, initial_input, predict, learning_rate=0.01, num_iterations=100):
        """
        Perform gradient descent on input parameters to minimize a chosen loss.
        """
        input_sample = np.array(initial_input, dtype=np.float32)
        losses = []

        for i in range(num_iterations):
            # Compute loss and gradient
            loss, grad = self.compute_loss_and_gradient(input_sample, predict)
            losses.append(loss)

            # Update input parameters
            input_sample = input_sample - learning_rate * grad

            if i % 10 == 0:
                print(f"Iteration {i}: Loss = {loss}, Input = {input_sample}")

        return input_sample, losses

    def plot_learning_curve(self, losses):
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(losses)), losses, label='Loss during optimization', color='b')
        plt.title('Learning Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('learning_curve.png')

    def plot_correlation_function(self, optimized_input, initial_input, rbins, predict='omega'):
        """
        Plot the correlation function (e.g., ω) before and after optimization.
        """
        means, aleo_std, epi_std = self.predict(optimized_input, predict=predict)
        initial_means, initial_aleo_std, initial_epi_std = self.predict(initial_input, predict=predict)
        rcParams['font.family'] = 'monospace'

        plt.figure(figsize=(8, 6))
        plt.errorbar(rbins, means, yerr=aleo_std, label=r'Optimized $\omega$', fmt='-o', color='blue', linewidth=2)
        plt.errorbar(rbins, initial_means, yerr=initial_aleo_std, label=r'Initial $\omega$', fmt='-o', color='pink', linewidth=2)

        # Fill between for uncertainties if desired
        plt.fill_between(rbins, initial_means - initial_aleo_std, initial_means + initial_aleo_std, color='lightcoral', alpha=0.2, label='Initial Aleatoric')
        plt.fill_between(rbins, initial_means - initial_epi_std, initial_means + initial_epi_std, color='red', alpha=0.2, label='Initial Epistemic')
        plt.fill_between(rbins, means - aleo_std, means + aleo_std, color='lightblue', alpha=0.2, label='Optimized Aleatoric')
        plt.fill_between(rbins, means - epi_std, means + epi_std, color='blue', alpha=0.2, label='Optimized Epistemic')

        plt.title(r'Optimized $\omega$ Correlation Function', fontsize=16)
        plt.xlabel(r"$r \, [h^{-1} \, \mathrm{Mpc}]$")
        plt.ylabel(r"$\omega$(r)")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'optimized_{predict}_correlation.png')

if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_dir)

    # Initialize using your original model class and checkpoint
    # Make sure these paths are correct and that the Model class code is available as model_utils.py
    from model_utils import Model
    rcParams['font.family'] = 'monospace'


    # Instantiate the Predict class using the integrated Model
    model = Predict(
        model_class=Model,
        model_checkpoint=os.path.join(file_dir,'iaemu.pt'),
        x_train_path=os.path.join(file_dir,'data/x_train_10x.npy'),
        y_train_path=os.path.join(file_dir,'data/y_train_10x.npy'),
        num_passes=50
    )

    sample3 = [0.54, 0.05, 11.61, 0.26, 11.8, 12.6, 1.0]
    sample2 = [0.71, 0.14, 11.93, 0.26, 12.05, 12.85, 1.0]
    sample1 = [0.81, 0.35, 12.54, 0.26, 12.68, 13.48, 1.0]
    
    pred1, var1, epi_var1 = model.predict(sample1, 'xi')
    pred2, var2, epi_var2 = model.predict(sample2, 'omega')
    pred3, var3, epi_var3 = model.predict(sample3, 'eta')
        
    r_bins = [ 0.11441248,  0.14739182,  0.18987745,  0.24460954,  0.31511813,
           0.40595079,  0.52296592,  0.67371062,  0.8679074 ,  1.11808132,
           1.44036775,  1.85555311,  2.39041547,  3.07945165,  3.96710221,
           5.11061765,  6.58375089,  8.48151414, 10.92630678, 14.07580982]
    
    # plt.errorbar(r_bins, pred3, yerr=var3, label='aleoatoric')
    # plt.errorbar(r_bins, pred3, yerr=epi_var3, label='epistemic')
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.legend()
    # plt.show()

    # Make three panel plot for xi, omega, eta
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    ax = axs[0]
    ax.errorbar(r_bins, pred1, yerr=var1, label='aleoatoric', color='blue', linewidth=4)
    ax.errorbar(r_bins, pred1, yerr=epi_var1, label='epistemic', color='pink', linewidth=4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"$r \, [h^{-1} \, \mathrm{Mpc}]$", fontsize=24)
    ax.set_ylabel(r"$\xi(r)$", fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=18, length=10, width=2)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=14, length=5, width=1)  # Minor ticks
    #ax.set_title(r"$\xi$", fontsize=20)
    ax.legend()

    ax = axs[1]
    ax.errorbar(r_bins, pred2, yerr=var2, label='aleoatoric', color='blue', linewidth=4)
    ax.errorbar(r_bins, pred2, yerr=epi_var2, label='epistemic', color='pink', linewidth=4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"$r \, [h^{-1} \, \mathrm{Mpc}]$", fontsize=24)
    ax.set_ylabel(r"$\omega(r)$", fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=18, length=10, width=2)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=14, length=5, width=1)  # Minor ticks
    #ax.set_title(r"$\omega$", fontsize=20)
    # ax.legend()

    ax = axs[2]
    ax.errorbar(r_bins, pred3, yerr=var3, label='aleoatoric', color='blue', linewidth=4)
    ax.errorbar(r_bins, pred3, yerr=epi_var3, label='epistemic', color='pink', linewidth=4)
    ax.set_xscale('log')
    ax.set_xlabel(r"$r \, [h^{-1} \, \mathrm{Mpc}]$", fontsize=24)
    ax.set_ylabel(r"$\eta(r)$", fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=18, length=10, width=2)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=14, length=5, width=1)  # Minor ticks
    #ax.set_title(r"$\eta$", fontsize=20)
    # ax.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig('three_panel_correlation_plot.png')
    print("Initial samples plotted")
    


    sample2 = [0.71, 0.14, 11.93, 0.26, 12.05, 12.85, 1.0]
    r_bins = [0.11441248, 0.14739182, 0.18987745, 0.24460954, 0.31511813,
              0.40595079, 0.52296592, 0.67371062, 0.8679074, 1.11808132,
              1.44036775, 1.85555311, 2.39041547, 3.07945165, 3.96710221,
              5.11061765, 6.58375089, 8.48151414, 10.92630678, 14.07580982]

    # Perform gradient descent on sample2 for 'omega'
    optimized_sample, losses = model.gradient_descent(sample2, 'omega', learning_rate=0.01, num_iterations=100)

    print("Optimized input sample:", optimized_sample)
    print("Initial input sample:", sample2)
    print("Difference:", np.abs(optimized_sample - sample2))

    model.plot_learning_curve(losses)
    model.plot_correlation_function(optimized_sample, sample2, r_bins, 'omega')

