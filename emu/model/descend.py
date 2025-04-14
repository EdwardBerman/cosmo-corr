import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from utils import IADatasetNew, ResidualMultiTaskEncoderDecoder
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import SymLogNorm
from matplotlib import rcParams

file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)

class Predict():
    
    def __init__(self, 
                 model, 
                 model_checkpoint,
                 x_train_path,
                 y_train_path,
                 num_passes):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.model_checkpoint = model_checkpoint
        self.model.load_state_dict(torch.load(model_checkpoint, map_location=device))
        self.x_train, self.y_train = np.load(x_train_path), np.load(y_train_path)
        self.train_dataset = IADatasetNew(self.x_train, self.y_train)
        
        self.scaler = StandardScaler().fit(self.x_train)

        self.num_passes = num_passes
        
    def predict(self, 
                input,
                predict: str):
        self.model.eval()
        
        x = np.array(input).reshape(1, -1)
        x = self.scaler.transform(x)
        x = torch.tensor(x, requires_grad=True).float()
        preds = []
        
        with torch.no_grad():
            for i in range(self.num_passes):
                y_pred = self.model.forward_with_dropout(x)
                preds.append(y_pred)
                
        y_pred = torch.stack(preds)
        y_pred_mean = torch.mean(y_pred, dim=0)
        y_pred_mc_var = torch.var(y_pred[:, :, :, :20], dim=0)
        scaled_means = self.train_dataset.inverse_transform(y_pred_mean.cpu().numpy()).squeeze()
        scaled_y_var = self.train_dataset.inverse_mc_variance(y_pred_mc_var.cpu().numpy()).squeeze()

        if predict == 'xi':
            means = scaled_means[0, :20]
            aleo_std = np.sqrt(scaled_means[0, 20:])
            epi_std = np.sqrt(np.abs(scaled_y_var[0]))
        elif predict == 'omega':
            means = scaled_means[1, :20]
            aleo_std = np.sqrt(scaled_means[1, 20:])
            epi_std = np.sqrt(np.abs(scaled_y_var[1]))
        elif predict == 'eta':
            means = scaled_means[2, :20]
            aleo_std = np.sqrt(scaled_means[2, 20:])
            epi_std = np.sqrt(np.abs(scaled_y_var[2]))
            
        return means, aleo_std, epi_std
    
    def compute_gradient(self, input, predict: str):
        self.model.eval()
        
        x = np.array(input).reshape(1, -1)
        x = self.scaler.transform(x)
        x = torch.tensor(x, requires_grad=True).float()
        
        with torch.enable_grad():
            y_pred = self.model.forward_with_dropout(x)
        
        if predict == 'xi':
            y_pred_to_use = y_pred[:, 0, :20]  # For xi
        elif predict == 'omega':
            y_pred_to_use = y_pred[:, 1, :20]  # For omega
        elif predict == 'eta':
            y_pred_to_use = y_pred[:, 2, :20]  # For eta
            
        # Compute gradients with respect to the input
        gradients = torch.autograd.grad(outputs=y_pred_to_use.sum(), inputs=x)[0].cpu().numpy()
        
        return gradients

    def compute_loss_and_gradient(self, input, predict: str):
        self.model.eval()
        
        x = np.array(input).reshape(1, -1)
        x = self.scaler.transform(x)
        x = torch.tensor(x, requires_grad=True).float()
        
        # Forward pass through the model
        with torch.enable_grad():
            y_pred = self.model.forward_with_dropout(x)
        
        if predict == 'xi':
            y_pred_to_use = y_pred[:, 0, :20]  # For xi
        elif predict == 'omega':
            y_pred_to_use = y_pred[:, 1, :20]  # For omega
        elif predict == 'eta':
            y_pred_to_use = y_pred[:, 2, :20]  # For eta

        loss = torch.sum(torch.abs(y_pred_to_use))
        
        # Compute gradients with respect to the input
        gradients = torch.autograd.grad(outputs=loss, inputs=x)[0].cpu().numpy()
        
        return loss.item(), gradients

    def gradient_descent(self, initial_input, predict, learning_rate=0.01, num_iterations=100):
        input_sample = np.array(initial_input)
        original_input = np.copy(input_sample)
        losses = []
        
        for i in range(num_iterations):
            # Compute the loss and the gradient
            loss, grad = self.compute_loss_and_gradient(input_sample, predict)
            
            # Save the loss for tracking
            losses.append(loss)
            
            # Update the input using the gradient (gradient descent step)
            input_sample = input_sample - learning_rate * grad
            
            # Print the progress every 10 iterations
            if i % 10 == 0:
                print(f"Iteration {i}: Loss = {loss}, Input = {input_sample}")
        
        return input_sample, losses

    
    def plot_learning_curve(self, losses):
        """Plot the learning curve (loss vs. iterations)."""
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(losses)), losses, label='Loss during training', color='b')
        plt.title('Learning Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('learning_curve.png')

    def plot_correlation_function(self, optimized_input, initial_input, rbins, predict):
        """Plot the learned xi correlation function."""
        rcParams['font.family'] = 'monospace'
        means, aleo_std, epi_std = self.predict(optimized_input, predict='omega')
        initial_means, initial_aleo_std, initial_epi_std = self.predict(initial_input, predict='omega')
        plt.figure(figsize=(8, 6))
        plt.errorbar(range(len(means)), means, yerr=aleo_std, label=r'Learned $\omega$ correlation', fmt='-o', color='blue', linewidth=4)
        plt.errorbar(range(len(initial_means)), initial_means, label=r'Initial $\omega$ correlation', fmt='-o', color='pink', linewidth=4)
        plt.fill_between(range(len(initial_means)), initial_means - initial_aleo_std, initial_means + initial_aleo_std, color='lightcoral', alpha=0.2, label='Initial Aleatoric uncertainty')
        plt.fill_between(range(len(initial_means)), initial_means - initial_epi_std, initial_means + initial_epi_std, color='red', alpha=0.2, label='Initial Epistemic uncertainty')
        plt.fill_between(range(len(means)), means - aleo_std, means + aleo_std, color='lightblue', alpha=0.2, label='Learned Aleatoric uncertainty')
        plt.fill_between(range(len(means)), means - epi_std, means + epi_std, color='blue', alpha=0.2, label='Learned Epistemic uncertainty')
        plt.title(r'Learned $\omega$ Correlation Function', fontsize=20)
        plt.xlabel(r"$r \, [h^{-1} \, \mathrm{Mpc}]$", fontsize=18)
        plt.ylabel(r"$\omega(r)$", fontsize=18)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'learned_{predict}_correlation.png')

    def compute_jacobian(self, input, predict: str):
        self.model.eval()

        x = np.array(input).reshape(1, -1)
        x = self.scaler.transform(x)
        x = torch.tensor(x, requires_grad=True).float()

        # Forward pass through the model
        y_pred = self.model.forward_with_dropout(x)
        
        if predict == 'xi':
            y_pred_to_use = y_pred[:, 0, :20]  # For xi
        elif predict == 'omega':
            y_pred_to_use = y_pred[:, 1, :20]  # For omega
        elif predict == 'eta':
            y_pred_to_use = y_pred[:, 2, :20]  # For eta

        # Create a list to hold the Jacobian
        jacobian = []

        # Compute gradients for each element in y_pred_to_use with respect to the input
        for i in range(y_pred_to_use.shape[-1]):
            grad_output = torch.zeros_like(y_pred_to_use)
            grad_output[:, i] = 1  # Set the gradient for the i-th output to 1
            
            grad = torch.autograd.grad(outputs=y_pred_to_use, 
                                       inputs=x, 
                                       grad_outputs=grad_output, 
                                       retain_graph=True, 
                                       create_graph=True)[0]
            # Detach the gradient from the computation graph and convert to numpy
            jacobian.append(grad.detach().cpu().numpy().flatten())
        
        jacobian = np.array(jacobian)
        return jacobian


    def plot_jacobian(self, input, predict: str):
        """Plot the Jacobian matrix as a heatmap."""
        jacobian = self.compute_jacobian(input, predict)
        
        column_labels = [r'$\mu_{\mathrm{cen}}$', r'$\mu_{\mathrm{sat}}$', r'$\log M_{\mathrm{min}}$', r'$\sigma_{\log M}$', r'$\log M_0$', r'$\log M_1$', r'$\alpha$']
        plt.figure(figsize=(14, 14))
        im = plt.imshow(jacobian, aspect='auto', cmap='cool')
        cbar = plt.colorbar(im)
        cbar.set_label('Partial Derivatives', fontsize=18)  # Bigger color bar label
        cbar.ax.tick_params(labelsize=16)  #
        plt.title(r'Jacobian $\eta$', fontsize=24)
        plt.xticks(ticks=range(len(column_labels)), labels=column_labels, rotation=45, ha="right", fontsize=18)
        plt.yticks(ticks=range(20), labels=range(1, 21), fontsize=18)
        plt.xlabel('Input Parameters', fontsize=20)
        plt.ylabel('Output Bin', fontsize=20)
        plt.savefig(f'jacobian_{predict}.png')
    
        
if __name__ == '__main__':
    
    model = Predict(ResidualMultiTaskEncoderDecoder(),
                        os.path.join(file_dir,'model.pt'),
                        os.path.join(file_dir,'x_train_10x.npy'),
                        os.path.join(file_dir,'y_train_means.npy'),
                        50)
    sample1 = [0.81, 0.35, 12.54, 0.26, 12.68, 13.48, 1.0]
    r_bins = [ 0.11441248,  0.14739182,  0.18987745,  0.24460954,  0.31511813,
           0.40595079,  0.52296592,  0.67371062,  0.8679074 ,  1.11808132,
           1.44036775,  1.85555311,  2.39041547,  3.07945165,  3.96710221,
           5.11061765,  6.58375089,  8.48151414, 10.92630678, 14.07580982]
    sample3 = [0.54, 0.05, 11.61, 0.26, 11.8, 12.6, 1.0]
    sample2 = [0.71, 0.14, 11.93, 0.26, 12.05, 12.85, 1.0]
    
    #optimized_sample, losses = model.gradient_descent(sample1, 'xi', learning_rate=0.01, num_iterations=1000)
    optimized_sample, losses = model.gradient_descent(sample2, 'omega', learning_rate=0.01, num_iterations=1000)
    
    print("Optimized input sample:", optimized_sample)
    print("Initial input sample:", sample2)
    print("Difference between the initial and optimized samples:", np.abs(optimized_sample - sample2))
    #print("Losses during optimization:", losses)

    #model.plot_jacobian(sample1, 'xi')
    #model.plot_jacobian(sample2, 'omega')
    #model.plot_jacobian(sample3, 'eta')
    model.plot_learning_curve(losses)
    model.plot_correlation_function(optimized_sample, sample2, r_bins, 'omega')
    #model.plot_correlation_function(sample2, 'omega')
    #model.plot_correlation_function(sample3, 'eta')
