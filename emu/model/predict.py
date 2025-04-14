import torch
from utils import IADatasetNew, ResidualMultiTaskEncoderDecoder
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

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
        x = torch.tensor(x).float()
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
        
        
if __name__ == '__main__':
    
    model = Predict(ResidualMultiTaskEncoderDecoder(),
                        os.path.join(file_dir,'model.pt'),
                        os.path.join(file_dir,'x_train_10x.npy'),
                        os.path.join(file_dir,'y_train_means.npy'),
                        50)
    
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
    ax.set_xlabel(r"$r \, [h^{-1} \, \mathrm{Mpc}]$", fontsize=18)
    ax.set_ylabel(r"$\xi(r)$", fontsize=18)
    ax.set_title(r"$\xi$", fontsize=20)
    ax.legend()

    ax = axs[1]
    ax.errorbar(r_bins, pred2, yerr=var2, label='aleoatoric', color='blue', linewidth=4)
    ax.errorbar(r_bins, pred2, yerr=epi_var2, label='epistemic', color='pink', linewidth=4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"$r \, [h^{-1} \, \mathrm{Mpc}]$", fontsize=18)
    ax.set_ylabel(r"$\omega(r)$", fontsize=18)
    ax.set_title(r"$\omega$", fontsize=20)
    # ax.legend()

    ax = axs[2]
    ax.errorbar(r_bins, pred3, yerr=var3, label='aleoatoric', color='blue', linewidth=4)
    ax.errorbar(r_bins, pred3, yerr=epi_var3, label='epistemic', color='pink', linewidth=4)
    ax.set_xscale('log')
    ax.set_xlabel(r"$r \, [h^{-1} \, \mathrm{Mpc}]$", fontsize=18)
    ax.set_ylabel(r"$\eta(r)$", fontsize=18)
    ax.set_title(r"$\eta$", fontsize=20)
    # ax.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig('three_panel_correlation_plot.png')
    
