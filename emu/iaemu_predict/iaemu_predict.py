import numpy as np
import torch

from model_utils import IADatasetNew, ResidualMultiTaskEncoderDecoder


class Model:
    def __init__(self, model, model_checkpoint, x_train_path, y_train_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model_checkpoint = model_checkpoint
        self.model.load_state_dict(
            torch.load(model_checkpoint, map_location=self.device)
        )
        self.x_train, self.y_train = np.load(x_train_path), np.load(y_train_path)
        self.train_dataset = IADatasetNew(self.x_train, self.y_train)

        # In the Model __init__ method
        self.input_scaler = self.train_dataset.input_scaler
        self.input_scaler_mean = torch.tensor(
            self.input_scaler.mean_, dtype=torch.float32
        ).to(self.device)
        self.input_scaler_std = torch.tensor(
            self.input_scaler.scale_, dtype=torch.float32
        ).to(self.device)

    def predict(self, x, predict: str, return_grad=False, num_passes=1):
        self.model.eval()

        if return_grad:
            for module in self.model.modules():
                if isinstance(module, torch.nn.BatchNorm1d) or isinstance(
                    module, torch.nn.BatchNorm2d
                ):
                    module.eval()

        x = x.to(self.device)
        x = x.reshape(1, -1)
        x = (x - self.input_scaler_mean) / self.input_scaler_std

        if num_passes > 1:
            self.model.train()
            for module in self.model.modules():
                if isinstance(module, torch.nn.BatchNorm1d) or isinstance(
                    module, torch.nn.BatchNorm2d
                ):
                    module.eval()

            preds = []
            for i in range(num_passes):
                y_pred = self.model.forward_with_dropout(x)
                preds.append(y_pred)
            y_pred = torch.stack(preds).to(self.device)
            y_pred_mean = torch.mean(y_pred, dim=0).to(self.device)
            y_pred_mc_var = torch.var(y_pred[:, :, :, :20], dim=0).to(self.device)
        else:
            y_pred = self.model(x)
            y_pred_mean = y_pred
            y_pred_mc_var = torch.zeros_like(y_pred[:, :, :20]).to(self.device)

        scaled_means = self.train_dataset.inverse_transform(y_pred_mean).squeeze()
        scaled_y_var = self.train_dataset.inverse_mc_variance(y_pred_mc_var).squeeze()
        # scaled_y_var = self.train_dataset.inverse_mc_variance(y_pred_mc_var, y_pred_mean[:,:,]).squeeze()

        if predict == "xi":
            means = scaled_means[0, :20]
            aleo_std = torch.sqrt(scaled_means[0, 20:])
            epi_std = torch.sqrt(torch.abs(scaled_y_var[0]))
        elif predict == "omega":
            means = scaled_means[1, :20]
            aleo_std = torch.sqrt(scaled_means[1, 20:])
            epi_std = torch.sqrt(torch.abs(scaled_y_var[1]))
        elif predict == "eta":
            means = scaled_means[2, :20]
            aleo_std = torch.sqrt(scaled_means[2, 20:])
            epi_std = torch.sqrt(torch.abs(scaled_y_var[2]))
        elif predict == "all":
            means = scaled_means[:, :20]
            aleo_std = torch.sqrt(scaled_means[:, 20:])
            epi_std = torch.sqrt(torch.abs(scaled_y_var))

        return means, aleo_std, epi_std


if __name__ == "__main__":
    model = Model(
        ResidualMultiTaskEncoderDecoder(),
        "/home/eddieberman/research/mcclearygroup/AstroCorr/emu/iaemu_predict/iaemu.pt",
        "/home/eddieberman/research/mcclearygroup/AstroCorr/emu/iaemu_predict/data/x_train_10x.npy",
        "/home/eddieberman/research/mcclearygroup/AstroCorr/emu/iaemu_predict/data/y_train_10x.npy",
    )

    sample3 = torch.tensor(
        [0.55, 0.03, 11.61, 0.26, 11.8, 12.6, 1.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    sample2 = torch.tensor(
        [0.68, 0.14, 11.93, 0.26, 12.05, 12.85, 1.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    sample1 = torch.tensor(
        [0.78, 0.33, 12.54, 0.26, 12.68, 13.48, 1.0],
        dtype=torch.float32,
        requires_grad=True,
    )

    pred1, var1 = model.predict(sample1, "all", return_grad=True)
    xi, omega, eta = (
        pred1[0].detach().cpu().numpy(),
        pred1[1].detach().cpu().numpy(),
        pred1[2].detach().cpu().numpy(),
    )
    xi_var, omega_var, eta_var = (
        var1[0].detach().cpu().numpy(),
        var1[1].detach().cpu().numpy(),
        var1[2].detach().cpu().numpy(),
    )
    # xi_epi_var, omega_epi_var, eta_epi_var = epi_var1[0].detach().cpu().numpy(), epi_var1[1].detach().cpu().numpy(), epi_var1[2].detach().cpu().numpy()

    # print(pred1.shape == var1.shape == epi_var1.shape)
    xi_, omega_, eta_ = pred1[0], pred1[1], pred1[2]
    xi_.sum().backward()
    grads = sample1.grad
    print(grads)

    # r_bins = [ 0.11441248,  0.14739182,  0.18987745,  0.24460954,  0.31511813,
    #        0.40595079,  0.52296592,  0.67371062,  0.8679074 ,  1.11808132,
    #        1.44036775,  1.85555311,  2.39041547,  3.07945165,  3.96710221,
    #        5.11061765,  6.58375089,  8.48151414, 10.92630678, 14.07580982]

    # plt.errorbar(r_bins, omega, yerr=omega_var, label='aleatoric')
    # plt.errorbar(r_bins, omega, yerr=omega_epi_var, label='epistemic')
    # plt.xscale('log')
    # # print('gradients', grad1)
    # plt.show()
