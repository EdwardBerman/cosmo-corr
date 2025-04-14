from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class IADatasetNew(Dataset):
    def __init__(self, inputs: List, outputs: List):
        super(IADatasetNew, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        self.outputs = torch.tensor(outputs, dtype=torch.float32).to(self.device)

        # Create a mask for samples with non-negative outputs
        negative_mask = torch.any(self.outputs[:, 0, :] < 0, dim=1)
        clean_mask = ~negative_mask

        # Filter inputs and outputs using the mask
        self.clean_outputs = self.outputs[clean_mask]
        self.clean_inputs = self.inputs[clean_mask]

        # Apply logarithm to the first sequence
        self.clean_outputs[:, 0, :] = torch.log(self.clean_outputs[:, 0, :])

        # Standardize inputs
        self.input_scaler = StandardScaler()
        clean_inputs_cpu = self.clean_inputs.cpu().numpy()
        self.inputs = torch.tensor(
            self.input_scaler.fit_transform(clean_inputs_cpu), dtype=torch.float32
        ).to(self.device)

        # Standardize outputs
        self.output_scalers = [StandardScaler() for _ in range(outputs.shape[1])]
        clean_outputs_normalized = []

        for i in range(outputs.shape[1]):
            clean_outputs_cpu = self.clean_outputs[:, i, :].cpu().numpy()
            scaled_output = (
                self.output_scalers[i]
                .fit_transform(clean_outputs_cpu.reshape(-1, 1))
                .reshape(self.clean_outputs[:, i, :].shape)
                .astype(np.float32)
            )
            clean_outputs_normalized.append(
                torch.tensor(scaled_output, dtype=torch.float32).to(self.device)
            )

        self.targets = torch.stack(clean_outputs_normalized, axis=1).to(self.device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        output_sample = self.targets[idx]

        return input_sample, output_sample

    def inverse_transform(self, scaled_outputs, keeplog=False):
        original_outputs = torch.empty_like(scaled_outputs, dtype=torch.float32).to(
            self.device
        )

        for sequence_index in range(3):
            scaler = self.output_scalers[sequence_index]
            scale = torch.tensor(scaler.scale_, dtype=torch.float32).to(self.device)
            mean = torch.tensor(scaler.mean_, dtype=torch.float32).to(self.device)

            if sequence_index == 0:
                log_means_scaled = scaled_outputs[:, sequence_index, :20]
                # log_means_original = torch.tensor(scaler.inverse_transform(log_means_scaled.detach().cpu().numpy()))
                log_means_original = log_means_scaled * scale + mean
                self.original_means = log_means_original
                self.means1 = log_means_original

                if not keeplog:
                    self.original_means = torch.exp(log_means_original)

                log_variances_scaled = scaled_outputs[:, sequence_index, 20:]
                sigma_squared = scale**2  # Convert sigma^2 to tensor
                adjusted_variances = log_variances_scaled * sigma_squared

                if not keeplog:
                    adjusted_variances = self.original_means * adjusted_variances

            else:
                means_scaled = scaled_outputs[:, sequence_index, :20]
                # self.original_means = torch.tensor(scaler.inverse_transform(means_scaled.detach().cpu().numpy()))
                self.original_means = means_scaled * scale + mean

                variances_scaled = scaled_outputs[:, sequence_index, 20:]
                sigma_squared = scale**2  # Convert sigma^2 to tensor
                adjusted_variances = variances_scaled * sigma_squared

            original_outputs[:, sequence_index, :20] = self.original_means
            original_outputs[:, sequence_index, 20:] = adjusted_variances

        return original_outputs

    def inverse_mc_variance(self, scaled_variances, keeplog=False):
        original_variances = torch.empty_like(scaled_variances, dtype=torch.float32).to(
            self.device
        )

        for sequence_index in range(3):
            scaler = self.output_scalers[sequence_index]
            scale = torch.tensor(
                scaler.scale_, dtype=scaled_variances.dtype, device=self.device
            )

            if sequence_index == 0:
                sigma_squared = scale**2  # Convert sigma^2 to tensor

                log_variances_scaled = scaled_variances[:, sequence_index, :]
                adjusted_variances = log_variances_scaled * sigma_squared

                if not keeplog:
                    adjusted_variances = self.means1 * adjusted_variances

            else:
                sigma_squared = scale**2  # Convert sigma^2 to tensor

                variances_scaled = scaled_variances[:, sequence_index, :]
                adjusted_variances = variances_scaled * sigma_squared

            original_variances[:, sequence_index, :] = adjusted_variances

        return original_variances


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 128,
        bottleneck_dim: int = 128,
        dropout: float = 0.2,
    ):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, int(2 * hidden_dim)),
            nn.BatchNorm1d(int(2 * hidden_dim)),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(2 * hidden_dim), int(4 * hidden_dim)),
            nn.BatchNorm1d(int(4 * hidden_dim)),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(4 * hidden_dim), bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)

    def forward_with_dropout(self, x):
        # Enable dropout
        for layer in self.layers:
            if isinstance(layer, nn.Dropout):
                layer.train(True)
        return self.layers(x)


class ResidualEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 128,
        bottleneck_dim: int = 128,
        dropout: float = 0.2,
    ):
        super(ResidualEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
        )
        # Dimension matching from 2*hidden_dim to 4*hidden_dim
        self.match_dim1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.Dropout(dropout),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.Dropout(dropout),
        )
        # Final layer
        self.final_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, bottleneck_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(bottleneck_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.layer1(x)
        x2 = self.layer2(x)
        identity1 = self.match_dim1(
            x2
        )  # Match dimensions for the first residual connection
        x3 = self.layer3(x2)
        x3 += identity1  # Add the first residual connection
        x4 = self.layer4(x3)
        identity2 = x3  # Use output from layer3 as the second skip connection
        x4 += identity2  # Add the second residual connection
        x_final = self.final_layer(x4)
        return x_final

    def forward_with_dropout(self, x):
        # Enable dropout
        for layer in [
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.final_layer,
        ]:
            for sub_layer in layer:
                if isinstance(sub_layer, nn.Dropout):
                    sub_layer.train(True)
        return self.forward(x)


class ResidualDecoder(nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 2, dropout: float = 0.2):
        super(ResidualDecoder, self).__init__()

        self.initial_layer = nn.Linear(input_dim, 700)
        self.initial_bn = nn.BatchNorm1d(700)
        self.initial_relu = nn.LeakyReLU()
        self.initial_dropout = nn.Dropout(dropout)

        # Define the convolutional layers with potential for residual connections
        self.conv1 = nn.Conv1d(in_channels=7, out_channels=14, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(14)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            in_channels=14, out_channels=28, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm1d(28)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(
            in_channels=28, out_channels=28, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm1d(28)
        self.relu3 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.conv4 = nn.Conv1d(
            in_channels=28, out_channels=56, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm1d(56)
        self.relu4 = nn.LeakyReLU()
        self.dropout4 = nn.Dropout(dropout)

        self.conv5 = nn.Conv1d(
            in_channels=56, out_channels=112, kernel_size=3, padding=1
        )
        self.bn5 = nn.BatchNorm1d(112)
        self.relu5 = nn.LeakyReLU()
        self.dropout5 = nn.Dropout(dropout)

        self.conv6 = nn.Conv1d(
            in_channels=112, out_channels=112, kernel_size=3, padding=1
        )
        self.bn6 = nn.BatchNorm1d(112)
        self.relu6 = nn.LeakyReLU()
        self.dropout6 = nn.Dropout(dropout)

        self.conv7 = nn.Conv1d(
            in_channels=112, out_channels=output_dim, kernel_size=5, padding=1, stride=5
        )
        self.conv7.bias.data[1] = 1.0

    def forward(self, x):
        x = self.initial_dropout(
            self.initial_bn(self.initial_relu(self.initial_layer(x)))
        )
        x = x.view(x.size(0), 7, -1)

        #### reverse ordering of batch and relu
        out = self.dropout1(self.bn1(self.relu1(self.conv1(x))))
        out = self.dropout2(self.bn2(self.relu2(self.conv2(out))))

        identity = out
        out = self.dropout3(self.bn3(self.relu3(self.conv3(out))))
        out += identity  # Add skip connection

        out = self.dropout4(self.bn4(self.relu4(self.conv4(out))))
        out = self.dropout5(self.bn5(self.relu5(self.conv5(out))))

        identity = out
        out = self.dropout6(self.bn6(self.relu6(self.conv6(out))))
        out += identity  # Add skip connection

        out = self.conv7(out)
        means, var = out[:, 0], F.softplus(out[:, 1])

        output = torch.cat([means, var], dim=1)
        return output

    def forward_with_dropout(self, x):
        # Activating dropout during inference, usually not recommended
        for layer in [self.dropout1, self.dropout2, self.dropout3, self.dropout4]:
            layer.train(True)
        return self.forward(x)


class ResidualMultiTaskEncoderDecoder(nn.Module):
    def __init__(self):
        super(ResidualMultiTaskEncoderDecoder, self).__init__()
        self.encoder = ResidualEncoder()
        self.decoder1 = ResidualDecoder()
        self.decoder2 = ResidualDecoder()
        self.decoder3 = ResidualDecoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded1 = self.decoder1(encoded)
        decoded2 = self.decoder2(encoded)
        decoded3 = self.decoder3(encoded)

        output = torch.stack([decoded1, decoded2, decoded3], dim=1)

        return output

    def forward_with_dropout(self, x):
        encoded = self.encoder.forward_with_dropout(x)
        decoded1 = self.decoder1.forward_with_dropout(encoded)
        decoded2 = self.decoder2.forward_with_dropout(encoded)
        decoded3 = self.decoder3.forward_with_dropout(encoded)

        output = torch.stack([decoded1, decoded2, decoded3], dim=1)
        return output

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

        # Store scalers for use in predict method
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

