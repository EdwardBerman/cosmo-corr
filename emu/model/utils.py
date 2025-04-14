from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class IADatasetNew(Dataset):
    def __init__(self, inputs: List, outputs: List):
        super(IADatasetNew, self).__init__()
        
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        #self.pos_pos_norm = np.load('/Users/snehpandya/Projects/IAsim/src/data/new_new_new_data/pos_pos_DM.npy', allow_pickle=True).astype(np.float64)
        self.outputs = np.array(outputs, dtype=np.float64)
        
        negative_idxs = np.where(outputs[:, 0, :] < 0)[0]
        clean_outputs = np.delete(self.outputs, negative_idxs, axis=0)
        clean_inputs = np.delete(self.inputs, negative_idxs, axis=0)
        
        # clean_outputs[:,0,:] /= self.pos_pos_norm
        clean_outputs[:,0,:] = np.log(clean_outputs[:,0,:])
        self.clean_outputs = clean_outputs
        
        self.input_scaler = StandardScaler()
        clean_inputs_float64 = np.array(clean_inputs, dtype=np.float64)  # Convert to float64 for scaling
        self.inputs = torch.tensor(self.input_scaler.fit_transform(clean_inputs_float64), dtype=torch.float32)
        
        self.output_scalers = [StandardScaler() for _ in range(outputs.shape[1])]

        clean_outputs_normalized = []
        for i in range(outputs.shape[1]):
            scaled_output = self.output_scalers[i].fit_transform(clean_outputs[:, i, :].reshape(-1, 1)).reshape(clean_outputs[:, i, :].shape).astype(np.float64)
            clean_outputs_normalized.append(scaled_output)
        
        outputs_normalized = np.stack(clean_outputs_normalized, axis=1)
        self.targets = torch.tensor(outputs_normalized, dtype=torch.float32)
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        output_sample = self.targets[idx]
        
        return input_sample, output_sample
    
    def inverse_transform(self, scaled_outputs, keeplog=False):
        
        if isinstance(scaled_outputs, torch.Tensor):
            try:
                scaled_outputs = scaled_outputs.numpy()
            except:
                scaled_outputs = scaled_outputs.detach().numpy()
        
        original_outputs = np.empty_like(scaled_outputs, dtype=np.float64)
        
        for sequence_index in range(3):  
            
            scaler = self.output_scalers[sequence_index]
            
            if sequence_index == 0:
                
                log_means_scaled = scaled_outputs[:, sequence_index, :20]
                log_means_original = scaler.inverse_transform(log_means_scaled)
                self.original_means = log_means_original
                self.means1 = log_means_original
                
                if not keeplog:
                    self.original_means = np.exp(log_means_original)
                
                log_variances_scaled = scaled_outputs[:, sequence_index, 20:]
                sigma_squared = scaler.scale_ ** 2  # Obtain sigma^2 from the scaler
                adjusted_variances = log_variances_scaled * sigma_squared
                
                if not keeplog:
                    # adjusted_variances = np.exp(log_means_original + adjusted_variances / 2) * (np.exp(adjusted_variances) - 1)
                    adjusted_variances = self.original_means * adjusted_variances
                
            else:
                means_scaled = scaled_outputs[:, sequence_index, :20]
                self.original_means = scaler.inverse_transform(means_scaled)
                
                variances_scaled = scaled_outputs[:, sequence_index, 20:]
                sigma_squared = scaler.scale_ ** 2  # Obtain sigma^2 from the scaler
                adjusted_variances = variances_scaled * sigma_squared
            
            original_outputs[:, sequence_index, :20] = self.original_means
            original_outputs[:, sequence_index, 20:] = adjusted_variances
        
        return original_outputs
    
    def inverse_mc_variance(self, scaled_variances, keeplog=False):
        
        if isinstance(scaled_variances, torch.Tensor):
            try:
                scaled_variances = scaled_variances.numpy()
            except:
                scaled_variances = scaled_variances.detach().numpy()
                
            # np.save('./scaled_variances.npy', scaled_variances)
        
        original_variances = np.empty_like(scaled_variances, dtype=np.float64)
        
        for sequence_index in range(3):
            
            scaler = self.output_scalers[sequence_index]
        
            if sequence_index == 0:
                
                sigma_squared = scaler.scale_ ** 2
                
                log_variances_scaled = scaled_variances[:, sequence_index, :]
                adjusted_variances = log_variances_scaled * sigma_squared
                
                if not keeplog:
                    # adjusted_variances = np.exp(log_variances_scaled + adjusted_variances / 2) * (np.exp(adjusted_variances) - 1)
                    adjusted_variances = self.means1 * adjusted_variances
                
            else:
                sigma_squared = scaler.scale_ ** 2
                
                variances_scaled = scaled_variances[:, sequence_index, :]
                adjusted_variances = variances_scaled * sigma_squared
                
            original_variances[:, sequence_index, :] = adjusted_variances
            
        return original_variances
    


class Encoder(nn.Module):
    def __init__(self, input_dim: int = 7, 
                 hidden_dim: int = 128, 
                 bottleneck_dim: int = 128, 
                 dropout: float = 0.2
                 ):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, int(2*hidden_dim)),
            nn.BatchNorm1d(int(2*hidden_dim)),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(2*hidden_dim), int(4*hidden_dim)),
            nn.BatchNorm1d(int(4*hidden_dim)),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(4*hidden_dim), bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
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
    def __init__(self, input_dim: int = 7, 
                 hidden_dim: int = 128, 
                 bottleneck_dim: int = 128, 
                 dropout: float = 0.2):
        super(ResidualEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        # Dimension matching from 2*hidden_dim to 4*hidden_dim
        self.match_dim1 = nn.Linear(hidden_dim, 2*hidden_dim)  
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2*hidden_dim),
            nn.Dropout(dropout)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2*hidden_dim),
            nn.Dropout(dropout)
        )
        # Final layer
        self.final_layer = nn.Sequential(
            nn.Linear(2*hidden_dim, bottleneck_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(bottleneck_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.layer1(x)
        x2 = self.layer2(x)
        identity1 = self.match_dim1(x2)  # Match dimensions for the first residual connection
        x3 = self.layer3(x2)
        x3 += identity1  # Add the first residual connection
        x4 = self.layer4(x3)
        identity2 = x3  # Use output from layer3 as the second skip connection
        x4 += identity2  # Add the second residual connection
        x_final = self.final_layer(x4)
        return x_final
    
    def forward_with_dropout(self, x):
        # Enable dropout
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.final_layer]:
            for sub_layer in layer:
                if isinstance(sub_layer, nn.Dropout):
                    sub_layer.train(True)
        return self.forward(x)


class ResidualDecoder(nn.Module):
    def __init__(self, 
                 input_dim: int = 128, 
                 output_dim: int = 2, 
                 dropout: float = 0.2):
        
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

        self.conv2 = nn.Conv1d(in_channels=14, out_channels=28, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(28)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.conv3 = nn.Conv1d(in_channels=28, out_channels=28, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(28)
        self.relu3 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.conv4 = nn.Conv1d(in_channels=28, out_channels=56, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(56)
        self.relu4 = nn.LeakyReLU()
        self.dropout4 = nn.Dropout(dropout)

        self.conv5 = nn.Conv1d(in_channels=56, out_channels=112, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(112)
        self.relu5 = nn.LeakyReLU()
        self.dropout5 = nn.Dropout(dropout)
        
        self.conv6 = nn.Conv1d(in_channels=112, out_channels=112, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(112)
        self.relu6 = nn.LeakyReLU()
        self.dropout6 = nn.Dropout(dropout)

        self.conv7 = nn.Conv1d(in_channels=112, out_channels=output_dim, kernel_size=5, padding=1, stride=5)
        self.conv7.bias.data[1] = 1.0

    def forward(self, x):
        x = self.initial_dropout(self.initial_bn(self.initial_relu(self.initial_layer(x))))
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