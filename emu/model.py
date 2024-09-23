import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(7, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        
        # Add bottleneck layer
        self.bottleneck = nn.Linear(2048, 200)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)  # Bottleneck
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # Add Conv1D layers starting from the bottleneck output
        self.decoder = nn.Sequential(
            nn.Conv1d(1, 20, kernel_size=3, stride=1, padding=1),  # First Conv1D layer
            nn.BatchNorm1d(20),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(20, 40, kernel_size=3, stride=1, padding=1),  # Second Conv1D layer
            nn.BatchNorm1d(40),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(40, 80, kernel_size=5, stride=1, padding=2),  # Third Conv1D layer
            nn.BatchNorm1d(80),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(80, 20, kernel_size=3, stride=5, padding=1),  # Fourth Conv1D layer with stride 5
            nn.BatchNorm1d(20),
            nn.LeakyReLU(),
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv1D
        return self.decoder(x)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Test the autoencoder
if __name__ == "__main__":
    autoencoder = Autoencoder()
    
    # Test with a batch size of 2 and input size of 7
    input_data = torch.randn(2, 7)
    output_data = autoencoder(input_data)
    
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output_data.shape}")

