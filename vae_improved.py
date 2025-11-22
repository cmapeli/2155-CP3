import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Union

class VAE(nn.Module):
    """
    Variational Autoencoder designed for conditional missing value imputation.
    
    Improvements:
    1. Stronger conditioning: Latent code 'z' is conditioned on observed features.
    2. Enhanced Decoder Input: Concatenates latent code, observed values, and positional encoding.
    """
    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dims: List[int] = [512, 256, 128],
                 use_residual: bool = True, dropout_rate: float = 0.3):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_residual = use_residual
        
        # --- Encoder (Encodes the input data + mask) ---
        # Input dimension is 2 * input_dim (data + mask)
        self.encoder_layers = nn.ModuleList()
        prev_dim = input_dim * 2 
        
        for i, hidden_dim in enumerate(hidden_dims):
            self.encoder_layers.append(nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2), # Use LeakyReLU for potentially better gradient flow
                nn.Dropout(dropout_rate)
            ))
            prev_dim = hidden_dim

        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Initialize Latent Layers with smaller weights for stability
        nn.init.xavier_normal_(self.fc_mu.weight, gain=0.1)
        nn.init.xavier_normal_(self.fc_logvar.weight, gain=0.1)
        nn.init.constant_(self.fc_logvar.bias, -2.0) # Start with Low variance

        # --- Decoder (Decodes latent code, conditioned on observed values) ---
        # Input dimension is latent_dim + input_dim (latent + observed features)
        self.decoder_layers = nn.ModuleList()
        # No positional encoding is added here as in the original, but can be added back
        prev_dim = latent_dim + input_dim 
        reversed_dims = list(reversed(hidden_dims))
        
        for i, hidden_dim in enumerate(reversed_dims):
            self.decoder_layers.append(nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ))
            prev_dim = hidden_dim

        # Final output Layer
        self.output_layer = nn.Linear(hidden_dims[0], input_dim)
        nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list]:
        """Encode input with missing value masking."""
        mask_float = mask.float()
        
        # Input is concatenated data and mask (stronger conditioning)
        encoder_input = torch.cat([x, mask_float], dim=1)

        # Pass through encoder layers
        h = encoder_input
        for i, layer in enumerate(self.encoder_layers):
            prev_h = h
            h = layer(h)
            
            # Add residual connection for deeper Layers
            if self.use_residual and i > 0 and h.shape == prev_h.shape:
                h = h + prev_h
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Clamp Logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        return mu, logvar, [] # Removed skip_connections for simplicity/stability

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, x_observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Decode latent representation conditioned on observed values."""
        mask_float = mask.float()
        x_masked = x_observed * mask_float # Keep only observed values
        
        # Concatenate Latent code with observed values (Conditioning)
        decoder_input = torch.cat([z, x_masked], dim=1)

        # Pass through decoder Layers
        h = decoder_input
        for layer in self.decoder_layers:
            h = layer(h)

        # Get reconstruction
        reconstruction = self.output_layer(h)

        return reconstruction

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE."""
        mu, logvar, _ = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, x, mask)

        return reconstruction, mu, logvar
    
    def impute(self, x_incomplete: torch.Tensor, mask: torch.Tensor, n_samples: int = 10):
        """Generate multiple imputation samples for missing values."""
        self.eval()
        with torch.no_grad():
            mu, logvar, _ = self.encode(x_incomplete, mask)
            samples = []
            
            for _ in range(n_samples):
                z = self.reparameterize(mu, logvar)
                reconstruction = self.decode(z, x_incomplete, mask)
                
                # Combine observed values with imputed values
                mask_float = mask.float()
                # imputed = observed * mask + reconstruction * (1 - mask)
                imputed = x_incomplete * mask_float + reconstruction * (1 - mask_float)
                samples.append(imputed.cpu().numpy())

        return np.stack(samples, axis=1) # Shape: (batch_size, n_samples, n_features)