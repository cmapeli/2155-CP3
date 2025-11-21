import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Union

class DenoisingMLP(nn.Module):
    """
    Denoising Network (MLP) for Structured Data Diffusion Model.
    This acts as the epsilon-predictor. It predicts the noise added to the data.
    
    Input: noisy_data, conditioning_data (observed_x + mask), time_step
    Output: predicted_noise
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], num_timesteps: int):
        super(DenoisingMLP, self).__init__()
        
        # Time step embedding (used to condition the network on 't')
        self.time_embedding = nn.Embedding(num_timesteps, hidden_dims[0])
        
        # CRITICAL FIX: The total input size is 3 * input_dim (x_t + observed_x + mask)
        input_layer_dim = input_dim * 3 

        layers = []
        prev_dim = input_layer_dim
        
        # Build MLP layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Integrate the time embedding into the first layer's dimension
            current_in_dim = prev_dim
            if i == 0:
                # The first layer must also account for the size of the time embedding
                current_in_dim = input_layer_dim + hidden_dims[0] # Data (111) + Time Embed (512)
            
            # Linear layer
            layers.append(nn.Linear(current_in_dim, hidden_dim))
            
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.SiLU()) # SiLU (Swish) is common in modern diffusion models
            
            prev_dim = hidden_dim

        # Final output layer predicts noise (same dimension as input_dim)
        layers.append(nn.Linear(hidden_dims[-1], input_dim))
        
        # The sequential MLP now handles layers 1 to L-1
        self.data_layers = nn.Sequential(*layers[:-1])
        self.output_layer = layers[-1]
        
    def forward(self, x_t: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: Noisy input data at time t (B, D)
            condition: Observed features concatenated with mask (B, D*2)
            t: Time step index (B,)
        """
        # Time embedding (B, hidden_dims[0])
        t_emb = self.time_embedding(t)
        
        # Concatenate noisy data and condition (B, 3*D)
        combined_data = torch.cat([x_t, condition], dim=1)
        
        # CRITICAL FIX: Concatenate the time embedding with the combined data input
        h = torch.cat([combined_data, t_emb], dim=1)

        # Pass through denoising layers
        h = self.data_layers(h)
        
        return self.output_layer(h)

class ConditionalDDPM(nn.Module):
    """
    Conditional Denoising Diffusion Probabilistic Model for Vector Imputation.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], num_timesteps: int = 1000, device: torch.device = None):
        super(ConditionalDDPM, self).__init__()
        self.input_dim = input_dim
        self.num_timesteps = num_timesteps
        
        # Determine device for internal tensors (defaults to CPU if not provided)
        self.device = device if device is not None else torch.device('cpu')

        # 1. Denoising Network (MLP)
        self.denoise_model = DenoisingMLP(input_dim, hidden_dims, num_timesteps)
        
        # 2. Diffusion Schedule (Linear is simple and common)
        # CRITICAL FIX: Explicitly move the schedule tensors to the determined device
        self.betas = self._build_linear_schedule(num_timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Required pre-calculated terms for fast forward/reverse process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def _build_linear_schedule(self, timesteps: int) -> torch.Tensor:
        """Linear schedule for betas: 0.0001 to 0.02"""
        return torch.linspace(1e-4, 2e-2, timesteps)

    def get_index_from_list(self, vals, t, x_shape):
        """Helper to extract correct element from a list based on batch time index t."""
        batch_size = t.shape[0]
        # Ensure we gather on CPU if vals is on CPU, then move to t's device (which might be CPU)
        out = vals.cpu().gather(-1, t.cpu()) 
        # Reshape to match batch size and input shape (B, D)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_process(self, x_start: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: Sample x_t from q(x_t | x_0)
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        # x_t
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def reverse_sampling_step(self, x_t: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Single reverse step: Denoise x_t to x_{t-1} using the learned noise prediction."""
        
        betas_t = self.get_index_from_list(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        alphas_t = self.get_index_from_list(self.alphas, t, x_t.shape)
        
        # 1. Predict noise epsilon
        predicted_noise = self.denoise_model(x_t, condition, t)
        
        # 2. Calculate mean of q(x_{t-1} | x_t, x_0)
        # The mean mu_t is estimated from x_t and predicted noise:
        # mu_t = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * predicted_noise)
        mean = (1.0 / torch.sqrt(alphas_t)) * \
               (x_t - (betas_t / sqrt_one_minus_alphas_cumprod_t) * predicted_noise)
        
        if t.min() == 0:
            return mean
        
        # 3. Calculate variance and add sampled noise
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        
        x_prev = mean + torch.sqrt(posterior_variance_t) * noise
        return x_prev

    @torch.no_grad()
    def impute_samples(self, x_start: torch.Tensor, mask: torch.Tensor, n_samples: int = 100) -> np.ndarray:
        """
        Iterative reverse process to generate samples, conditioned on observed values.
        This performs the core conditional sampling required for imputation.
        """
        self.eval()
        device = x_start.device
        batch_size = x_start.size(0)
        
        # Define condition (observed data + mask)
        mask_float = mask.float()
        x_observed = x_start * mask_float # Observed values
        condition = torch.cat([x_observed, mask_float], dim=1) 
        
        all_generated_samples = []

        for _ in range(n_samples):
            # 1. Start with pure noise x_T (B, D)
            x_t = torch.randn_like(x_start) 

            # 2. Iteratively denoise from T down to 0
            for t_idx in tqdm(reversed(range(self.num_timesteps)), desc="DDPM Sampling", total=self.num_timesteps):
                t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
                
                # Perform one reverse step: Predict the distribution mean mu(x_t)
                x_t_minus_1 = self.reverse_sampling_step(x_t, condition, t)
                
                # --- CONDITIONING (CRITICAL STEP FOR IMPUTATION) ---
                # At every step, force the imputed sample to respect the observed values.
                # Only the missing (masked=0) features are updated by the DDPM.
                x_t = x_t_minus_1 * (1.0 - mask_float) + x_observed 
                # x_t = imputed sample * missing_mask + observed_data * observed_mask

            all_generated_samples.append(x_t.cpu().numpy())

        # Shape: (batch_size, n_samples, n_features)
        return np.stack(all_generated_samples, axis=1)