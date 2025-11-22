import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Union
from tqdm import tqdm 

class DenoisingMLP(nn.Module):
    """
    Denoising Network (MLP) for Structured Data Diffusion Model.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], num_timesteps: int):
        super(DenoisingMLP, self).__init__()
        
        
        # Time step embedding (B, hidden_dims[0])
        self.time_embedding = nn.Embedding(num_timesteps, hidden_dims[0])
        
        # D: 37, D*3 = 111 (x_t + x_observed + mask)
        data_condition_dim = input_dim * 3 
        
        # Total input dimension for the FIRST Linear layer
        first_layer_input_dim = data_condition_dim + hidden_dims[0] # 111 + 512 = 623

        # --- DIAGNOSTIC PRINT ---
        print("START!!!!!!!!!!!!")
        print(f"\n[DenoisingMLP Init Debug]")
        print(f"Input Dim (D): {input_dim}")
        print(f"Time Emb Dim: {hidden_dims[0]}")
        print(f"Data+Condition Dim (3*D): {data_condition_dim}")
        print(f"First Layer Input Dim (Calculated): {first_layer_input_dim}")
        # ------------------------

        # --- CRITICAL STRUCTURAL FIX: Use explicit layers ---
        
        # 1. First Layer (L1): Takes combined data/time input
        self.fc1 = nn.Linear(first_layer_input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.act1 = nn.SiLU()

        layers = []
        prev_dim = hidden_dims[0] # Start subsequent layers from the output of L1
        
        # 2. Subsequent Hidden Layers (L2, L3, etc.)
        for i in range(1, len(hidden_dims)):
            hidden_dim = hidden_dims[i]
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.SiLU()) 
            prev_dim = hidden_dim

        # 3. Final Output Layer
        self.output_layer = nn.Linear(hidden_dims[-1], input_dim)
        
        self.hidden_layers = nn.Sequential(*layers)
        
    def forward(self, x_t: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: Noisy input data at time t (B, D)
            condition: Observed features concatenated with mask (B, D*2)
            t: Time step index (B,)
        """
        # 1. Time embedding (B, hidden_dims[0])
        t_emb = self.time_embedding(t)
        
        # 2. Concatenate noisy data and condition (B, 3*D = 111)
        combined_data = torch.cat([x_t, condition], dim=1)
        
        # 3. Concatenate all inputs (B, 111 + 512 = 623)
        h = torch.cat([combined_data, t_emb], dim=1)

        # 4. Pass through L1
        h = self.act1(self.bn1(self.fc1(h)))

        # 5. Pass through subsequent hidden layers
        h = self.hidden_layers(h)
        
        # 6. Output noise prediction
        return self.output_layer(h)

class ConditionalDDPM(nn.Module):
    """
    Conditional Denoising Diffusion Probabilistic Model for Vector Imputation.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], num_timesteps: int = 1000, device: torch.device = None):
        super(ConditionalDDPM, self).__init__()
        self.input_dim = input_dim
        self.num_timesteps = num_timesteps
        
        self.device = device if device is not None else torch.device('cpu')

        # 1. Denoising Network (MLP)
        self.denoise_model = DenoisingMLP(input_dim, hidden_dims, num_timesteps)
        
        # --- DIAGNOSTIC PRINT ---
        print(f"First Linear Layer Weight Shape: {self.denoise_model.fc1.weight.shape}")
        print("-" * 30)
        # ------------------------

        # 2. Diffusion Schedule 
        self.betas = self._build_linear_schedule(num_timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def _build_linear_schedule(self, timesteps: int) -> torch.Tensor:
        """Linear schedule for betas: 0.0001 to 0.02"""
        return torch.linspace(1e-4, 2e-2, timesteps)

    def get_index_from_list(self, vals, t, x_shape):
        """Helper to extract correct element from a list based on batch time index t."""
        batch_size = t.shape[0]
        out = vals.cpu().gather(-1, t.cpu()) 
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_process(self, x_start: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: Sample x_t from q(x_t | x_0)
        """
        noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def reverse_sampling_step(self, x_t: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Single reverse step: Denoise x_t to x_{t-1} using the learned noise prediction."""
        
        betas_t = self.get_index_from_list(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        alphas_t = self.get_index_from_list(self.alphas, t, x_t.shape)
        
        predicted_noise = self.denoise_model(x_t, condition, t)
        
        mean = (1.0 / torch.sqrt(alphas_t)) * \
               (x_t - (betas_t / sqrt_one_minus_alphas_cumprod_t) * predicted_noise)
        
        if t.min() == 0:
            return mean
        
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        
        x_prev = mean + torch.sqrt(posterior_variance_t) * noise
        return x_prev

    @torch.no_grad()
    def impute_samples(self, x_start: torch.Tensor, mask: torch.Tensor, n_samples: int = 100) -> np.ndarray:
        """
        Iterative reverse process to generate samples, conditioned on observed values.
        """
        self.eval()
        device = x_start.device
        batch_size = x_start.size(0)
        
        mask_float = mask.float()
        x_observed = x_start * mask_float 
        condition = torch.cat([x_observed, mask_float], dim=1) 
        
        all_generated_samples = []

        for _ in range(n_samples):
            x_t = torch.randn_like(x_start) 

            for t_idx in tqdm(reversed(range(self.num_timesteps)), desc="DDPM Sampling", total=self.num_timesteps):
                t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
                x_t_minus_1 = self.reverse_sampling_step(x_t, condition, t)
                x_t = x_t_minus_1 * (1.0 - mask_float) + x_observed 

            all_generated_samples.append(x_t.cpu().numpy())

        return np.stack(all_generated_samples, axis=1)