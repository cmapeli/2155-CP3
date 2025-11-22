import torch
import torch.nn.functional as F
import numpy as np

# --- VAE Loss Function with Free Bits and Weighting ---
def vae_loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, 
                      logvar: torch.Tensor, mask: torch.Tensor, beta: float = 1.0, 
                      free_bits: float = 0.5):
    """
    Enhanced VAE loss function with missing value handling and KL Free Bits.
    
    Args:
        recon_x: Reconstructed data
        x: Original data (ground truth)
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        mask: Binary mask (1 for observed, 0 for missing)
        beta: Weight for KL divergence term (annealing schedule)
        free_bits: The maximum number of bits the KL loss is allowed to drop to (prevents collapse).
    """

    # 1. Reconstruction Loss (MSE on OBSERVED values only)
    # The reconstruction should match the ground truth on known values.
    # We use a combined loss for robustness, as in the original.
    reconstruction_diff = (recon_x - x) ** 2
    
    # Only consider observed values
    masked_loss = reconstruction_diff * mask
    
    # Normalize by the count of observed values
    recon_loss_masked = masked_loss.sum() / (mask.sum() + 1e-8)
    
    # Add stability term (optional, but kept from your original code)
    # standard_recon_loss = F.mse_loss(recon_x * mask, x * mask, reduction='mean')
    # recon_loss = 0.7 * recon_loss_masked + 0.3 * standard_recon_loss
    
    # Let's simplify and rely purely on the normalized masked loss for the improvement strategy
    recon_loss = recon_loss_masked

    # 2. KL Divergence Loss
    # KL(N(mu, exp(logvar)^2) || N(0, 1))
    kl_loss_raw = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss_raw = kl_loss_raw / x.size(0) # Normalize by batch size
    
    # Apply KL Free Bits (Crucial Improvement)
    # If the KL loss is smaller than 'free_bits', it is treated as 'free_bits'
    # This prevents the KL term from forcing mu=0, logvar=0 (posterior collapse)
    kl_loss_free_bits = torch.clamp(kl_loss_raw, min=free_bits)

    # 3. Total Loss (with Beta-Annealing)
    total_loss = recon_loss + beta * kl_loss_free_bits
    
    # Return raw KL for monitoring purposes
    return total_loss, recon_loss, kl_loss_raw

# --- Beta Scheduling (Unchanged, included for completeness) ---
def get_beta_schedule(epoch, total_epochs, schedule_type='cosine'):
    """Get beta value for KL annealing schedule."""
    if schedule_type == 'linear':
        return min(1.0, epoch / (total_epochs * 0.5))
    elif schedule_type == 'sigmoid':
        return 1.0 / (1.0 + np.exp(-(epoch - total_epochs * 0.5) / (total_epochs * 0.1)))
    elif schedule_type == 'cosine':
        # This is a good aggressive schedule, kept from your original code
        return 0.5 * (1 + np.cos(np.pi * (1 - epoch / total_epochs)))
    elif schedule_type == 'constant':
        return 1.0
    else:
        return 1.0
        
# Placeholder for data evaluation (assuming evaluate_imputation is available in the env)
# The full training script will import and use this.