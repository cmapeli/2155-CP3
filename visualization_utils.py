import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def plot_prediction_scatter(imputations: np.ndarray, originals: np.ndarray, masks: np.ndarray, 
                            feature_names: list, n_features: int = 25):
    """
    Plots the imputed value vs. the ground truth value for features that had missing data.
    
    Args:
        imputations (np.ndarray): Array of imputed data (transformed scale).
        originals (np.ndarray): Array of original data (transformed scale).
        masks (np.ndarray): Binary mask (1=Observed, 0=Missing).
        feature_names (list): List of feature names.
        n_features (int): Number of features to plot.
    """
    N_COLS = 5
    N_ROWS = int(np.ceil(n_features / N_COLS))
    
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(4 * N_COLS, 4 * N_ROWS))
    fig.suptitle(f'Predicted vs Ground Truth Values (Missing Positions Only, N={n_features} Features)', 
                 fontsize=16, y=1.02)
    axes = axes.flatten()
    
    # Filter features that had missing values
    missing_features_indices = [i for i in range(imputations.shape[1]) if (masks[:, i] == 0).sum() > 0]
    plot_indices = missing_features_indices[:n_features]

    for plot_idx, feature_idx in enumerate(plot_indices):
        ax = axes[plot_idx]
        
        # Select only the missing values (where mask is 0)
        missing_mask_feat = (masks[:, feature_idx] == 0)
        true_values = originals[missing_mask_feat, feature_idx]
        predicted_values = imputations[missing_mask_feat, feature_idx]
        
        if len(true_values) < 5:
            ax.axis('off')
            continue
            
        # Scatter plot
        sns.scatterplot(x=true_values, y=predicted_values, ax=ax, s=15, alpha=0.6, color='b')
        
        # Calculate metrics
        mse = mean_squared_error(true_values, predicted_values)
        corr, _ = pearsonr(true_values, predicted_values)
        
        # Add identity line
        min_val = min(true_values.min(), predicted_values.min())
        max_val = max(true_values.max(), predicted_values.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('Ground Truth (Transformed)', fontsize=10)
        ax.set_ylabel('DDPM Imputation (Mean)', fontsize=10)
        ax.set_title(f'{feature_names[feature_idx]} | Corr={corr:.3f}, MSE={mse:.4f}', 
                     fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    # Hide unused axes
    for i in range(len(plot_indices), N_ROWS * N_COLS):
        axes[i].axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

def plot_distribution_comparison(imputations: np.ndarray, originals: np.ndarray, masks: np.ndarray, 
                                 feature_names: list, n_features: int = 25):
    """
    Plots the distribution (histogram/KDE) of the imputed values against the ground truth values.
    
    Args:
        imputations (np.ndarray): Array of imputed data (transformed scale).
        originals (np.ndarray): Array of original data (transformed scale).
        masks (np.ndarray): Binary mask (1=Observed, 0=Missing).
        feature_names (list): List of feature names.
        n_features (int): Number of features to plot.
    """
    N_COLS = 5
    N_ROWS = int(np.ceil(n_features / N_COLS))
    
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(4 * N_COLS, 3 * N_ROWS))
    fig.suptitle(f'Distribution Comparison: Dataset vs DDPM Imputed Values (N={n_features} Features)', 
                 fontsize=16, y=1.02)
    axes = axes.flatten()
    
    # Filter features that had missing values
    missing_features_indices = [i for i in range(imputations.shape[1]) if (masks[:, i] == 0).sum() > 0]
    plot_indices = missing_features_indices[:n_features]

    for plot_idx, feature_idx in enumerate(plot_indices):
        ax = axes[plot_idx]
        
        # Select only the missing values (where mask is 0)
        missing_mask_feat = (masks[:, feature_idx] == 0)
        true_values = originals[missing_mask_feat, feature_idx]
        predicted_values = imputations[missing_mask_feat, feature_idx]
        
        if len(true_values) < 5:
            ax.axis('off')
            continue
        
        # Plot distributions using KDE
        sns.histplot(true_values, kde=True, ax=ax, color="skyblue", label="Ground Truth", 
                     stat="density", alpha=0.6, linewidth=1, bins=20)
        
        sns.histplot(predicted_values, kde=True, ax=ax, color="red", label="Imputation Mean", 
                     stat="density", alpha=0.6, linewidth=1, bins=20)
        
        ax.set_title(f'{feature_names[feature_idx]}', fontsize=10)
        ax.set_xlabel('Value (Transformed)', fontsize=8)
        ax.tick_params(labelsize=8)
        
        # Add legend once
        if plot_idx == 0:
            ax.legend(fontsize=8)

    # Hide unused axes
    for i in range(len(plot_indices), N_ROWS * N_COLS):
        axes[i].axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()