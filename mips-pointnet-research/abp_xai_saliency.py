"""
abp_xai_saliency.py -- Explainable AI (XAI) Gradient Saliency Analysis
======================================================================
Computes input gradients through the PointNet bottleneck to determine 
which neighbor configurations and distances dominate the phase transition signature.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from abp_engine import ABPParams, simulate
from abp_pointnet_autoencoder import compute_pointcloud_features, PointNetAutoencoder

def run_saliency_analysis():
    print("=== Initializing XAI Gradient Saliency & Attribution Pipeline ===")
    
    N_particles = 100
    L_box = 30.0
    max_neigh = 20
    flat_recon_dim = N_particles * max_neigh * 2
    
    # 1. Initialize model
    model = PointNetAutoencoder(latent_dim=2, output_dim=flat_recon_dim)
    model.eval()
    
    # 2. Simulate trajectory at the MIPS transition regime
    print("Simulating active matter trajectory for saliency tracking...")
    params = ABPParams(N=N_particles, L=L_box, v0=35.0, D_R=2.0, dt=1e-3)
    traj = simulate(params, n_steps=400, sample_every=30, seed=42)
    
    # 3. Extract point cloud features and enable gradient tracking
    print("Extracting point cloud tensors and activating gradient tracking...")
    pc_tensor = compute_pointcloud_features(traj, L=L_box, max_neighbors=max_neigh)
    pc_tensor = pc_tensor.clone().detach().requires_grad_(True)
    
    # 4. Forward pass
    recon, latent = model(pc_tensor)
    
    # Select latent dimension 0 to analyze gradient sensitivity
    target_latent_dim = 0
    latent_score = latent[:, target_latent_dim].sum()
    
    # 5. Backward pass to compute input gradients: d(latent) / d(dx, dy)
    model.zero_grad()
    latent_score.backward()
    
    # Extract gradients
    gradients = pc_tensor.grad.detach().numpy() # Shape: [Frames, N, K, 2]
    
    # Compute saliency magnitude: sqrt(grad_dx^2 + grad_dy^2)
    saliency_map = np.sqrt(np.sum(gradients**2, axis=-1)) # Shape: [Frames, N, K]
    
    # Extract corresponding neighbor distances for correlation analysis
    print("Correlating feature attribution with physical radial distances...")
    distances_list = []
    saliency_list = []
    
    n_frames, N, K, _ = pc_tensor.shape
    pc_np = pc_tensor.detach().numpy()
    
    for f in range(n_frames):
        for i in range(N):
            for k in range(K):
                dx = pc_np[f, i, k, 0]
                dy = pc_np[f, i, k, 1]
                r = np.sqrt(dx**2 + dy**2)
                if r > 0: # Exclude padding zeros
                    distances_list.append(r)
                    saliency_list.append(saliency_map[f, i, k])
                    
    # 6. Plot Feature Attribution vs. Radial Distance
    print("Generating XAI feature attribution spectrum...")
    plt.figure(figsize=(9, 5.5))
    
    plt.scatter(distances_list, saliency_list, alpha=0.3, color='#1f77b4', s=15, label='Neighbor Attribution')
    plt.title(f"XAI Gradient Saliency Attribution ($z_{target_latent_dim}$ Sensitivity)", fontsize=12, pad=12)
    plt.xlabel("Local Neighbor Distance ($r = \\sqrt{\\Delta x^2 + \\Delta y^2}$)", fontsize=11)
    plt.ylabel("Gradient Saliency Magnitude $\\|\\partial z / \\partial \\mathbf{r}_{ij}\\|$", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.axvline(x=1.122, color='#d62728', linestyle='--', linewidth=2, label='WCA Core Diameter ($2^{1/6} \\sigma$)')
    plt.legend(frameon=True, facecolor='white', loc='upper right')
    plt.tight_layout()
    
    output_filename = "abp_xai_saliency_attribution.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nXAI saliency attribution plot successfully saved as '{output_filename}'!")
    plt.show()

if __name__ == "__main__":
    run_saliency_analysis()