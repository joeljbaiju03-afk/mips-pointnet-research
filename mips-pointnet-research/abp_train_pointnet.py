"""
abp_train_pointnet.py -- End-to-End Training & Evaluation with PointNet
========================================================================
Integrates compute_pointcloud_features() and PointNetAutoencoder() into 
a complete training loop to compute unsupervised MIPS phase variance.
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from abp_engine import ABPParams, simulate
from abp_pointnet_autoencoder import compute_pointcloud_features, PointNetAutoencoder

def train_and_evaluate_pointnet():
    print("=== Initializing PointNet-Based ABP Pipeline ===")
    
    N_particles = 100
    L_box = 30.0
    max_neigh = 20
    flat_recon_dim = N_particles * max_neigh * 2
    
    print("Simulating baseline active matter trajectory for training...")
    params = ABPParams(N=N_particles, L=L_box, v0=25.0, D_R=2.0, dt=1e-3)
    traj = simulate(params, n_steps=600, sample_every=30, seed=42)
    
    print("Extracting permutation-invariant point cloud tensors...")
    pointcloud_tensor = compute_pointcloud_features(traj, L=L_box, max_neighbors=max_neigh)
    
    # Initialize model with matching reconstruction dimension
    model = PointNetAutoencoder(latent_dim=2, output_dim=flat_recon_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    print("Training PointNet autoencoder on local structural geometry...")
    model.train()
    epochs = 25
    for epoch in range(epochs):
        optimizer.zero_grad()
        target_flat = pointcloud_tensor.view(pointcloud_tensor.shape[0], -1)
        
        recon, latent = model(pointcloud_tensor)
        loss = criterion(recon, target_flat)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f" Epoch [{epoch+1}/{epochs}] | Reconstruction Loss: {loss.item():.6f}")
            
    model.eval()
    
    print("\nExecuting phase sweep using the trained PointNet bottleneck...")
    v0_sweep = [2.0, 10.0, 25.0, 40.0]
    latent_variances = []
    
    for v0 in v0_sweep:
        print(f" -> Processing v0 = {v0}")
        eval_params = ABPParams(N=N_particles, L=L_box, v0=v0, D_R=2.0, dt=1e-3)
        eval_traj = simulate(eval_params, n_steps=400, sample_every=40, seed=101)
        
        eval_pc_tensor = compute_pointcloud_features(eval_traj, L=L_box, max_neighbors=max_neigh)
        
        with torch.no_grad():
            _, latent_coords = model(eval_pc_tensor)
            var = torch.var(latent_coords, dim=0).mean().item()
            latent_variances.append(var)
            
    plt.figure(figsize=(8, 5))
    plt.plot(v0_sweep, latent_variances, marker='o', color='#2b5c8f', linewidth=2.5, label='PointNet Latent Variance')
    plt.title("PointNet Unsupervised MIPS Phase Transition Signature", fontsize=12, pad=12)
    plt.xlabel("Self-Propulsion Speed ($v_0$)", fontsize=11)
    plt.ylabel("Latent Space Variance (Order Metric)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    
    output_filename = "abp_pointnet_transition_curve.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nPointNet phase transition curve successfully saved as '{output_filename}'!")
    plt.show()

if __name__ == "__main__":
    train_and_evaluate_pointnet()