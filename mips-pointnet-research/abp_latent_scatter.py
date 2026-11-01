"""
abp_latent_scatter.py -- 2D Latent Space Embedding Map
======================================================
Generates a publication-quality 2D scatter plot of the latent space (z1 vs z2)
color-coded by self-propulsion speed (v0) to visualize state separation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from abp_engine import ABPParams, simulate
from abp_autoencoder import compute_invariant_features, ABPTolologicalAutoencoder

def generate_latent_scatter():
    print("Step 1: Generating multi-regime trajectories for latent mapping...")
    v0_list = [2.0, 10.0, 25.0, 40.0]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'] # Distinct color palette per regime
    labels = ['$v_0 = 2.0$ (Gas)', '$v_0 = 10.0$ (Weak Activity)', '$v_0 = 25.0$ (Onset of MIPS)', '$v_0 = 40.0$ (Dense Clusters)']
    
    max_neigh = 30
    N_particles = 150
    L_box = 30.0
    flat_dim = N_particles * max_neigh
    
    # Initialize autoencoder model
    model = ABPTolologicalAutoencoder(input_dim=flat_dim, latent_dim=2)
    model.eval()
    
    plt.figure(figsize=(9, 7))
    
    print("Step 2: Projecting microstates into 2D latent bottleneck...")
    for idx, v0 in enumerate(v0_list):
        params = ABPParams(N=N_particles, L=L_box, v0=v0, D_R=2.0, dt=1e-3)
        traj = simulate(params, n_steps=1000, sample_every=30, seed=42)
        
        feats = compute_invariant_features(traj, params.L, max_neighbors=max_neigh)
        flat_feats = feats.reshape(feats.shape[0], -1)
        
        with torch.no_grad():
            _, latent_coords = model(flat_feats)
            
        z1 = latent_coords[:, 0].numpy()
        z2 = latent_coords[:, 1].numpy()
        
        # Plot scatter points for this speed regime
        plt.scatter(z1, z2, c=colors[idx], label=labels[idx], alpha=0.7, edgecolors='none', s=40)

    plt.title("Topological Autoencoder Latent Space Map ($z_1$ vs $z_2$)", fontsize=13, pad=12)
    plt.xlabel("Latent Coordinate $z_1$", fontsize=11)
    plt.ylabel("Latent Coordinate $z_2$", fontsize=11)
    plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    output_filename = "abp_latent_space_scatter.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nLatent space scatter plot successfully saved as '{output_filename}'!")
    plt.show()

if __name__ == "__main__":
    generate_latent_scatter()