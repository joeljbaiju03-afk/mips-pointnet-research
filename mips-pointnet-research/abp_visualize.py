"""
abp_visualize.py -- Publication-Quality Figure Generation
==========================================================
Generates multi-panel figures illustrating the real-space phase states,
latent space embeddings, and the variance phase-transition signature.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from abp_engine import ABPParams, simulate
from abp_autoencoder import compute_invariant_features, ABPTolologicalAutoencoder

def generate_research_figures():
    print("Generating simulation regimes for visualization...")
    
    # Run two key regimes: Gas phase (low v0) and Clustered/MIPS phase (high v0)
    params_gas = ABPParams(N=300, L=35.0, v0=5.0, D_R=2.0, dt=1e-3)
    params_cluster = ABPParams(N=300, L=35.0, v0=40.0, D_R=2.0, dt=1e-3)
    
    traj_gas = simulate(params_gas, n_steps=1000, sample_every=50, seed=42)
    traj_cluster = simulate(params_cluster, n_steps=1000, sample_every=50, seed=42)
    
    # Setup model for feature extraction & projection
    max_neigh = 30
    flat_dim = 300 * max_neigh
    model = ABPTolologicalAutoencoder(input_dim=flat_dim, latent_dim=2)
    model.eval()
    
    # Compute features and project latent coordinates
    feats_gas = compute_invariant_features(traj_gas, params_gas.L, max_neighbors=max_neigh)
    feats_cluster = compute_invariant_features(traj_cluster, params_cluster.L, max_neighbors=max_neigh)
    
    with torch.no_grad():
        _, latent_gas = model(feats_gas.reshape(feats_gas.shape[0], -1))
        _, latent_cluster = model(feats_cluster.reshape(feats_cluster.shape[0], -1))
        
    # --- Plotting Figure Layout ---
    fig = plt.figure(figsize=(15, 5))
    
    # Panel 1: Real-Space Gas Phase
    ax1 = fig.add_subplot(131)
    final_gas = traj_gas[-1]
    ax1.scatter(final_gas[:, 0], final_gas[:, 1], s=15, c='royalblue', alpha=0.8)
    ax1.set_title(f"Gas Regime ($v_0$ = {params_gas.v0})")
    ax1.set_xlim(0, params_gas.L)
    ax1.set_ylim(0, params_gas.L)
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Panel 2: Real-Space Clustered Phase (MIPS)
    ax2 = fig.add_subplot(132)
    final_cluster = traj_cluster[-1]
    ax2.scatter(final_cluster[:, 0], final_cluster[:, 1], s=15, c='crimson', alpha=0.8)
    ax2.set_title(f"MIPS Regime ($v_0$ = {params_cluster.v0})")
    ax2.set_xlim(0, params_cluster.L)
    ax2.set_ylim(0, params_cluster.L)
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Panel 3: 2D Latent Space Dispersion Curve (Phase Signature)
    ax3 = fig.add_subplot(133)
    v0_sweep = [2.0, 10.0, 25.0, 40.0]
    variances = []
    
    for v in v0_sweep:
        p = ABPParams(N=300, L=35.0, v0=v, D_R=2.0, dt=1e-3)
        t = simulate(p, n_steps=800, sample_every=40, seed=42)
        f = compute_invariant_features(t, p.L, max_neighbors=max_neigh)
        with torch.no_grad():
            _, z = model(f.reshape(f.shape[0], -1))
        variances.append(torch.var(z, dim=0).mean().item())
        
    ax3.plot(v0_sweep, variances, marker='o', color='darkviolet', linewidth=2, markersize=6)
    ax3.set_title("Unsupervised Phase Transition Signature")
    ax3.set_xlabel("Self-Propulsion Speed ($v_0$)")
    ax3.set_ylabel("Latent Space Variance")
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_filename = "abp_phase_transition_figure.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Figure successfully saved as '{output_filename}'!")
    plt.show()

if __name__ == "__main__":
    generate_research_figures()