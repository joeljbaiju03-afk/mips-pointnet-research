"""
abp_phase_diagram.py -- 2D Phase Diagram Generation (v0 vs Density)
====================================================================
Sweeps a 2D grid of propulsion speeds and particle densities, computes
latent variance via the topological autoencoder, and plots a 2D phase map.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from abp_engine import ABPParams, simulate
from abp_autoencoder import compute_invariant_features, ABPTolologicalAutoencoder

def generate_phase_diagram():
    print("Initializing 2D Phase Diagram Grid Sweep...")
    
    # Grid parameters
    v0_list = [2.0, 10.0, 25.0, 40.0]
    # Varying box lengths to alter density/packing fraction (smaller box = higher density)
    box_lengths = [20.0, 30.0, 45.0] 
    
    max_neigh = 30
    # Base model for latent projection
    model = ABPTolologicalAutoencoder(input_dim=150 * max_neigh, latent_dim=2)
    model.eval()
    
    variance_grid = np.zeros((len(box_lengths), len(v0_list)))
    
    print("\nRunning grid simulations across density and activity axes...")
    for i, L in enumerate(box_lengths):
        for j, v0 in enumerate(v0_list):
            print(f" -> Simulating Box L={L}, v0={v0}...")
            # Keep particle count fixed at 150, change box size to control density
            params = ABPParams(N=150, L=L, v0=v0, D_R=2.0, dt=1e-3)
            traj = simulate(params, n_steps=800, sample_every=40, seed=42)
            
            feats = compute_invariant_features(traj, params.L, max_neighbors=max_neigh)
            flat_feats = feats.reshape(feats.shape[0], -1)
            
            with torch.no_grad():
                _, latent_coords = model(flat_feats)
                
            # Calculate latent variance as the phase indicator metric
            dispersion = torch.var(latent_coords, dim=0).mean().item()
            variance_grid[i, j] = dispersion

    print("\nPlotting 2D Phase Diagram Heatmap...")
    plt.figure(figsize=(8, 6))
    
    # Create heatmap using imshow or pcolormesh
    im = plt.imshow(variance_grid, origin='lower', aspect='auto', cmap='viridis',
                    extent=[v0_list[0], v0_list[-1], box_lengths[0], box_lengths[-1]])
    
    cbar = plt.colorbar(im)
    cbar.set_label('Latent Space Variance (Structural Order / Transition Metric)')
    
    plt.xlabel('Self-Propulsion Speed ($v_0$)')
    plt.ylabel('Box Length ($L$ -- Inversely proportional to Density)')
    plt.title('Unsupervised Non-Equilibrium Phase Diagram')
    
    # Annotate grid points
    for i, L in enumerate(box_lengths):
        for j, v0 in enumerate(v0_list):
            plt.text(v0, L, f"{variance_grid[i, j]:.4f}", ha='center', va='center', color='white', fontsize=8, weight='bold')

    plt.tight_layout()
    output_filename = "abp_phase_diagram_2d.png"
    plt.savefig(output_filename, dpi=300)
    print(f"2D Phase Diagram successfully saved as '{output_filename}'!")
    plt.show()

if __name__ == "__main__":
    generate_phase_diagram()