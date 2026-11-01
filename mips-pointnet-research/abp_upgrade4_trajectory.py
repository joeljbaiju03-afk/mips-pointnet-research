"""
abp_upgrade4_trajectory.py -- Latent Space Trajectory Tracking & Phase Space Flow
=================================================================================
Tracks continuous simulation paths through the 2D bottleneck manifold (z1 vs z2),
visualizing how the system's internal representation flows over time.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from abp_engine import ABPParams, simulate
from abp_pointnet_autoencoder import compute_pointcloud_features, PointNetAutoencoder

def run_latent_trajectory_analysis():
    print("=== Initializing Upgrade 4: Latent Space Trajectory Flow Pipeline ===")
    
    N_particles = 100
    L_box = 30.0
    max_neigh = 20
    flat_recon_dim = N_particles * max_neigh * 2
    
    # Initialize trained model
    model = PointNetAutoencoder(latent_dim=2, output_dim=flat_recon_dim)
    model.eval()
    
    # Simulate a dynamic trajectory undergoing active transitions
    print("Simulating continuous dynamical trajectory...")
    params = ABPParams(N=N_particles, L=L_box, v0=30.0, D_R=2.0, dt=1e-3)
    traj = simulate(params, n_steps=800, sample_every=15, seed=42)
    
    print("Extracting sequential PointNet latent coordinates...")
    pc_tensor = compute_pointcloud_features(traj, L=L_box, max_neighbors=max_neigh)
    
    with torch.no_grad():
        _, latent_coords = model(pc_tensor)
        latent_coords = latent_coords.numpy() # Shape: [n_frames, 2]
        
    z1 = latent_coords[:, 0]
    z2 = latent_coords[:, 1]
    n_points = len(z1)
    
    print("Plotting phase space flow trajectory...")
    plt.figure(figsize=(9, 6))
    
    # Create segments for LineCollection colored by time/step
    points = np.array([z1, z2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Colormap representing temporal progression
    norm = plt.Normalize(0, n_points)
    lc = LineCollection(segments, cmap='plasma', norm=norm, linewidth=2.5)
    lc.set_array(np.arange(n_points))
    
    line = plt.gca().add_collection(lc)
    plt.colorbar(line, label='Simulation Time Steps (Progress)')
    
    # Highlight start and end points
    plt.scatter(z1[0], z2[0], color='green', s=100, zorder=5, label='Start (Initial State)')
    plt.scatter(z1[-1], z2[-1], color='red', s=100, zorder=5, label='End (Equilibrated State)')
    
    # Add directional arrows along the trajectory
    skip = max(1, n_points // 15)
    plt.quiver(z1[:-1:skip], z2[:-1:skip], 
               z1[1::skip] - z1[:-1:skip], z2[1::skip] - z2[:-1:skip], 
               angles='xy', scale_units='xy', scale=1, color='#333333', alpha=0.6, width=0.003)
    
    plt.xlim(z1.min() - 0.1, z1.max() + 0.1)
    plt.ylim(z2.min() - 0.1, z2.max() + 0.1)
    
    plt.title("Latent Space Trajectory Flow ($z_1$ vs $z_2$ Dynamics)", fontsize=12, pad=12)
    plt.xlabel("Latent Coordinate $z_1$", fontsize=11)
    plt.ylabel("Latent Coordinate $z_2$", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(frameon=True, facecolor='white', loc='best')
    plt.tight_layout()
    
    output_filename = "abp_upgrade4_latent_trajectory.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nLatent trajectory flow plot successfully saved as '{output_filename}'!")
    plt.show()

if __name__ == "__main__":
    run_latent_trajectory_analysis()