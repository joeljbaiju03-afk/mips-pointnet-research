"""
abp_upgrade2_pmax.py -- Dual-Axis Classical vs. Unsupervised Validation
=======================================================================
Computes classical P_max (largest cluster fraction) and unsupervised 
PointNet latent variance across a v0 sweep, plotting them on a dual-axis chart.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from abp_engine import ABPParams, simulate
from abp_pointnet_autoencoder import compute_pointcloud_features, PointNetAutoencoder

def compute_classical_pmax(traj, L, r_cut=1.5):
    """
    Computes the fraction of particles in the largest cluster (P_max) 
    using spatial adjacency graphs and connected components.
    """
    n_frames, N, _ = traj.shape
    pmax_list = []
    
    for f in range(n_frames):
        pos = traj[f] % L
        tree = cKDTree(pos, boxsize=L)
        
        # Find pairs within bonding cutoff r_cut
        pairs = tree.query_pairs(r=r_cut)
        
        if len(pairs) == 0:
            pmax_list.append(1.0 / N)
            continue
            
        row = np.array([p[0] for p in pairs])
        col = np.array([p[1] for p in pairs])
        data = np.ones(len(pairs))
        
        # Build symmetric adjacency matrix
        mat_size = (N, N)
        adj_matrix = coo_matrix((data, (row, col)), shape=mat_size).tocsr()
        
        # Find connected components (clusters)
        n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
        
        # Count sizes of each cluster
        _, counts = np.unique(labels, return_counts=True)
        max_cluster_size = np.max(counts)
        
        pmax_list.append(max_cluster_size / N)
        
    return np.mean(pmax_list)


def run_pmax_latent_correlation():
    print("=== Initializing Upgrade 2: Classical P_max vs Unsupervised Latent Variance ==0")
    
    N_particles = 100
    L_box = 30.0
    max_neigh = 20
    flat_recon_dim = N_particles * max_neigh * 2
    
    # Initialize model
    model = PointNetAutoencoder(latent_dim=2, output_dim=flat_recon_dim)
    model.eval()
    
    v0_sweep = [2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
    
    pmax_values = []
    latent_variances = []
    
    print("\nExecuting multi-regime sweep across propulsion speeds...")
    for v0 in v0_sweep:
        print(f" -> Processing v0 = {v0:4.1f}")
        params = ABPParams(N=N_particles, L=L_box, v0=v0, D_R=2.0, dt=1e-3)
        traj = simulate(params, n_steps=500, sample_every=30, seed=42)
        
        # 1. Compute Classical Order Parameter (P_max)
        pmax = compute_classical_pmax(traj, L=L_box, r_cut=1.5)
        pmax_values.append(pmax)
        
        # 2. Compute Unsupervised Latent Variance
        pc_tensor = compute_pointcloud_features(traj, L=L_box, max_neighbors=max_neigh)
        with torch.no_grad():
            _, latent_coords = model(pc_tensor)
            var = torch.var(latent_coords, dim=0).mean().item()
            latent_variances.append(var)
            
    print("\nGenerating dual-axis validation chart...")
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    
    # Left Axis: Classical P_max
    color = '#1f77b4'
    ax1.set_xlabel('Self-Propulsion Speed ($v_0$)', fontsize=11)
    ax1.set_ylabel('Fraction in Largest Cluster ($P_{max}$)', color=color, fontsize=11)
    line1 = ax1.plot(v0_sweep, pmax_values, color=color, marker='s', linewidth=2.5, label='Classical $P_{max}$')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Right Axis: Unsupervised Latent Variance
    ax2 = ax1.twinx()
    color = '#d62728'
    ax2.set_ylabel('Latent Space Variance (Unsupervised Metric)', color=color, fontsize=11)
    line2 = ax2.plot(v0_sweep, latent_variances, color=color, marker='o', linewidth=2.5, linestyle='--', label='Latent Variance')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True, facecolor='white')
    
    plt.title("Validation: Unsupervised Latent Variance vs. Classical Order Parameter ($P_{max}$)", fontsize=12, pad=12)
    fig.tight_layout()
    
    output_filename = "abp_upgrade2_pmax_correlation.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nDual-axis validation plot successfully saved as '{output_filename}'!")
    plt.show()

if __name__ == "__main__":
    run_pmax_latent_correlation()