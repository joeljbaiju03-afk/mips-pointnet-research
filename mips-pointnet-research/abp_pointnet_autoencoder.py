"""
abp_pointnet_autoencoder.py -- Permutation-Invariant PointNet Autoencoder
========================================================================
Processes local particle neighborhoods as point clouds, using shared MLPs 
and symmetric max-pooling to achieve strict permutation invariance.
Includes periodic boundary coordinate wrapping for cKDTree stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree

def compute_pointcloud_features(traj, L, max_neighbors=30):
    """
    Converts trajectory positions [Frames, N, 2] into a point cloud tensor
    of relative neighbor coordinates with shape [Frames, N, max_neighbors, 2].
    Handles periodic boundary conditions (PBC) and coordinate wrapping.
    """
    n_frames, N, _ = traj.shape
    pointcloud_data = np.zeros((n_frames, N, max_neighbors, 2), dtype=np.float32)
    
    for f in range(n_frames):
        # Wrap positions strictly within [0, L) to satisfy cKDTree boxsize requirements
        pos = traj[f] % L  
        tree = cKDTree(pos, boxsize=L)
        
        for i in range(N):
            distances, indices = tree.query(pos[i], k=min(max_neighbors + 1, N))
            
            # Exclude self (index 0 is the particle itself)
            neighbor_indices = indices[1:] if len(indices) > 1 else [i]
            n_actual = len(neighbor_indices)
            
            if n_actual > 0:
                # Compute minimum image convention relative displacements
                dx = pos[neighbor_indices, 0] - pos[i, 0]
                dy = pos[neighbor_indices, 1] - pos[i, 1]
                dx = dx - L * np.round(dx / L)
                dy = dy - L * np.round(dy / L)
                
                # Assign to point cloud tensor
                fill_len = min(n_actual, max_neighbors)
                pointcloud_data[f, i, :fill_len, 0] = dx[:fill_len]
                pointcloud_data[f, i, :fill_len, 1] = dy[:fill_len]
                
    return torch.tensor(pointcloud_data, dtype=torch.float32)


class PointNetABPEncoder(nn.Module):
    """
    PointNet-based encoder for active matter microstates.
    Input shape: [Batch, N_particles, max_neighbors, 2]
    """
    def __init__(self, latent_dim=2):
        super(PointNetABPEncoder, self).__init__()
        
        # Shared MLP applied to each neighbor coordinate vector (dx, dy)
        self.mlp1 = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        
        # Post-pooling feature transformation layers
        self.fc_global = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Bottleneck latent projections
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        batch_size, N, K, _ = x.shape
        
        # 1. Apply shared MLP to every neighbor point
        x_flat = x.view(-1, 2)
        h = self.mlp1(x_flat)
        h = h.view(batch_size, N, K, 256)
        
        # 2. Symmetric Max Pooling across neighbors dimension (K) -> Permutation Invariant!
        h_pooled = torch.max(h, dim=2)[0]
        
        # 3. Global aggregation across all particles (N) via mean pooling
        h_system = torch.mean(h_pooled, dim=1)
        
        # 4. Dense feature extraction
        features = self.fc_global(h_system)
        
        # 5. Latent coordinates
        latent = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        
        return latent, latent, logvar


class PointNetABPDecoder(nn.Module):
    """
    Decoder network to reconstruct global spatial structure from the latent bottleneck.
    """
    def __init__(self, latent_dim=2, output_dim=100 * 20 * 2):
        super(PointNetABPDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, z):
        return self.decoder(z)


class PointNetAutoencoder(nn.Module):
    def __init__(self, latent_dim=2, output_dim=100 * 20 * 2):
        super(PointNetAutoencoder, self).__init__()
        self.encoder = PointNetABPEncoder(latent_dim=latent_dim)
        self.decoder = PointNetABPDecoder(latent_dim=latent_dim, output_dim=output_dim)

    def forward(self, x):
        latent, mu, logvar = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent