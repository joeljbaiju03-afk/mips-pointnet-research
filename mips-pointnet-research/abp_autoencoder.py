"""
abp_autoencoder.py -- Unsupervised Latent Space Encoder for ABP Trajectories
===========================================================================
Takes trajectory outputs from abp_engine.py, builds permutation-invariant
pairwise distance representations, and maps them via a PyTorch autoencoder
to a 2D latent space to detect non-equilibrium phase changes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --------------------------------------------------------------------------- #
# 1. Permutation-Invariant Feature Extraction (Pairwise Distance Matrix)
# --------------------------------------------------------------------------- #
def compute_invariant_features(snapshots: np.ndarray, L: float, max_neighbors: int = 50) -> torch.Tensor:
    """
    Converts raw (N, 3) [x, y, theta] snapshots into rotation-, translation-, 
    and permutation-invariant local structural descriptors (sorted pair distances).
    
    Output shape: (Batch_Size, N, max_neighbors)
    """
    batch_size, N, _ = snapshots.shape
    features = np.empty((batch_size, N, max_neighbors), dtype=np.float32)
    
    for b in range(batch_size):
        pos = snapshots[b, :, :2]
        # Minimum image distance matrix
        d = pos[:, None, :] - pos[None, :, :]
        d -= L * np.round(d / L)
        r = np.sqrt((d ** 2).sum(axis=-1))
        
        # Sort distances for each particle (invariant to particle indexing)
        r_sorted = np.sort(r, axis=1)
        # Take up to max_neighbors (ignoring self-distance at index 0)
        k = min(max_neighbors, N - 1)
        features[b, :, :k] = r_sorted[:, 1:k+1]
        if k < max_neighbors:
            features[b, :, k:] = 0.0 # padding if N < max_neighbors
            
    return torch.tensor(features)


# --------------------------------------------------------------------------- #
# 2. Autoencoder Architecture
# --------------------------------------------------------------------------- #
class ABPTolologicalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 2):
        super().__init__()
        # Encoder: compresses particle-level invariant features down to a global latent vector
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, latent_dim) # Latent bottleneck (e.g., d=2)
        )
        
        # Decoder: reconstructs local distance histograms from latent representation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, 256),
            nn.ELU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


# --------------------------------------------------------------------------- #
# 3. Training Loop Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from abp_engine import ABPParams, simulate
    
    # Generate a small training dataset across different propulsion speeds v0
    print("Generating simulation datasets for latent training...")
    params = ABPParams(N=100, L=20.0, v0=5.0, D_R=2.0, dt=1e-3)
    traj = simulate(params, n_steps=2000, sample_every=100, seed=42)
    
    # Extract invariant features
    max_neigh = 30
    tensor_features = compute_invariant_features(traj, params.L, max_neighbors=max_neigh)
    flat_dim = 100 * max_neigh
    
    # Initialize model, optimizer, and loss
    model = ABPTolologicalAutoencoder(input_dim=flat_dim, latent_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Quick training loop
    model.train()
    dataset_tensor = tensor_features.reshape(tensor_features.shape[0], -1)
    
    print("Training autoencoder on spatial topology snapshots...")
    for epoch in range(150):
        optimizer.zero_grad()
        recon, latent = model(dataset_tensor)
        loss = criterion(recon, dataset_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/150], Reconstuction Loss: {loss.item():.6f}")
            
    # Extract final latent states
    with torch.no_grad():
        _, final_latent = model(dataset_tensor)
    print(f"Latent trajectory matrix shape (Time_Frames, Latent_Dims): {final_latent.shape}")
    print("Successfully mapped microstates into unsupervised latent coordinates!")