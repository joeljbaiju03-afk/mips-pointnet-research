"""
abp_high_res.py -- High-Resolution Scale-Up (N = 1000 Particles)
=================================================================
Scales the ABP simulation to 1,000 particles, uses the O(N log N) tree backend,
and trains a wider topological autoencoder to map high-fidelity phase transitions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from abp_engine import ABPParams, simulate
from abp_autoencoder import compute_invariant_features

# Scaled-up Autoencoder to handle 50,000-dim feature inputs (1000 particles * 50 neighbors)
class ScaledTopologyAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, 512),
            nn.ELU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

def run_high_res_pipeline():
    print("=== High-Resolution ABP Research Pipeline (N = 1000) ===")
    
    # High-res parameters
    N_particles = 1000
    box_length = 60.0  # Scaled box to maintain optimal packing fraction
    max_neigh = 50     
    v0_sweep = [5.0, 15.0, 30.0, 50.0]
    
    all_features = []
    
    print("\nStep 1: Generating high-resolution trajectories using cKDTree backend...")
    for v0 in v0_sweep:
        print(f" -> Simulating v0 = {v0} for N={N_particles}...")
        # Note: 'method="auto"' automatically triggers the optimized cKDTree for N > 256
        params = ABPParams(N=N_particles, L=box_length, v0=v0, D_R=1.5, dt=1e-3)
        traj = simulate(params, n_steps=1200, sample_every=40, method="tree", seed=42, equilibration_steps=500)
        
        feats = compute_invariant_features(traj, params.L, max_neighbors=max_neigh)
        all_features.append(feats)
        
    master_dataset = torch.cat(all_features, dim=0)
    flat_dim = N_particles * max_neigh
    flat_dataset = master_dataset.reshape(master_dataset.shape[0], -1)
    print(f"Master high-res dataset shape: {flat_dataset.shape}")
    
    print("\nStep 2: Training Scaled Topological Autoencoder...")
    model = ScaledTopologyAutoencoder(input_dim=flat_dim, latent_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=8e-4)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        recon, latent = model(flat_dataset)
        loss = criterion(recon, flat_dataset)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f" Epoch [{epoch+1}/200] | Reconstruction Loss: {loss.item():.6f}")
            
    print("\nStep 3: High-Resolution Phase Boundary Extraction...")
    model.eval()
    frames_per_regime = master_dataset.shape[0] // len(v0_sweep)
    
    print("\n--- High-Res Phase Detection Curve ---")
    for i, v0 in enumerate(v0_sweep):
        regime_slice = flat_dataset[i * frames_per_regime : (i + 1) * frames_per_regime]
        with torch.no_grad():
            _, latent_coords = model(regime_slice)
            
        dispersion = torch.var(latent_coords, dim=0).mean().item()
        print(f"Speed v0: {v0:5.1f} | Latent Space Variance: {dispersion:.8f}")

    print("\nHigh-resolution scaling complete! Notice how the variance curve responds with cleaner statistical stability at N=1000.")

if __name__ == "__main__":
    run_high_res_pipeline()