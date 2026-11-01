"""
abp_train_and_detect.py -- Unified Training & Phase Transition Detection
========================================================================
Trains the topological autoencoder on multi-regime active matter data,
then maps latent space dispersion to detect phase transitions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from abp_engine import ABPParams, simulate
from abp_autoencoder import compute_invariant_features, ABPTolologicalAutoencoder

def main():
    print("Step 1: Generating multi-regime training data across speeds (v0)...")
    v0_values = [2.0, 10.0, 25.0, 45.0]
    
    all_features = []
    max_neigh = 30
    N_particles = 100
    L_box = 20.0
    
    # Collect snapshots across all phase regimes
    for v0 in v0_values:
        params = ABPParams(N=N_particles, L=L_box, v0=v0, D_R=2.0, dt=1e-3)
        traj = simulate(params, n_steps=1500, sample_every=50, seed=42)
        feats = compute_invariant_features(traj, params.L, max_neighbors=max_neigh)
        all_features.append(feats)
        
    # Concatenate all regimes into a single master training tensor: shape (Total_Frames, N, max_neigh)
    master_dataset = torch.cat(all_features, dim=0)
    flat_dim = N_particles * max_neigh
    flat_dataset = master_dataset.reshape(master_dataset.shape[0], -1)
    
    print(f"Master dataset prepared. Shape: {flat_dataset.shape}")
    
    print("\nStep 2: Training the Topological Autoencoder on structural states...")
    model = ABPTolologicalAutoencoder(input_dim=flat_dim, latent_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(250):
        optimizer.zero_grad()
        recon, latent = model(flat_dataset)
        loss = criterion(recon, flat_dataset)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/250] | Reconstruction Loss: {loss.item():.6f}")
            
    print("\nStep 3: Analyzing Latent Dispersions per Regime (Phase Boundary Detection)...")
    model.eval()
    
    frames_per_regime = master_dataset.shape[0] // len(v0_values)
    
    print("\n--- Final Phase Detection Analysis ---")
    for i, v0 in enumerate(v0_values):
        # Slice out the dataset corresponding to this specific v0 regime
        regime_slice = flat_dataset[i * frames_per_regime : (i + 1) * frames_per_regime]
        
        with torch.no_grad():
            _, latent_coords = model(regime_slice)
            
        # Measure variance in the 2D latent space
        dispersion = torch.var(latent_coords, dim=0).mean().item()
        print(f"Propulsion Speed v0: {v0:5.1f} | Latent Space Variance: {dispersion:.6f}")

    print("\nSuccess! Watch how the latent variance jumps or drops across speeds—that inflection point is your unsupervised phase transition signature.")

if __name__ == "__main__":
    main()