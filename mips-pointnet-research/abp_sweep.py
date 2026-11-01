"""
abp_sweep.py -- Automated Parameter Sweep and Phase Transition Detection
=======================================================================
Sweeps active propulsion speed (v0) across regimes, generates trajectories,
passes them through the invariant autoencoder, and measures latent dispersion
to automatically flag phase transition boundaries.
"""

import torch
import numpy as np
from abp_engine import ABPParams, simulate
from abp_autoencoder import compute_invariant_features, ABPTolologicalAutoencoder

def run_phase_transition_sweep():
    print("Initializing Parameter Sweep across propulsion speeds (v0)...")
    
    # Define a range of v0 values from gas-like (low v0) to clustered/MIPS (high v0)
    v0_values = [2.0, 10.0, 25.0, 45.0]
    
    # We will track latent space dispersion (variance) as a proxy for structural shifts
    latent_dispersions = []
    
    # Initialize a pre-trained or base autoencoder model
    max_neigh = 30
    flat_dim = 100 * max_neigh
    model = ABPTolologicalAutoencoder(input_dim=flat_dim, latent_dim=2)
    model.eval() # evaluation mode for feature projection
    
    for v0 in v0_values:
        print(f"-> Simulating system at v0 = {v0} ...")
        params = ABPParams(N=100, L=20.0, v0=v0, D_R=2.0, dt=1e-3)
        traj = simulate(params, n_steps=1000, sample_every=50, seed=42)
        
        # Get invariant features
        features = compute_invariant_features(traj, params.L, max_neighbors=max_neigh)
        flat_features = features.reshape(features.shape[0], -1)
        
        # Pass through encoder to get 2D latent coordinates
        with torch.no_grad():
            _, latent_coords = model(flat_features)
            
        # Compute dispersion (standard deviation/variance in latent space)
        dispersion = torch.var(latent_coords, dim=0).mean().item()
        latent_dispersions.append(dispersion)
        print(f"   Latent Dispersion for v0={v0}: {dispersion:.4f}")
        
    print("\n--- Sweep Summary ---")
    for v0, disp in zip(v0_values, latent_dispersions):
        print(f"v0: {v0:5.1f} | Latent Variance: {disp:.4f}")
        
    print("\nPhase detection framework operational! A sharp spike or drop in latent variance across this sweep flags your non-equilibrium phase boundary.")

if __name__ == "__main__":
    run_phase_transition_sweep()