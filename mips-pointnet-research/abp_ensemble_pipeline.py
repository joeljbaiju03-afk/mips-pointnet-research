"""
abp_ensemble_pipeline.py -- Publication-Grade Ensemble Analysis
=================================================================
Runs multi-seed stochastic ensembles across propulsion speeds, 
computing mean latent variance with standard deviation error ribbons.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from abp_engine import ABPParams, simulate
from abp_autoencoder import compute_invariant_features, ABPTolologicalAutoencoder

def run_ensemble_analysis():
    print("=== Initializing Max-Spec Ensemble Physics Pipeline ===")
    
    v0_sweep = [2.0, 8.0, 15.0, 25.0, 35.0, 45.0]
    seeds = [42, 101, 2026] # Multiple independent stochastic realizations
    max_neigh = 30
    N_particles = 150
    L_box = 30.0
    flat_dim = N_particles * max_neigh
    
    # Initialize model
    model = ABPTolologicalAutoencoder(input_dim=flat_dim, latent_dim=2)
    model.eval()
    
    ensemble_results = {v0: [] for v0 in v0_sweep}
    
    print("\nExecuting multi-seed stochastic sweeps...")
    for v0 in v0_sweep:
        for seed in seeds:
            print(f" -> Simulating v0 = {v0:4.1f} | Seed = {seed}")
            params = ABPParams(N=N_particles, L=L_box, v0=v0, D_R=2.0, dt=1e-3)
            traj = simulate(params, n_steps=800, sample_every=40, seed=seed)
            
            feats = compute_invariant_features(traj, params.L, max_neighbors=max_neigh)
            flat_feats = feats.reshape(feats.shape[0], -1)
            
            with torch.no_grad():
                _, latent_coords = model(flat_feats)
                
            variance = torch.var(latent_coords, dim=0).mean().item()
            ensemble_results[v0].append(variance)
            
    # Calculate Mean and Standard Deviation across seeds
    v0_arr = np.array(v0_sweep)
    mean_vars = np.array([np.mean(ensemble_results[v0]) for v0 in v0_sweep])
    std_vars = np.array([np.std(ensemble_results[v0]) for v0 in v0_sweep])
    
    print("\nPlotting Publication-Grade Curve with Error Ribbons...")
    plt.figure(figsize=(8, 5))
    plt.plot(v0_arr, mean_vars, marker='o', color='#2b5c8f', linewidth=2.5, label='Mean Latent Variance')
    plt.fill_between(v0_arr, mean_vars - std_vars, mean_vars + std_vars, color='#2b5c8f', alpha=0.25, label='±1 Std. Deviation')
    
    plt.title("Unsupervised MIPS Phase Transition Signature (Ensemble Averaged)", fontsize=12, pad=10)
    plt.xlabel("Self-Propulsion Speed ($v_0$)", fontsize=11)
    plt.ylabel("Latent Space Variance (Order Metric)", fontsize=11)
    plt.legend(frameon=True, facecolor='white')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    output_file = "abp_ensemble_transition_curve.png"
    plt.savefig(output_file, dpi=300)
    print(f"Max-spec ensemble figure successfully saved as '{output_file}'!")
    plt.show()

if __name__ == "__main__":
    run_ensemble_analysis()