"""
abp_upgrade3_msd.py -- Mean Squared Displacement & Transport Dynamics
======================================================================
Computes unwrapped MSD <\Delta r^2(\tau)> across gas and MIPS regimes 
to capture active ballistic motion versus dense caging plateaus.
"""

import numpy as np
import matplotlib.pyplot as plt
from abp_engine import ABPParams, simulate

def compute_msd_unwrapped(traj, L, dt_sample):
    """
    Computes the time-averaged MSD for each particle using unwrapped coordinates 
    to correctly handle periodic boundary conditions.
    traj shape: [n_frames, N, 2]
    """
    n_frames, N, _ = traj.shape
    
    # 1. Unwrap coordinates across frames using minimum image convention
    unwrapped = np.zeros_like(traj)
    unwrapped[0] = traj[0]
    
    for f in range(1, n_frames):
        dr = traj[f] - traj[f-1]
        dr = dr - L * np.round(dr / L)
        unwrapped[f] = unwrapped[f-1] + dr
        
    # 2. Compute MSD for various time lags tau
    max_lag = n_frames // 2
    tau_values = np.arange(1, max_lag)
    msd_means = []
    
    for tau in tau_values:
        # Displacements over lag tau: unwrapped[t + tau] - unwrapped[t]
        displacements = unwrapped[tau:] - unwrapped[:-tau] # shape: [n_frames - tau, N, 2]
        squared_disps = np.sum(displacements**2, axis=-1) # shape: [n_frames - tau, N]
        mean_sq_disp = np.mean(squared_disps) # average over all time origins and particles
        msd_means.append(mean_sq_disp)
        
    time_lags = tau_values * dt_sample
    return time_lags, np.array(msd_means)

def run_msd_analysis():
    print("=== Initializing Upgrade 3: MSD & Transport Dynamics Pipeline ===")
    
    N_particles = 100
    L_box = 30.0
    dt = 1e-3
    sample_every = 20
    dt_sample = dt * sample_every
    
    v0_regimes = {
        'Gas Phase ($v_0 = 2.0$)': 2.0,
        'Transition ($v_0 = 15.0$)': 15.0,
        'MIPS Caged Phase ($v_0 = 40.0$)': 40.0
    }
    
    plt.figure(figsize=(9, 5.5))
    
    for label, v0 in v0_regimes.items():
        print(f" -> Simulating trajectory for {label}...")
        params = ABPParams(N=N_particles, L=L_box, v0=v0, D_R=2.0, dt=dt)
        traj = simulate(params, n_steps=1200, sample_every=sample_every, seed=42)
        
        time_lags, msd = compute_msd_unwrapped(traj, L=L_box, dt_sample=dt_sample)
        
        # Log-log plot
        plt.loglog(time_lags, msd, linewidth=2.5, label=label)
        
    plt.title("Active Transport Dynamics: Mean Squared Displacement (MSD)", fontsize=12, pad=12)
    plt.xlabel("Time Lag $\\tau$", fontsize=11)
    plt.ylabel("Mean Squared Displacement $\\langle \\Delta r^2(\\tau) \\rangle$", fontsize=11)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(frameon=True, facecolor='white', loc='upper left')
    plt.tight_layout()
    
    output_filename = "abp_upgrade3_msd_kinetics.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nMSD transport kinetics plot successfully saved as '{output_filename}'!")
    plt.show()

if __name__ == "__main__":
    run_msd_analysis()