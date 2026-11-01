"""
abp_structure_factors.py -- Liquid-State Physics: g(r) & S(q)
=============================================================
Calculates the Radial Distribution Function g(r) and the reciprocal-space 
Static Structure Factor S(q) to benchmark MIPS phase transitions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from abp_engine import ABPParams, simulate

def compute_radial_distribution(traj, L, r_max=10.0, dr=0.1):
    """
    Computes time-averaged radial distribution function g(r).
    """
    n_frames, N, _ = traj.shape
    nbins = int(r_max / dr)
    bins = np.linspace(0, r_max, nbins + 1)
    r_centers = 0.5 * (bins[1:] + bins[:-1])
    
    rho = N / (L**2)
    hist_total = np.zeros(nbins)
    
    for f in range(n_frames):
        pos = traj[f] % L
        tree = cKDTree(pos, boxsize=L)
        
        # Query all pairs up to r_max
        pairs = tree.query_pairs(r=r_max, output_type='ndarray')
        if len(pairs) == 0:
            continue
            
        dx = pos[pairs[:, 0], 0] - pos[pairs[:, 1], 0]
        dy = pos[pairs[:, 0], 1] - pos[pairs[:, 1], 1]
        dx = dx - L * np.round(dx / L)
        dy = dy - L * np.round(dy / L)
        distances = np.sqrt(dx**2 + dy**2)
        
        counts, _ = np.histogram(distances, bins=bins)
        hist_total += counts * 2 # Each pair accounts for 2 interactions
        
    # Normalization by shell area and ideal density
    shell_areas = np.pi * (bins[1:]**2 - bins[:-1]**2)
    g_r = hist_total / (n_frames * N * rho * shell_areas)
    
    return r_centers, g_r

def compute_structure_factor(traj, L, q_max=12.0, dq=0.3):
    """
    Computes isotropically averaged Static Structure Factor S(q).
    """
    n_frames, N, _ = traj.shape
    q_vals = np.arange(2 * np.pi / L, q_max, dq)
    S_q = []
    
    for q in q_vals:
        S_q_frame = []
        # Sample discrete wavevectors on a ring |q_vec| = q
        n_angles = 16
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        qx = q * np.cos(angles)
        qy = q * np.sin(angles)
        
        for f in range(n_frames):
            pos = traj[f]
            # Spatial Fourier sum: sum_j exp(-i * q * r_j)
            qr = np.outer(pos[:, 0], qx) + np.outer(pos[:, 1], qy) # [N, n_angles]
            density_ft = np.sum(np.exp(-1j * qr), axis=0) # [n_angles]
            sq_val = np.mean(np.abs(density_ft)**2) / N
            S_q_frame.append(sq_val)
            
        S_q.append(np.mean(S_q_frame))
        
    return q_vals, np.array(S_q)

def run_liquid_state_analysis():
    print("=== Initializing Liquid-State Physics Suite: g(r) & S(q) ===")
    
    N_particles = 100
    L_box = 30.0
    
    regimes = {
        'Gas Phase ($v_0 = 2.0$)': 2.0,
        'MIPS Phase ($v_0 = 40.0$)': 40.0
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for label, v0 in regimes.items():
        print(f" -> Simulating trajectory for {label}...")
        params = ABPParams(N=N_particles, L=L_box, v0=v0, D_R=2.0, dt=1e-3)
        traj = simulate(params, n_steps=600, sample_every=20, seed=42)
        
        # 1. Radial Distribution Function g(r)
        r, g_r = compute_radial_distribution(traj, L=L_box)
        ax1.plot(r, g_r, linewidth=2.5, label=label)
        
        # 2. Static Structure Factor S(q)
        q, S_q = compute_structure_factor(traj, L=L_box)
        ax2.plot(q, S_q, linewidth=2.5, label=label)
        
    ax1.set_title("Radial Distribution Function $g(r)$", fontsize=11)
    ax1.set_xlabel("Distance $r$", fontsize=10)
    ax1.set_ylabel("$g(r)$", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.axvline(x=1.122, color='gray', linestyle=':', label='WCA $\\sigma$')
    ax1.legend(frameon=True, facecolor='white')
    
    ax2.set_title("Static Structure Factor $S(q)$", fontsize=11)
    ax2.set_xlabel("Wavevector Magnitude $q$", fontsize=10)
    ax2.set_ylabel("$S(q)$", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(frameon=True, facecolor='white')
    
    plt.suptitle("Liquid-State Condensed Matter Benchmarking", fontsize=13, y=1.02)
    plt.tight_layout()
    
    output_filename = "abp_liquid_state_statistics.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nLiquid-state statistics plot saved as '{output_filename}'!")
    plt.show()

if __name__ == "__main__":
    run_liquid_state_analysis()