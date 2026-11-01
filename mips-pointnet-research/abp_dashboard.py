"""
abp_dashboard.py -- Interactive Streamlit Web Dashboard for MIPS Pipeline
========================================================================
Provides real-time interactive controls for simulation parameters, 
clustering metrics, and PointNet latent space manifold tracking.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from abp_engine import ABPParams, simulate
from abp_pointnet_autoencoder import compute_pointcloud_features, PointNetAutoencoder

st.set_page_config(page_title="MIPS Active Matter Research Dashboard", layout="wide")

st.title("🔬 Active Matter & MIPS Computational Research Dashboard")
st.markdown("Interactive suite for simulating Overdamped Langevin dynamics, computing classical cluster fractions ($P_{\text{max}}$), and monitoring PointNet latent space representations.")

st.sidebar.header("Simulation Parameters")
v0 = st.sidebar.slider("Self-Propulsion Speed ($v_0$)", min_value=1.0, max_value=50.0, value=30.0, step=1.0)
N_particles = st.sidebar.slider("Particle Count ($N$)", min_value=50, max_value=300, value=100, step=25)
L_box = st.sidebar.slider("Box Size ($L$)", min_value=20.0, max_value=60.0, value=30.0, step=5.0)
steps = st.sidebar.slider("Simulation Steps", min_value=200, max_value=1000, value=400, step=100)

if st.button("Run Simulation & Compute Analytics", type="primary"):
    with st.spinner("Executing Langevin simulation and PointNet embedding..."):
        params = ABPParams(N=int(N_particles), L=float(L_box), v0=float(v0), D_R=2.0, dt=1e-3)
        traj = simulate(params, n_steps=int(steps), sample_every=20, seed=42)
        
        final_pos = traj[-1] % L_box
        tree = cKDTree(final_pos, boxsize=L_box)
        pairs = tree.query_pairs(r=1.5)
        if len(pairs) > 0:
            row = np.array([p[0] for p in pairs])
            col = np.array([p[1] for p in pairs])
            adj = coo_matrix((np.ones(len(pairs)), (row, col)), shape=(N_particles, N_particles)).tocsr()
            _, labels = connected_components(csgraph=adj, directed=False)
            _, counts = np.unique(labels, return_counts=True)
            p_max = np.max(counts) / N_particles
        else:
            p_max = 1.0 / N_particles
            
        pc_tensor = compute_pointcloud_features(traj, L=float(L_box), max_neighbors=min(20, N_particles-1))
        model = PointNetAutoencoder(latent_dim=2, output_dim=N_particles * min(20, N_particles-1) * 2)
        model.eval()
        with torch.no_grad():
            _, latent = model(pc_tensor)
            lat_var = torch.var(latent, dim=0).mean().item()
            
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Largest Cluster Fraction ($P_{max}$)", value=f"{p_max:.4f}")
    with col2:
        st.metric(label="PointNet Latent Variance", value=f"{lat_var:.6e}")
        
    st.subheader("Spatial Configuration & Latent Space Flow")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(final_pos[:, 0], final_pos[:, 1], c='#1f77b4', s=30, edgecolor='k', alpha=0.8)
    ax1.set_xlim(0, L_box)
    ax1.set_ylim(0, L_box)
    ax1.set_title(f"Final Particle Configuration ($v_0$ = {v0})")
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    lat_np = latent.numpy()
    ax2.plot(lat_np[:, 0], lat_np[:, 1], color='#d62728', marker='o', markersize=3, alpha=0.7)
    ax2.set_title("Latent Manifold Trajectory ($z_1$ vs $z_2$)")
    ax2.set_xlabel("Latent $z_1$")
    ax2.set_ylabel("Latent $z_2$")
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    st.pyplot(fig)
else:
    st.info("Adjust parameters in the sidebar and click the button above to run the active matter suite.")