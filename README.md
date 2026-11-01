# Unsupervised Discovery of Motility-Induced Phase Separation via Permutation-Invariant PointNet Autoencoders

A high-performance computational physics framework combining Langevin dynamics, permutation-invariant PointNet autoencoders, and Explainable AI (XAI) to autonomously discover and characterize Motility-Induced Phase Separation (MIPS) in active matter systems.

## Abstract
Motility-induced phase separation (MIPS) represents a hallmark non-equilibrium phenomenon in active matter systems where self-propelled particles cluster spontaneously without explicit attractive interactions. This repository implements an unsupervised machine learning pipeline that structures local coordination neighborhoods as 2D point clouds to autonomously learn latent space representations mirroring classical phase boundaries.

## Key Features
- Overdamped Langevin Dynamics Simulation of active Brownian particles (ABPs) using WCA repulsive potentials.
- Permutation-Invariant PointNet Architecture for processing local coordination shells via shared MLPs and symmetric max-pooling.
- Comprehensive Statistical Suite computing Mean Squared Displacement (MSD), Radial Distribution Functions $g(r)$, and Static Structure Factors $S(q)$.
- Explainable AI (XAI) Gradient Saliency attributions mapping neural network focus to core steric interactions.

## Project Structure
- `abp_upgrade3_msd_kinetics.png`: Transport kinetics & MSD plot
- `abp_structure_factors.png`: Radial distribution $g(r)$ and structure factor $S(q)$
- `abp_xai_saliency_attribution.jpg`: XAI gradient saliency heatmap
- `abp_structure_factors.py`: Simulation and analysis script
- `mips_pointnet_manuscript.pdf`: APS Physical Review E formatted manuscript
- `README.md`: Documentation

## Getting Started
Prerequisites:
```bash
pip install numpy matplotlib scipy torch
Running the Analysis:

Bash
python abp_structure_factors.py
License
Distributed under the MIT License. See LICENSE for details.

Author
Joel Joseph Baiju
