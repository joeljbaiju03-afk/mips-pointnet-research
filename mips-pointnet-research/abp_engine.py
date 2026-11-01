"""
abp_engine.py -- Vectorized 2D Active Brownian Particle (ABP) simulator
=======================================================================

Overdamped Langevin dynamics in a periodic square box [0, L)^2:

    dr_i     = [ v0 * e(theta_i) + mu * F_i ] dt + sqrt(2 D_T dt) * xi_i
    dtheta_i = sqrt(2 D_R dt) * eta_i

    e(theta) = (cos theta, sin theta),   xi ~ N(0, I_2),   eta ~ N(0, 1)
    F_i      = sum_{j != i} F_WCA(r_i - r_j)   (minimum image convention)

WCA (shifted, purely repulsive Lennard-Jones, cut at r_c = 2^(1/6) sigma):

    U(r) = 4 eps [ (sigma/r)^12 - (sigma/r)^6 ] + eps ,   r < r_c ; 0 otherwise
    F(r) = 24 eps / r^2 * [ 2 (sigma/r)^12 - (sigma/r)^6 ] * r_vec

Two pair-search back-ends (identical physics):
    "dense" : full NxN minimum-image distance matrix (pure NumPy, O(N^2) memory).
    "tree"  : periodic cKDTree neighbour search (scipy), O(N) memory; best for N > ~256.
    "auto"  : dense for N <= 256, tree otherwise (falls back to dense w/o scipy).

Output: float array of shape (n_frames, N, 3) with columns [x, y, theta].

Usage
-----
    from abp_engine import ABPParams, simulate
    p = ABPParams(N=400, L=40.0, v0=40.0, D_R=3.0, D_T=1.0, dt=1e-5)
    traj = simulate(p, n_steps=200_000, sample_every=1000, seed=0)
    print(traj.shape)   # (200, 400, 3)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

TWO_PI = 2.0 * np.pi
R_CUT_FACTOR = 2.0 ** (1.0 / 6.0)


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass
class ABPParams:
    N: int = 400                # number of particles
    L: float = 40.0             # box side length
    v0: float = 20.0            # self-propulsion speed
    D_R: float = 3.0            # rotational diffusion coefficient
    D_T: float = 1.0            # translational diffusion coefficient (0 => noise-free)
    dt: float = 1e-4            # integration timestep
    sigma: float = 1.0          # WCA length scale (particle diameter)
    epsilon: float = 1.0        # WCA energy scale
    mobility: float = 1.0       # mu in  dr = mu F dt  (set mu = D_T/kT for kT-consistent units)
    f_max: float | None = None  # force-magnitude cap; None -> 0.1*sigma/(mobility*dt) (max 0.1 sigma step)
    dtype: str = "float64"      # "float32" halves memory / speeds dense mode

    def validate(self) -> None:
        if self.f_max is None:
            self.f_max = 0.1 * self.sigma / (self.mobility * self.dt)
        if self.N < 1 or self.L <= 0 or self.dt <= 0:
            raise ValueError("N, L, dt must be positive.")
        if min(self.D_R, self.D_T, self.v0) < 0:
            raise ValueError("v0, D_R, D_T must be non-negative.")
        if self.L < 2 * R_CUT_FACTOR * self.sigma:
            raise ValueError("L must exceed twice the WCA cutoff for minimum image to be valid.")
        # Rough stability check: max single-step displacement from contact force
        max_step = self.mobility * self.f_max * self.dt
        if max_step > 0.1 * self.sigma:
            print(f"[warning] mobility*f_max*dt = {max_step:.3g} > 0.1*sigma; "
                  f"consider smaller dt or f_max.")

    @property
    def packing_fraction(self) -> float:
        return self.N * np.pi * (self.sigma / 2) ** 2 / self.L ** 2


# --------------------------------------------------------------------------- #
# Initial condition: jittered square lattice (guarantees no overlap if phi < ~0.78)
# --------------------------------------------------------------------------- #
def init_state(p: ABPParams, rng: np.random.Generator):
    n_side = int(np.ceil(np.sqrt(p.N)))
    spacing = p.L / n_side
    if spacing < p.sigma * 0.95:
        raise ValueError(f"Box too dense for lattice init (spacing={spacing:.3f} < sigma). "
                         f"Increase L or reduce N.")
    gx, gy = np.meshgrid(np.arange(n_side), np.arange(n_side), indexing="ij")
    sites = (np.stack([gx.ravel(), gy.ravel()], axis=1) + 0.5) * spacing
    idx = rng.permutation(sites.shape[0])[: p.N]
    jitter = max(0.0, 0.5 * (spacing - p.sigma)) * 0.5
    pos = sites[idx] + rng.uniform(-jitter, jitter, size=(p.N, 2))
    theta = rng.uniform(0.0, TWO_PI, size=p.N)
    return pos.astype(p.dtype) % p.L, theta.astype(p.dtype)


# --------------------------------------------------------------------------- #
# WCA forces
# --------------------------------------------------------------------------- #
def _wca_pair_factor(r2, sigma2, eps, f_max):
    """Return scalar s such that F_vec = s * r_vec, with magnitude capped at f_max."""
    r2 = np.maximum(r2, 1e-12)
    sr2 = sigma2 / r2
    sr6 = sr2 ** 3
    s = 24.0 * eps * (2.0 * sr6 * sr6 - sr6) / r2
    # cap |F| = s*r  at f_max
    r = np.sqrt(r2)
    return np.minimum(s, f_max / r)


def forces_dense(pos, p: ABPParams):
    """Full NxN minimum-image WCA force (fully vectorized)."""
    d = pos[:, None, :] - pos[None, :, :]                 # (N,N,2)
    d -= p.L * np.round(d / p.L)                          # minimum image
    r2 = np.einsum("ijk,ijk->ij", d, d)                   # (N,N)
    rc2 = (R_CUT_FACTOR * p.sigma) ** 2
    mask = r2 < rc2
    np.fill_diagonal(mask, False)
    s = np.zeros_like(r2)
    s[mask] = _wca_pair_factor(r2[mask], p.sigma ** 2, p.epsilon, p.f_max)
    return np.einsum("ij,ijk->ik", s, d)                  # (N,2)


def forces_tree(pos, p: ABPParams):
    """Periodic KD-tree neighbour search + vectorized pair forces."""
    tree = cKDTree(pos, boxsize=p.L)
    pairs = tree.query_pairs(R_CUT_FACTOR * p.sigma, output_type="ndarray")
    F = np.zeros((p.N, 2), dtype=pos.dtype)
    if pairs.size == 0:
        return F
    i, j = pairs[:, 0], pairs[:, 1]
    d = pos[i] - pos[j]
    d -= p.L * np.round(d / p.L)
    r2 = np.einsum("ij,ij->i", d, d)
    f = _wca_pair_factor(r2, p.sigma ** 2, p.epsilon, p.f_max)[:, None] * d
    # Newton's 3rd law accumulation
    for k in range(2):
        F[:, k] = (np.bincount(i, f[:, k], p.N) - np.bincount(j, f[:, k], p.N))
    return F


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
def simulate(
    p: ABPParams,
    n_steps: int,
    sample_every: int = 100,
    seed: int | None = None,
    method: str = "auto",
    wrap_positions: bool = True,
    wrap_theta: bool = False,
    init_pos: np.ndarray | None = None,
    init_theta: np.ndarray | None = None,
    equilibration_steps: int = 0,
    progress: bool = False,
) -> np.ndarray:
    """
    Run the simulation and return traj of shape (n_frames, N, 3) = [x, y, theta].

    Parameters
    ----------
    n_steps            : production steps (after equilibration).
    sample_every       : record a frame every `sample_every` steps (frame 0 = initial state).
                         n_frames = n_steps // sample_every + 1.
    wrap_positions     : True -> x,y in [0, L).  False -> unwrapped (continuous) coordinates,
                         which is what you want for MSD / displacement-based ML features.
    wrap_theta         : True -> theta in [-pi, pi).  False -> unwrapped angle (default),
                         so rotational diffusion is directly recoverable from the series.
    init_pos/theta     : optional user-supplied initial condition.
    equilibration_steps: steps run (and discarded) before recording starts.
    """
    p.validate()
    rng = np.random.default_rng(seed)
    dt_type = np.dtype(p.dtype)

    if init_pos is None:
        pos, theta = init_state(p, rng)
    else:
        pos = np.asarray(init_pos, dtype=dt_type) % p.L
        theta = (np.asarray(init_theta, dtype=dt_type) if init_theta is not None
                 else rng.uniform(0, TWO_PI, p.N).astype(dt_type))

    if method == "auto":
        method = "tree" if (p.N > 256 and _HAS_SCIPY) else "dense"
    if method == "tree" and not _HAS_SCIPY:
        raise ImportError("method='tree' requires scipy.")
    force_fn = forces_tree if method == "tree" else forces_dense

    # Pre-compute constants
    dt = dt_type.type(p.dt)
    mu_dt = dt_type.type(p.mobility * p.dt)
    v0_dt = dt_type.type(p.v0 * p.dt)
    noise_T = dt_type.type(np.sqrt(2.0 * p.D_T * p.dt))
    noise_R = dt_type.type(np.sqrt(2.0 * p.D_R * p.dt))

    unwrapped = pos.copy()          # accumulates true displacement (no PBC jumps)
    n_frames = n_steps // sample_every + 1
    traj = np.empty((n_frames, p.N, 3), dtype=dt_type)

    def record(k):
        xy = (pos if wrap_positions else unwrapped)
        traj[k, :, :2] = xy
        th = theta
        if wrap_theta:
            th = (theta + np.pi) % TWO_PI - np.pi
        traj[k, :, 2] = th

    def step():
        nonlocal pos, theta, unwrapped
        F = force_fn(pos, p)
        e = np.empty((p.N, 2), dtype=dt_type)
        e[:, 0] = np.cos(theta)
        e[:, 1] = np.sin(theta)
        dr = v0_dt * e + mu_dt * F
        if p.D_T > 0:
            dr += noise_T * rng.standard_normal((p.N, 2)).astype(dt_type)
        pos = (pos + dr) % p.L
        unwrapped += dr
        if p.D_R > 0:
            theta = theta + noise_R * rng.standard_normal(p.N).astype(dt_type)

    for _ in range(equilibration_steps):
        step()
    unwrapped = pos.copy()          # reset reference after equilibration

    record(0)
    k = 1
    for n in range(1, n_steps + 1):
        step()
        if n % sample_every == 0:
            record(k)
            k += 1
            if progress and k % max(1, n_frames // 10) == 0:
                print(f"  {100 * n / n_steps:5.1f}%")
    return traj


# --------------------------------------------------------------------------- #
# ML-oriented helpers
# --------------------------------------------------------------------------- #
def to_ml_features(traj: np.ndarray, L: float | None = None) -> np.ndarray:
    """
    Convert (T, N, 3)[x,y,theta] -> (T, N, 4)[x/L, y/L, cos(theta), sin(theta)].
    Periodic-safe, bounded, and free of angle discontinuities.
    Pass L to normalise positions; if None, positions are left as-is.
    """
    out = np.empty(traj.shape[:2] + (4,), dtype=traj.dtype)
    out[..., 0] = traj[..., 0] / L if L else traj[..., 0]
    out[..., 1] = traj[..., 1] / L if L else traj[..., 1]
    out[..., 2] = np.cos(traj[..., 2])
    out[..., 3] = np.sin(traj[..., 2])
    return out


def min_pair_distance(pos_xy: np.ndarray, L: float) -> float:
    """Diagnostic: smallest minimum-image pair separation in one frame."""
    d = pos_xy[:, None] - pos_xy[None]
    d -= L * np.round(d / L)
    r = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(r, np.inf)
    return float(r.min())


# --------------------------------------------------------------------------- #
# Demo / self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time

    p = ABPParams(N=400, L=40.0, v0=20.0, D_R=3.0, D_T=0.1, dt=1e-4)
    print(f"packing fraction phi = {p.packing_fraction:.3f}")

    t0 = time.perf_counter()
    traj = simulate(p, n_steps=20_000, sample_every=200, seed=1,
                    equilibration_steps=2_000)
    print(f"traj shape {traj.shape}, dtype {traj.dtype}, {time.perf_counter() - t0:.1f}s")

    # Sanity 1: no catastrophic overlaps
    print("min pair distance (last frame):", round(min_pair_distance(traj[-1, :, :2], p.L), 3),
          "(WCA allows slight overlap << sigma at high v0)")

    # Sanity 2: dense and tree back-ends agree
    rng = np.random.default_rng(0)
    pos0, _ = init_state(p, rng)
    pos0 += rng.normal(0, 0.15, pos0.shape); pos0 %= p.L
    if _HAS_SCIPY:
        diff = np.abs(forces_dense(pos0, p) - forces_tree(pos0, p)).max()
        print("max |F_dense - F_tree| =", diff)

    # Sanity 3: free-particle rotational diffusion  <cos(dtheta)> = exp(-D_R t)
    pf = ABPParams(N=2000, L=500.0, v0=0.0, D_R=1.0, D_T=0.0, dt=1e-3)
    tf = simulate(pf, n_steps=1000, sample_every=1000, seed=2)
    est = np.cos(tf[-1, :, 2] - tf[0, :, 2]).mean()
    print(f"<cos dtheta>(t=1) = {est:.3f}   theory = {np.exp(-1.0):.3f}")

    ml = to_ml_features(traj, p.L)
    print("ML feature tensor:", ml.shape)
