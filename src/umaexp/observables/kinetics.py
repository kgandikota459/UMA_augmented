r"""Dynamical observables: diffusion, VACF/VDOS, and a research gradient estimator.

Equilibrium frame reweighting is NOT a kinetic gradient estimator: it
reweights configurations, not paths. Transport coefficients and rates need
physical dynamics (NVE or a validated weak thermostat), time-correlation
functions converged over the relevant timescale, and finite-size
corrections.

Implemented for evaluation (no training gradient):
  self-diffusion from the Einstein relation with the Yeh-Hummer correction
      D_inf = D_PBC + k_B T xi / (6 pi eta L),  xi = 2.837297
  velocity autocorrelation and its spectrum (classical VDOS)

Research option (`unrolled_nve_gradient`): back-propagation through a short
velocity-Verlet window with create_graph forces. Memory grows linearly with
the window and gradients can explode for chaotic dynamics; use only with
short windows, variance checks, and comparisons against finite differences.
Path-reweighting estimators for Langevin dynamics (Girsanov) are the more
principled route and are listed as future work.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .. import units
from ..potential.base import Potential
from ..systems.system import System

YEH_HUMMER_XI = 2.837297


def msd_diffusion(positions_fs: np.ndarray, dt_fs: float, fit_range: tuple[float, float] = (0.2, 0.8),
                  molecule_index=None, masses=None) -> dict:
    """Self-diffusion (A^2/fs and cm^2/s) from unwrapped positions (F, N, 3).

    With molecule_index, uses molecular centres of mass.
    """
    x = np.asarray(positions_fs)
    if molecule_index is not None:
        mol = np.asarray(molecule_index)
        m = np.asarray(masses)
        nm = mol.max() + 1
        com = np.zeros((x.shape[0], nm, 3))
        for k in range(nm):
            sel = mol == k
            com[:, k] = (x[:, sel] * m[sel, None]).sum(1) / m[sel].sum()
        x = com
    F = x.shape[0]
    lags = np.arange(1, F // 2)
    msd = np.array([((x[lag:] - x[:-lag]) ** 2).sum(-1).mean() for lag in lags])
    t = lags * dt_fs
    lo, hi = int(fit_range[0] * len(t)), int(fit_range[1] * len(t))
    slope = np.polyfit(t[lo:hi], msd[lo:hi], 1)[0]
    D = slope / 6.0
    return {"D_A2_per_fs": D, "D_cm2_per_s": D * 1e-16 / 1e-15, "t_fs": t, "msd": msd}


def yeh_hummer(D_pbc_cm2_s: float, temperature_k: float, viscosity_pa_s: float, box_length_a: float) -> float:
    corr = units.BOLTZMANN_J_K * temperature_k * YEH_HUMMER_XI / (6 * math.pi * viscosity_pa_s * box_length_a * 1e-10)
    return D_pbc_cm2_s + corr * 1e4  # m^2/s -> cm^2/s


def vacf_spectrum(velocities: np.ndarray, masses: np.ndarray, dt_fs: float):
    """Mass-weighted VACF and its cosine transform (classical VDOS), cm^-1 axis."""
    v = np.asarray(velocities)
    m = np.asarray(masses)[None, :, None]
    F = v.shape[0]
    n_lag = F // 2
    c = np.array([(m * v[: F - t] * v[t:]).sum() / (F - t) for t in range(n_lag)])
    c = c / c[0]
    window = np.hanning(2 * n_lag)[n_lag:]
    spec = np.abs(np.fft.rfft(c * window, n=4 * n_lag))
    freq_cm = np.fft.rfftfreq(4 * n_lag, d=dt_fs * 1e-15) / (units.SPEED_OF_LIGHT_M_S * 100.0)
    return {"vacf": c, "wavenumber": freq_cm, "vdos": spec}


def unrolled_nve_gradient(potential: Potential, system: System, x0: torch.Tensor, v0: torch.Tensor,
                          n_steps: int, dt_fs: float, observable) -> torch.Tensor:
    """Differentiable NVE window: returns observable(trajectory) with graph to w.

    observable: callable on the list of positions [(N,3)] -> scalar tensor.
    RESEARCH ESTIMATOR: see module docstring.
    """
    m = system.mass_tensor()[:, None]
    dt = dt_fs * units.FS
    x, v = x0.detach().clone(), v0.detach().clone()
    traj = [x]
    _, f = potential.energy_and_forces(system, x.requires_grad_(True), create_graph=True)
    for _ in range(n_steps):
        v = v + 0.5 * dt * f / m
        x = x + dt * v
        _, f = potential.energy_and_forces(system, x, create_graph=True)
        v = v + 0.5 * dt * f / m
        traj.append(x)
    return observable(traj)
