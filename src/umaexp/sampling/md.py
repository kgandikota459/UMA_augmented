"""Run equilibrium sampling with any Potential and produce a SampleCache."""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from ..potential.base import Potential
from ..systems.system import System
from .cache import SampleCache, SamplingProtocol, ThermoState
from .diagnostics import detect_equilibration, statistical_inefficiency
from .integrators import ForceEvaluator, LangevinBAOAB, MonteCarloBarostat, instantaneous_temperature

log = logging.getLogger(__name__)

_DTYPES = {"float64": torch.float64, "float32": torch.float32}


def run_sampling(
    potential: Potential,
    system: System,
    state: ThermoState,
    protocol: SamplingProtocol,
    positions: torch.Tensor | None = None,
    cell: torch.Tensor | None = None,
    replica: int = 0,
    progress_every: int = 0,
) -> SampleCache:
    """Langevin (NVT) or Langevin + MC barostat (NPT) sampling.

    Frames are stored every `protocol.stride` production steps together with
    the sampling potential's energies. Barostat step size adapts only during
    equilibration.
    """
    dtype = _DTYPES[protocol.dtype]
    pos0 = system.positions_tensor() if positions is None else positions.detach().to(torch.float64)
    cell0 = None
    if system.is_periodic:
        cell0 = system.cell_tensor() if cell is None else cell.detach().to(torch.float64)
    if state.ensemble == "NPT" and cell0 is None:
        raise ValueError("NPT sampling needs a periodic system")
    evaluator = ForceEvaluator(potential, system, device=protocol.device, dtype=dtype)
    seed = protocol.seed + 1000 * replica
    integ = LangevinBAOAB(evaluator, system.mass_tensor(), state.temperature_k, protocol.dt_fs,
                          protocol.friction_per_ps, seed)
    baro = None
    if state.ensemble == "NPT":
        baro = MonteCarloBarostat(evaluator, system, state.temperature_k, state.pressure_bar, seed,
                                  max_dlnv=protocol.barostat_max_dlnv)
    s = integ.init_state(pos0, cell0)
    t0 = time.time()
    temps = []
    for step in range(protocol.n_equilibration):
        s = integ.step(s)
        if baro is not None and (step + 1) % protocol.barostat_every == 0:
            s = baro.attempt(s)
            if (step + 1) % (10 * protocol.barostat_every) == 0:
                baro.adapt()
        if not np.isfinite(s.energy):
            raise FloatingPointError(f"non-finite energy during equilibration at step {step}")
    if baro is not None:
        baro.n_accepted = baro.n_trials = 0
    frames, cells, energies = [], [], []
    for step in range(protocol.n_production):
        s = integ.step(s)
        if baro is not None and (step + 1) % protocol.barostat_every == 0:
            s = baro.attempt(s)
        if not np.isfinite(s.energy):
            raise FloatingPointError(f"non-finite energy during production at step {step}")
        if (step + 1) % protocol.stride == 0:
            frames.append(s.positions.numpy().copy())
            energies.append(s.energy)
            # Langevin noise thermalizes all 3N momenta (no conserved COM momentum)
            temps.append(instantaneous_temperature(integ.m, s.velocities, n_constraints=0))
            if s.cell is not None:
                cells.append(s.cell.numpy().copy())
        if progress_every and (step + 1) % progress_every == 0:
            log.info("step %d / %d, E=%.4f eV", step + 1, protocol.n_production, s.energy)
    u = np.asarray(energies)
    diag = {
        "wall_s": time.time() - t0,
        "force_calls": evaluator.n_calls,
        "mean_temperature_k": float(np.mean(temps)) if temps else float("nan"),
        "g_energy": statistical_inefficiency(u),
    }
    t_eq, g, neff = detect_equilibration(u)
    diag.update(equilibration_frame=int(t_eq), g_after_equilibration=float(g), n_eff_energy=float(neff))
    if cells:
        vols = np.abs(np.linalg.det(np.asarray(cells)))
        diag["g_volume"] = statistical_inefficiency(vols)
        diag["barostat_acceptance"] = baro.acceptance if baro is not None else None
        diag["mean_volume_A3"] = float(vols.mean())
    return SampleCache(
        system_id=system.system_id,
        system_hash=system.atom_order_hash(),
        state=state,
        protocol=protocol,
        sampling_fingerprint=potential.parameter_fingerprint(),
        positions=np.asarray(frames),
        cells=np.asarray(cells) if cells else None,
        u_sampling=u,
        replica=replica,
        diagnostics=diag,
    )
