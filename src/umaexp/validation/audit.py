"""Stage-0 capability audit for a UMA checkpoint (or the random architecture).

Gate: the model loads with parity to FAIRChem's own predictor under the same
numerical settings, and every derivative order used by the objectives works
on small systems (coordinates, cell, parameters, mixed third order).
"""

from __future__ import annotations

import copy
import json
import os
import platform
import time

import numpy as np
import torch
from ase.build import bulk, molecule

from ..potential import UMAPotential, select_parameters, write_synthetic_checkpoint
from ..potential.derivatives import hessian
from ..potential.parameters import PRESETS, inventory
from ..systems.system import System
from .gradcheck import check_cell_derivative, check_forces, check_hessian, check_parameter_gradient

PARITY_SET = [("H2O", 0, 1), ("CH4", 0, 1), ("NH3", 0, 1), ("CH3CN", 0, 1), ("CH3", 0, 2), ("NH4", 1, 1)]


def _mol(name, q, s):
    if name == "NH4":
        a = molecule("NH3")
        a += molecule("H")
        a.positions[-1] = a.positions[0] + np.array([0.0, 0.0, -1.02])
    else:
        a = molecule(name)
    a.info.update(charge=q, spin=s)
    return System.from_ase(a, system_id=f"{name}_q{q}_s{s}", charge=q, spin=s), torch.tensor(a.get_positions())


def environment() -> dict:
    import ase
    import fairchem.core

    return {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "fairchem_core": fairchem.core.__version__, "ase": ase.__version__, "numpy": np.__version__,
            "mps_available": torch.backends.mps.is_available(), "cuda_available": torch.cuda.is_available(),
            "threads": torch.get_num_threads()}


def parity_against_fairchem(pot: UMAPotential, checkpoint_path: str) -> dict:
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    out = {}
    same = InferenceSettings(merge_mole=False, compile=False, activation_checkpointing=False,
                             base_precision_dtype=pot.dtype, execution_mode="general")
    for label, settings in [("same_settings", same), ("default_fast_path", "default")]:
        calc = FAIRChemCalculator(load_predict_unit(checkpoint_path, inference_settings=settings, device="cpu"),
                                  task_name=pot.task)
        de, df = 0.0, 0.0
        for name, q, s in PARITY_SET:
            sys, pos = _mol(name, q, s)
            a = sys.to_ase()
            a.calc = calc
            e, f = pot.energy_and_forces(sys, pos)
            de = max(de, abs(float(e) - a.get_potential_energy()))
            df = max(df, float(np.abs(f.numpy() - a.get_forces()).max()))
        out[label] = {"max_abs_energy_diff_eV": de, "max_abs_force_diff_eV_A": df}
    return out


def device_benchmark(pot: UMAPotential, sizes=(3, 20, 64), reps: int = 3) -> list[dict]:
    from ase import Atoms

    rows = []
    rng = np.random.default_rng(0)
    for nw in sizes:
        side = int(np.ceil(nw ** (1 / 3)))
        atoms = Atoms()
        k = 0
        for i in range(side):
            for j in range(side):
                for l in range(side):
                    if k < nw:
                        w = molecule("H2O")
                        w.rotate(rng.uniform(0, 360), rng.normal(size=3))
                        w.translate(np.array([i, j, l]) * 3.1)
                        atoms += w
                        k += 1
        sys = System.from_ase(atoms, system_id=f"water{nw}")
        pos = torch.tensor(atoms.get_positions())
        for dev, dt in [("cpu", torch.float64), ("cpu", torch.float32), ("mps", torch.float32)]:
            if dev == "mps" and not torch.backends.mps.is_available():
                continue
            p = pot if (dev == "cpu" and dt == pot.dtype) else pot.to_device(dev, dt)
            x = pos.to(dt).to(dev)
            p.energy_and_forces(sys, x)
            if dev == "mps":
                torch.mps.synchronize()
            t = time.perf_counter()
            for _ in range(reps):
                p.energy_and_forces(sys, x)
            if dev == "mps":
                torch.mps.synchronize()
            rows.append({"atoms": len(atoms), "device": dev, "dtype": str(dt).replace("torch.", ""),
                         "ms_energy_forces": 1e3 * (time.perf_counter() - t) / reps})
            if p is not pot:
                del p
    return rows


def run_audit(checkpoint: str | None = None, task: str = "omol", out_dir: str = "runs/audit",
              benchmark: bool = True, workdir: str | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    report: dict = {"environment": environment(), "random_architecture": checkpoint is None}
    t0 = time.time()
    if checkpoint is None:
        refs = torch.zeros(101, dtype=torch.float64)
        base = UMAPotential.random_architecture(task=task, element_references=refs)
        ckpt_path = write_synthetic_checkpoint(base, os.path.join(workdir or out_dir, "synthetic_uma.pt"))
        report["note"] = ("RANDOM WEIGHTS: checks the architecture, loading path and derivative machinery; "
                          "no physical meaning. Re-run with the gated checkpoint for Stage-0 sign-off.")
    else:
        ckpt_path = checkpoint
    pot = UMAPotential.from_checkpoint(ckpt_path, task=task, dtype=torch.float64)
    ckpt_path = pot._checkpoint_path  # registry names (e.g. "uma-s-1p1") resolve to the cached file
    report["load_seconds"] = time.time() - t0
    ck = torch.load(pot._checkpoint_path, map_location="cpu", weights_only=False)
    bb = ck.model_config.get("backbone", {})
    report["checkpoint"] = {
        "path": str(checkpoint or "synthetic(random)"), "sha256": pot.provenance.checkpoint_sha256,
        "model_id": ck.model_config.get("model_id"),
        "backbone": {k: bb.get(k) for k in ("model", "num_experts", "sphere_channels", "lmax", "mmax", "num_layers",
                                            "cutoff", "max_neighbors", "direct_forces", "regress_forces",
                                            "regress_stress", "chg_spin_emb_type", "dataset_list",
                                            "use_composition_embedding", "model_version")},
        "tasks": [t.get("name") for t in ck.tasks_config],
        "energy_convention": {"task": pot.convention.task, "mean": pot.convention.mean,
                              "rmsd": pot.convention.rmsd,
                              "element_references": pot.convention.element_references is not None,
                              "source": pot.convention.source},
    }
    inv = inventory(pot.model)
    report["parameters"] = {"total": int(sum(g["numel"] for g in inv)), "groups": inv}
    presets = {}
    for name in PRESETS:
        rep = select_parameters(copy.deepcopy(pot.model) if name == "full" else pot.model, preset=name)
        presets[name] = {"tensors": rep.n_tensors, "params": rep.n_params}
    report["presets"] = presets
    report["parity"] = parity_against_fairchem(pot, ckpt_path)

    sys, pos = _mol("H2O", 0, 1)
    checks = {"forces_fd": str(check_forces(pot, sys, pos)), "hessian_fd": str(check_hessian(pot, sys, pos))}
    select_parameters(pot.model, preset="readout_norm")
    params = pot.trainable_parameters()

    def force_loss():
        e, f = pot.energy_and_forces(sys, pos, create_graph=True)
        return e + (f**2).sum()

    checks["param_grad_energy_force"] = str(check_parameter_gradient(force_loss, params, n_probe=3, h=1e-6))

    def hess_loss():
        return torch.linalg.eigvalsh(hessian(pot, sys, pos, create_graph=True)).pow(2).sum()

    checks["param_grad_third_order"] = str(check_parameter_gradient(hess_loss, params, n_probe=2, h=1e-6, rtol=1e-3))
    if task == "omat" or checkpoint is None:
        # cell derivative on a periodic crystal using a periodic task
        si = bulk("Si", "diamond", a=5.43)
        ptask = "omat"
        ppot = UMAPotential.from_checkpoint(ckpt_path, task=ptask, dtype=torch.float64)
        psys = System.from_ase(si, system_id="Si_diamond", task=ptask, charge=0, spin=0)
        checks["cell_strain_fd"] = str(check_cell_derivative(ppot, psys, torch.tensor(si.get_positions()),
                                                             torch.tensor(si.cell.array)))
    report["derivative_checks"] = checks
    if benchmark:
        report["device_benchmark"] = device_benchmark(pot)
    report["passed"] = all(v.startswith("[PASS]") for v in checks.values()) and \
        report["parity"]["same_settings"]["max_abs_energy_diff_eV"] < 1e-6
    report["wall_seconds"] = time.time() - t0
    with open(os.path.join(out_dir, "audit.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report
