"""Vib-v1 experiment helpers (experiments/vib_v1/PROTOCOL.md).

Development data only: the sealed held-out set is read exclusively by
scripts/unblind_vib_v1.py.

  * load_dev_molecules: development molecules with G2 starting geometries
    (fixes the conformer; isotopes carried over from the SMILES);
  * base_precompute: unchanged-model minimum, symmetry assignment, GVPT2
    anharmonic corrections and the scored-mode set S_m (label-free rules, 3.2);
  * make_task: training/evaluation task with the corrected forward model
    nu = omega_model + Delta_anh(base) and value-free symmetry assignment.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from .config import DataConfig
from .data import ExperimentRegistry
from .data.schema import Observation
from .observables.symmetry import assign_by_symmetry, find_point_group, symmetry_mode_groups
from .observables.vibrational import (
    ExperimentalMode,
    VibrationalConfig,
    VibrationalForwardModel,
    VibrationalObservable,
    harmonic_modes,
    optimize_geometry,
)
from .observables.vpt2 import anharmonic_corrections
from .systems import System, from_smiles

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEV_FILES = [os.path.join(ROOT, "data", "processed", "vib_v1_dev_pilot.parquet"),
             os.path.join(ROOT, "data", "processed", "vib_v1_dev_ext.parquet")]
# PROTOCOL 3.2 (label-free scored-mode rules, applied from the base model)
MIN_OMEGA = 400.0
MAX_REL_DELTA = 0.08  # DECISIONS D-20: ~1.7x the Dev maximum (0.048); larger corrections = VPT2 breakdown
MIN_WEIGHT = 0.7
# PROTOCOL 3.1 / 6 forward-model variances
ANH_REL_SD = 0.15
FLOOR_SD = 10.0


@dataclass
class Molecule:
    system_id: str
    system: System
    observations: list[Observation]  # full source table, sorted by mode number
    point_group: str
    group: str  # isotopologue-merging molecule group for CV folds


def _usable(o: Observation, exclude=None) -> bool:
    exclude = DataConfig().exclude_flags if exclude is None else exclude
    return not any(f in o.quality_flags for f in exclude) and o.uncertainty.standard_sd() is not None


def g2_by_block() -> dict[str, str]:
    """InChIKey first block -> G2 name (closed-shell neutral species with perceivable bonds)."""
    from ase.collections import g2
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    out = {}
    for name in g2.names:
        a = g2[name]
        if np.abs(a.get_initial_magnetic_moments()).sum() > 0 or abs(a.get_initial_charges().sum()) > 1e-9:
            continue
        xyz = f"{len(a)}\n\n" + "\n".join(f"{s} {x:.8f} {y:.8f} {z:.8f}"
                                          for s, (x, y, z) in zip(a.get_chemical_symbols(), a.positions))
        try:
            m = Chem.MolFromXYZBlock(xyz)
            rdDetermineBonds.DetermineBonds(m, charge=0)
            out.setdefault(Chem.MolToInchiKey(m).split("-")[0], name)
        except Exception:  # noqa: BLE001 - species without a perceivable bond graph are skipped
            continue
    return out


def _system_for(sid: str, smiles: str, g2_name: str | None) -> System:
    """G2 geometry (with SMILES isotopes when uniform per element), else RDKit embedding."""
    from ase.collections import g2
    from rdkit import Chem

    if g2_name is None:
        return from_smiles(smiles, system_id=sid)
    atoms = g2[g2_name]
    m = Chem.MolFromSmiles(smiles)
    iso = defaultdict(set)
    for at in Chem.AddHs(m).GetAtoms():
        iso[at.GetAtomicNum()].add(at.GetIsotope())
    if any(len(v) > 1 for v in iso.values()):  # mixed isotopes of one element: positions matter
        return from_smiles(smiles, system_id=sid)
    over = {i: next(iter(iso[z])) for i, z in enumerate(atoms.get_atomic_numbers()) if next(iter(iso[z]))}
    return System.from_ase(atoms, system_id=sid, isotopes=over or None, smiles=smiles,
                           provenance={"initial_geometry": f"ase.collections.g2:{g2_name}"})


def load_dev_molecules(skip_linear: bool = True) -> tuple[list[Molecule], dict]:
    from rdkit import Chem

    by = defaultdict(list)
    for f in DEV_FILES:
        if os.path.exists(f):
            for o in ExperimentRegistry.from_parquet(f):
                by[o.system_ids[0]].append(o)
    blocks = g2_by_block()
    mols, skipped = [], {}
    for sid, obs in sorted(by.items()):
        obs = sorted(obs, key=lambda o: o.labels["mode_number"])
        smiles = obs[0].labels.get("smiles")
        parent = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)
        g2_name = obs[0].labels.get("g2_name") or blocks.get(Chem.MolToInchiKey(Chem.MolFromSmiles(parent))[:14])
        system = _system_for(sid, smiles, g2_name)
        pg = obs[0].labels.get("point_group") or ""
        if skip_linear and ("∞" in pg or "inf" in pg.lower()):
            skipped[sid] = "linear (PROTOCOL 2.3)"
            continue
        mols.append(Molecule(sid, system, obs, pg, obs[0].groups.molecule_group))
    from .data.reserved import assert_not_reserved

    assert_not_reserved([m.group for m in mols], "vib_v1 development molecules")
    return mols, skipped


def _symmetrized_hessian(H: torch.Tensor, pg) -> torch.Tensor:
    from .observables.symmetry import operation_matrix

    Ms = [torch.tensor(operation_matrix(o)) for o in pg.ops]
    return sum(M @ H @ M.T for M in Ms) / len(Ms)


def base_precompute(potential, mol: Molecule, step: float = 0.025, fmax: float = 1e-5) -> dict:
    """Unchanged-model minimum, symmetry assignment, GVPT2 corrections (primary + sensitivity variants)
    and the pre-registered scored set S_m with per-mode flags for the sensitivity analyses."""
    from .observables.internal import torsion_fraction
    from .observables.vibrational import modes_from_hessian
    from .observables.vpt2 import normal_mode_force_field

    s = mol.system
    x, opt = optimize_geometry(potential, s, None, fmax=fmax, max_steps=5000)
    nm = harmonic_modes(potential, s, x, create_graph=False)
    nu = nm.wavenumbers.detach().numpy()
    rec = {"system_id": mol.system_id, "point_group": mol.point_group, "converged": bool(opt["converged"]),
           "max_force": float(opt["max_force"]), "n_imaginary": int((nu < 0).sum()),
           "geometry": x.numpy().tolist(), "omega_all": nu.tolist(), "step": step}
    masses = np.asarray(s.masses, float)
    pg = find_point_group(x.numpy(), masses, np.asarray(s.atomic_numbers))
    groups = symmetry_mode_groups(pg, nm.eigenvalues.detach().numpy(), nm.modes.numpy(), nu)
    obs = mol.observations
    args = ([o.labels["species"] for o in obs], [o.labels["mode_number"] for o in obs], [o.value for o in obs])
    res = assign_by_symmetry(pg, groups, *args, expected_point_group=mol.point_group)
    if not res.point_group_ok and any("non-irreducible" in n for n in res.notes):
        # fallback (PROTOCOL 3.3): symmetrize the Hessian over the detected group, re-diagonalize
        nm = modes_from_hessian(_symmetrized_hessian(nm.hessian.detach(), pg), x, s.mass_tensor(dtype=torch.float64))
        nu = nm.wavenumbers.detach().numpy()
        groups = symmetry_mode_groups(pg, nm.eigenvalues.detach().numpy(), nm.modes.numpy(), nu)
        res = assign_by_symmetry(pg, groups, *args, expected_point_group=mol.point_group)
        res.notes.append("assignment from the group-symmetrized Hessian")
    rec.update(point_group_ok=res.point_group_ok, merged_classes=res.merged_classes, notes=res.notes,
               point_group_order=pg.order, point_group_name=pg.name, modes_matrix=nm.modes.numpy().tolist())
    tf = torsion_fraction(x.numpy(), masses, s.atomic_numbers, nm.modes.numpy())
    decouple = (nu < MIN_OMEGA) | (tf >= 0.5)
    rec["torsion_fraction_all"] = tf.tolist()
    ac = ac_cpl = ac_nocor = None
    if rec["converged"] and rec["n_imaginary"] == 0:
        ff = normal_mode_force_field(potential, s, x, step=step, nm=nm)
        gidx = [g.indices for g in groups]  # average only within irrep-pure groups (critic report #3 P1)
        ac = anharmonic_corrections(potential, s, x, step=step, nm=nm, ff=ff, decouple=decouple, groups=gidx)
        ac_cpl = anharmonic_corrections(potential, s, x, step=step, nm=nm, ff=ff, groups=gidx)  # coupled
        ac_nocor = anharmonic_corrections(potential, s, x, step=step, nm=nm, ff=ff, decouple=decouple,
                                          coriolis=False, groups=gidx)  # no Coriolis terms
        rec["vpt2"] = {"delta_all": ac.delta.tolist(), "weight_all": ac.weight.tolist(),
                       "resonances": [[v if isinstance(v, str) else float(v) for v in r] for r in ac.resonances],
                       "harmonic_check": ac.harmonic_check, "decoupled": decouple.tolist()}
    # irreps of the base groups (crossing risk: a same-irrep neighbour within 30 cm-1)
    key_of = {}
    for g in groups:
        for i in g.indices:
            key_of[i] = tuple(np.round(g.characters, 1))
    in_res = {int(i) for r in (ac.resonances if ac else []) for i in r[1:4]}
    match = dict(res.matches)
    modes = []
    for e, o in enumerate(obs):
        m = {"label": f"{o.labels['species']}{o.labels['mode_number']}", "species": o.labels["species"],
             "number": o.labels["mode_number"], "degeneracy": o.labels["degeneracy"], "experiment": o.value,
             "sigma": o.uncertainty.standard_sd(), "rating": o.labels.get("rating"),
             "approximate_type": o.labels.get("approximate_type") or "", "usable_source": _usable(o),
             "comp_indices": match.get(e)}
        why = None
        if not m["usable_source"]:
            why = "source flags"
        elif e in res.excluded_exp or e not in match:
            why = "symmetry class count mismatch" if res.point_group_ok else "point group mismatch"
        elif ac is None:
            why = "base model invalid (not converged or imaginary modes)"
        if m["comp_indices"] is not None and ac is not None:
            idx = m["comp_indices"]
            V = nm.modes.numpy()[:, idx].reshape(-1, 3, len(idx))
            h = np.asarray(s.atomic_numbers) == 1
            om = float(np.mean(nu[idx]))
            same = [j for j in range(len(nu)) if j not in idx and key_of.get(j) == key_of.get(idx[0])]
            m.update(omega_base=om, delta=float(np.mean(ac.delta[idx])),
                     delta_coupled=float(np.mean(ac_cpl.delta[idx])),
                     delta_nocoriolis=float(np.mean(ac_nocor.delta[idx])),
                     weight=float(np.min(ac.weight[idx])), h_fraction=float((V[h] ** 2).sum() / (V**2).sum()),
                     torsion_fraction=float(np.max(tf[idx])),
                     crossing_risk=bool(any(abs(nu[j] - om) < 30.0 for j in same)),
                     resonance=bool(set(idx) & in_res))
            if why is None:
                if om < MIN_OMEGA:
                    why = f"omega_base < {MIN_OMEGA:.0f}"
                elif "torsion" in m["approximate_type"].lower() or m["torsion_fraction"] >= 0.5:
                    why = "torsion"
                elif abs(m["delta"]) > MAX_REL_DELTA * om:
                    why = f"|Delta_anh| > {MAX_REL_DELTA} omega"
                elif m["weight"] < MIN_WEIGHT:
                    why = f"resonance weight < {MIN_WEIGHT}"
        m["scored"] = why is None
        m["excluded_because"] = why
        modes.append(m)
    rec["modes"] = modes
    rec["n_scored"] = sum(m["scored"] for m in modes)
    return rec


def save_precompute(rec: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{rec['system_id']}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    return path


def make_task(mol: Molecule, pre: dict, memory_mode: str = "hvp", fmax: float = 1e-5, anchor: bool = True,
              weighting: str = "per_molecule"):
    """DirectTask scoring the pre-registered set S_m with the corrected forward model.

    anchor: identify this model's modes with the base model's assigned modes by eigenvector-subspace
      tracking (never by re-ranking; critic report #2 M1).
    weighting: "per_molecule" (training objective, PROTOCOL 6: every molecule counts equally and its modes
      uniformly, matching the per-molecule endpoint) or "likelihood" (sigma_exp, sigma_anh, floor).
    """
    from .losses.objective import Target
    from .observables.vibrational import ModeMatch
    from .training.tasks import DirectTask

    n_scored = max(1, sum(bool(m["scored"]) for m in pre["modes"]))
    modes = []
    for m in pre["modes"]:
        scored = bool(m["scored"])
        per_mol = weighting == "per_molecule"
        modes.append(ExperimentalMode(
            label=m["label"], value=float(m["experiment"]), sigma=0.0 if per_mol else float(m["sigma"] or 0.0),
            degeneracy=int(m["degeneracy"]), species=m["species"], number=int(m["number"]),
            anharmonic_correction=float(m["delta"]) if scored else None,
            anharmonic_sigma=(0.0 if per_mol else ANH_REL_SD * abs(float(m["delta"]))) if scored else None,
            usable=scored))
    floor = FLOOR_SD * (np.sqrt(n_scored) if weighting == "per_molecule" else 1.0)
    fwd = VibrationalForwardModel(kind="fundamental_corrected", rel_model_sd=0.0, abs_model_sd=floor,
                                  min_abs_wavenumber=50.0)
    cfg = VibrationalConfig(geometry_response="implicit", fmax=fmax, max_opt_steps=5000, memory_mode=memory_mode,
                            assignment="symmetry", point_group=mol.point_group, forward=fwd, min_track_overlap=0.5)
    system = mol.system.with_positions(np.asarray(pre["geometry"]))
    obs = VibrationalObservable(system, modes, cfg)
    obs.last_geometry = torch.tensor(pre["geometry"], dtype=torch.float64)
    if anchor:
        obs.base_matches = [ModeMatch(e, list(m["comp_indices"])) for e, m in enumerate(pre["modes"]) if m["scored"]]
        obs.reference_modes = np.asarray(pre["modes_matrix"])
        obs.assignment_info = {"anchored_to_base": True, "point_group_order": pre.get("point_group_order")}

    def target_fn(pred, _obs=obs, _sid=mol.system_id):
        y, var = _obs.targets()
        used = [f"{_sid}:{_obs.exp_modes[i].label}" for i in pred.meta["exp_indices"]]
        return Target(used, "vibrational", y, var.sqrt())

    return DirectTask(f"vib:{mol.system_id}", "vibrational", obs, target_fn=target_fn)


def evaluate_anchored(pot, mol: Molecule, pre: dict, min_overlap: float = 0.5) -> tuple[dict | None, str]:
    """Predicted fundamentals on S_m for any model, anchored to the base assignment.

    Invalid (None, reason) if the geometry does not converge, any mode is imaginary, the model's minimum
    has a different point group from the base minimum, or a scored mode's tracking overlap < min_overlap.
    """
    task = make_task(mol, pre, memory_mode="hvp", anchor=True, weighting="likelihood")
    task.observable.config.geometry_response = "fixed"  # values only
    pred = task.observable.predict(pot)
    if not pred.valid:
        return None, pred.reason
    xg = task.observable.last_geometry.numpy()
    pg = find_point_group(xg, np.asarray(mol.system.masses, float), np.asarray(mol.system.atomic_numbers))
    if pg.name != pre.get("point_group_name"):
        return None, f"point group {pg.name} at this model's minimum vs {pre.get('point_group_name')} (base)"
    worst = 1.0
    for mm in task.observable.matches:
        if mm.note.startswith("overlap="):
            worst = min(worst, float(mm.note.split("=")[1]))
    if worst < min_overlap:
        return None, f"mode tracking overlap {worst:.2f} < {min_overlap}"
    labels = [task.observable.exp_modes[i].label for i in pred.meta["exp_indices"]]
    vals = pred.value.detach().numpy()
    return dict(zip(labels, vals.tolist())), f"min overlap {worst:.3f}"


# ---------------------------------------------------------------------------
# scoring (shared by the dev rehearsal and the one-shot unblinding)
# ---------------------------------------------------------------------------

KT_300K_EV = 0.025852  # k_B * 300 K


def lsq_scale(pre_by_sid: dict) -> float:
    """B-scale: one least-squares scale on the scored modes of the given (development) molecules."""
    num = den = 0.0
    for r in pre_by_sid.values():
        for m in r["modes"]:
            if m["scored"]:
                num += m["experiment"] * m["omega_base"]
                den += m["omega_base"] ** 2
    return num / den


def mode_table(model_name: str, pot, mol: Molecule, pre: dict, scale: float | None = None) -> list[dict]:
    """Per-mode predictions on S_m. model_name 'B-VPT2'/'B-scale' use the precompute; others are evaluated."""
    scored = [m for m in pre["modes"] if m["scored"]]
    merged = {sp for c in pre.get("merged_classes", []) for sp in c}

    hcno = set(np.asarray(mol.system.atomic_numbers).tolist()) <= {1, 6, 7, 8}

    def row(m, prediction):
        return {"system": mol.system_id, "model": model_name, "mode": m["label"], "experiment": m["experiment"],
                "prediction": prediction, "omega_base": m["omega_base"], "delta": m["delta"],
                "delta_coupled": m["delta_coupled"], "delta_nocoriolis": m["delta_nocoriolis"],
                "rating": m.get("rating"), "crossing_risk": m["crossing_risk"], "resonance": m["resonance"],
                "xh": bool(m["h_fraction"] >= 0.5 and m["omega_base"] > 1800.0),
                "merged_class": m["species"] in merged, "stratum": "HCNO_only" if hcno else "other_elements",
                "n_scored": len(scored)}

    if model_name == "B-VPT2":
        return [row(m, m["omega_base"] + m["delta"]) for m in scored]
    if model_name == "B-scale":
        return [row(m, scale * m["omega_base"]) for m in scored]
    got, why = evaluate_anchored(pot, mol, pre)
    if got is None:
        return [{"system": mol.system_id, "model": model_name, "invalid": why}]
    if set(got) != {m["label"] for m in scored}:
        return [{"system": mol.system_id, "model": model_name, "invalid": "anchored modes do not cover S_m"}]
    return [row(m, got[m["label"]]) for m in scored]


def selfconsistent_rows(ft, mol: Molecule, pre: dict, step: float = 0.025) -> list[dict]:
    """FT with Delta_anh recomputed from FT's own force field (sensitivity, PROTOCOL 5)."""
    from .observables.internal import torsion_fraction
    from .observables.vibrational import track_modes
    from .observables.vpt2 import normal_mode_force_field

    rows = mode_table("FT", ft, mol, pre)
    if any("invalid" in r for r in rows):
        return [dict(r, model="FT-selfconsistent") for r in rows]
    s = mol.system
    x, _ = optimize_geometry(ft, s, torch.tensor(pre["geometry"]), fmax=1e-5, max_steps=5000)
    nm = harmonic_modes(ft, s, x, create_graph=False)
    nu = nm.wavenumbers.detach().numpy()
    from .observables.vibrational import ModeMatch

    base_matches = [ModeMatch(e, list(m["comp_indices"])) for e, m in enumerate(pre["modes"]) if m["scored"]]
    pg = find_point_group(x.numpy(), np.asarray(s.masses, float), np.asarray(s.atomic_numbers))
    gidx = [g.indices for g in symmetry_mode_groups(pg, nm.eigenvalues.detach().numpy(), nm.modes.numpy(), nu)]
    tracked = {mm.exp_index: mm.comp_indices for mm in track_modes(np.asarray(pre["modes_matrix"]), base_matches,
                                                                  nm.modes.numpy(), nu, gidx)}
    tf = torsion_fraction(x.numpy(), np.asarray(s.masses, float), s.atomic_numbers, nm.modes.numpy())
    ff = normal_mode_force_field(ft, s, x, step=step, nm=nm)
    ac = anharmonic_corrections(ft, s, x, step=step, nm=nm, ff=ff, decouple=(nu < MIN_OMEGA) | (tf >= 0.5),
                                groups=gidx)
    by_label = {r["mode"]: r for r in rows}
    out = []
    for e, m in enumerate(pre["modes"]):
        if m["scored"]:
            r = dict(by_label[m["label"]], model="FT-selfconsistent")
            idx = tracked[e]
            r["prediction"] = float(np.mean(nu[idx]) + np.mean(ac.delta[idx]))
            out.append(r)
    return out


def drift(base, ft, mol: Molecule, pre: dict, n: int = 20, seed: int = 20261006) -> dict:
    """RMS force / energy-per-atom deviation of ft from base on 300 K normal-mode displacements."""
    from .observables.vibrational import modes_from_hessian
    from .potential.derivatives import hessian

    s = mol.system
    x0 = torch.tensor(pre["geometry"], dtype=torch.float64)
    masses = s.mass_tensor(dtype=torch.float64)
    nm = modes_from_hessian(hessian(base, s, x0, create_graph=False).detach(), x0, masses)
    from . import units

    lam_200 = (200.0 / units.EIGVAL_TO_WAVENUMBER) ** 2  # amplitude cap: modes below 200 cm-1 move like 200 cm-1
    lam = nm.eigenvalues.detach().clamp_min(lam_200)
    rng = np.random.default_rng(seed)
    mih = masses.repeat_interleave(3).rsqrt()
    df, de = [], []
    for _ in range(n):
        Q = torch.tensor(rng.normal(size=lam.numel())) * (KT_300K_EV / lam).sqrt()
        x = x0 + (mih * (nm.modes @ Q)).reshape(-1, 3)
        eb, fb = base.energy_and_forces(s, x)
        ef, ff = ft.energy_and_forces(s, x)
        df.append(float(((ff - fb) ** 2).mean()))
        de.append(float(ef - eb))
    de = np.asarray(de)
    return {"system": mol.system_id, "force_rms_dev": float(np.sqrt(np.mean(df))),
            "rel_energy_rms_dev": float(np.sqrt(np.mean((de - de.mean()) ** 2))),  # offset removed
            "energy_offset_per_atom": float(de.mean()) / len(s.atomic_numbers)}


MEI = 2.5  # cm^-1, minimum effect of interest (PROTOCOL 5; critic report #2 M4)
DRIFT_FORCE_RMS = 0.05  # eV/A, absolute bound fixed in advance (PROTOCOL 5.1)
DRIFT_REL_ENERGY_RMS = 0.02  # eV per molecule (relative energies, offset removed)


def _e(rows, model, field="prediction"):
    from .validation.paired import per_molecule_rmse

    return per_molecule_rmse(rows[rows.model == model], field)


def analyse(rows: list[dict], drift_rows: list[dict], seed: int = 20261006) -> dict:
    """The pre-registered analysis (PROTOCOL 5) on per-mode prediction rows."""
    import pandas as pd

    from .validation.paired import cluster_bootstrap_metric, paired_bootstrap, sign_flip_test, wilcoxon_paired

    df = pd.DataFrame(rows)
    if "invalid" not in df:
        df["invalid"] = None
    ok = df[df["invalid"].isna()].copy()
    models = sorted(df.model.unique())
    mols = sorted(ok[ok.model == "B-VPT2"].system.unique())  # S_m defined by the base model
    e = {mdl: _e(ok, mdl) for mdl in models}
    n_invalid = {mdl: int(df[(df.model == mdl) & df["invalid"].notna()].system.nunique()) for mdl in models}
    # primary: conservative imputation (critic report #3 P3): an invalid prediction gets
    # max(e_m of B-VPT2, that model's worst valid e_m), so a failure can never count as an improvement
    imp = {}
    eb_all = e["B-VPT2"].reindex(mols)
    for mdl, v in e.items():
        worst = float(v.max()) if len(v) else np.nan
        w = v.reindex(mols)
        imp[mdl] = w.where(w.notna(), np.maximum(eb_all, worst))
    out = {"n_molecules": len(mols), "n_invalid": n_invalid, "mean_e": {k: float(v.mean()) for k, v in imp.items()},
           "mei_cm": MEI}
    h1 = paired_bootstrap(imp["FT"], imp["B-VPT2"], seed=seed)
    out["H1_FT_vs_BVPT2"] = h1
    out["H1_meaningful_improvement"] = bool(h1["ci"][1] < -MEI)  # primary decision
    out["any_improvement"] = bool(h1["ci"][1] < 0)
    out["harm"] = bool(h1["ci"][0] > 0)
    out["meaningful_harm"] = bool(h1["ci"][0] > MEI)
    out["sign_flip_FT_vs_BVPT2"] = sign_flip_test(imp["FT"], imp["B-VPT2"], seed=seed)
    out["wilcoxon_FT_vs_BVPT2"] = wilcoxon_paired(imp["FT"], imp["B-VPT2"])
    out["FT_vs_Bscale"] = paired_bootstrap(imp["FT"], imp["B-scale"], seed=seed)
    sub = ok[ok.system.isin(mols)].astype({"crossing_risk": bool, "resonance": bool, "merged_class": bool,
                                            "xh": bool, "n_scored": int})
    out["pooled"] = {mdl: {m: cluster_bootstrap_metric(sub[sub.model == mdl], "prediction", metric=m, seed=seed)
                           for m in ("rmse", "mae")} for mdl in models}
    out["by_class_rmse"] = {mdl: {cls: float(np.sqrt(((g.prediction - g.experiment) ** 2).mean()))
                                  for cls, g in sub[sub.model == mdl].groupby(
                                      np.where(sub[sub.model == mdl].xh, "XH_stretch", "other"))}
                            for mdl in models}
    sens = {}

    def paired_on(frame, name):
        a, b = _e(frame, "FT"), _e(frame, "B-VPT2")
        idx = a.index.intersection(b.index)
        if len(idx) >= 2:
            sens[name] = paired_bootstrap(a.loc[idx], b.loc[idx], seed=seed)

    paired_on(sub, "complete_case")
    paired_on(sub[sub.rating.isin(["A", "B", "C"])], "ratings_A_to_C")
    paired_on(sub[~sub.crossing_risk], "no_crossing_risk_modes")
    paired_on(sub[~sub.resonance], "no_resonance_modes")
    paired_on(sub[~sub.merged_class], "no_merged_class_modes")
    paired_on(sub[sub.n_scored >= 3], "molecules_with_3plus_modes")
    for st in ("HCNO_only", "other_elements"):
        paired_on(sub[sub.stratum == st], f"stratum_{st}")
    for alt in ("delta_coupled", "delta_nocoriolis"):
        t = sub.copy()
        c = t.model.isin(["FT", "B-VPT2"])
        t.loc[c, "prediction"] = t.loc[c, "prediction"] - t.loc[c, "delta"] + t.loc[c, alt]
        paired_on(t, f"vpt2_{alt}")
    for f in (0.85, 1.15):
        t = sub.copy()
        c = t.model.isin(["FT", "B-VPT2"])
        t.loc[c, "prediction"] = t.loc[c, "prediction"] + (f - 1.0) * t.loc[c, "delta"]
        paired_on(t, f"delta_x{f}")
    if "FT-selfconsistent" in models:
        a, b = imp.get("FT-selfconsistent"), imp["B-VPT2"]
        sens["ft_selfconsistent_vpt2"] = paired_bootstrap(a, b, seed=seed)
    d = imp["FT"] - imp["B-VPT2"]
    loo = [float((d.sum() - d.loc[k]) / (len(d) - 1)) for k in d.index] if len(d) > 2 else []
    sens["loo_delta_range"] = [min(loo), max(loo)] if loo else None
    out["sensitivity"] = sens
    if drift_rows:
        dr = pd.DataFrame(drift_rows)
        out["drift"] = {"force_rms_dev_max": float(dr.force_rms_dev.max()),
                        "force_rms_dev_median": float(dr.force_rms_dev.median()),
                        "rel_energy_rms_dev_max": float(dr.rel_energy_rms_dev.max()),
                        "bounds": {"force_rms": DRIFT_FORCE_RMS, "rel_energy_rms": DRIFT_REL_ENERGY_RMS},
                        "exceeds_force": bool(dr.force_rms_dev.max() > DRIFT_FORCE_RMS),
                        "exceeds_rel_energy": bool(dr.rel_energy_rms_dev.max() > DRIFT_REL_ENERGY_RMS),
                        "energy_offset_per_atom_median": float(dr.energy_offset_per_atom.median())}
    return out
