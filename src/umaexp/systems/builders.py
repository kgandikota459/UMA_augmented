"""System builders: chemical identity + conditions -> explicit atomic systems.

Preparation heuristics here (ETKDG conformers, force-field pre-relaxation,
random packing) only produce *starting coordinates*. They are recorded in
provenance and are never treated as experimental evidence.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.neighborlist import NeighborList, natural_cutoffs

from .. import units
from .isotopes import standard_atomic_weight
from .system import OMOL_TASK, System


def from_smiles(
    smiles: str,
    system_id: str,
    seed: int = 0,
    isotopes: dict[int, int] | None = None,
    spin: int | None = None,
    mmff_relax: bool = True,
    task: str = OMOL_TASK,
) -> System:
    """Single conformer from SMILES (RDKit ETKDGv3, optional MMFF relaxation).

    Total charge is the sum of formal charges. Spin multiplicity defaults to
    (number of radical electrons) + 1; pass `spin` to override (e.g. triplet O2,
    whose SMILES does not encode the ground-state multiplicity).
    """
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem, Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"ETKDG embedding failed for {smiles!r}")
    relaxed = False
    if mmff_relax and AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        relaxed = True
    conf = mol.GetConformer()
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    zs = [a.GetAtomicNum() for a in mol.GetAtoms()]
    if isotopes is None:  # isotope labels written in the SMILES, e.g. [2H]O[2H]
        isotopes = {a.GetIdx(): a.GetIsotope() for a in mol.GetAtoms() if a.GetIsotope()} or None
    charge = int(Chem.GetFormalCharge(mol))
    if spin is None:
        spin = int(Descriptors.NumRadicalElectrons(mol)) + 1
    atoms = Atoms(numbers=zs, positions=pos)
    atoms.center()
    return System.from_ase(
        atoms,
        system_id=system_id,
        task=task,
        charge=charge,
        spin=spin,
        isotopes=isotopes,
        molecule_index=[0] * len(zs),
        smiles=Chem.MolToSmiles(Chem.RemoveHs(mol)),
        provenance={
            "builder": "rdkit_etkdg_v3",
            "rdkit_version": rdBase.rdkitVersion,
            "seed": int(seed),
            "mmff_relaxed": relaxed,
            "input_smiles": smiles,
            "note": "starting geometry only; not experimental evidence",
        },
    )


def from_file(path: str, system_id: str, **kwargs) -> System:
    atoms = ase_read(path)
    prov = {"builder": "ase.io.read", "path": str(path)}
    prov.update(kwargs.pop("provenance", {}))
    return System.from_ase(atoms, system_id=system_id, provenance=prov, **kwargs)


def molecule_index_from_connectivity(atoms: Atoms, scale: float = 1.15) -> list[int]:
    """Connected components under scaled covalent radii (ASE natural cutoffs)."""
    cutoffs = natural_cutoffs(atoms, mult=scale)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    n = len(atoms)
    label = [-1] * n
    current = 0
    for start in range(n):
        if label[start] >= 0:
            continue
        stack = [start]
        label[start] = current
        while stack:
            i = stack.pop()
            for j in nl.get_neighbors(i)[0]:
                if label[j] < 0:
                    label[j] = current
                    stack.append(j)
        current += 1
    return label


def liquid_box(
    molecule: System,
    n_molecules: int,
    density_g_cm3: float,
    seed: int = 0,
    min_distance: float = 1.8,
    max_attempts: int = 20000,
    system_id: str | None = None,
    task: str | None = None,
    acknowledge_periodic_omol: bool = False,
) -> System:
    """Randomly packed cubic periodic box of rigid copies of `molecule`.

    Uses standard atomic weights for the box masses (classical MD convention;
    configurational averages do not depend on masses). The initial density is
    a packing choice, not a measurement: NPT equilibration must follow.
    """
    if molecule.positions is None:
        raise ValueError("molecule needs reference positions")
    rng = np.random.default_rng(seed)
    mol_pos = np.asarray(molecule.positions) - np.asarray(molecule.positions).mean(axis=0)
    mol_mass = sum(standard_atomic_weight(z) for z in molecule.atomic_numbers)
    volume = n_molecules * mol_mass / (density_g_cm3 / units.U_PER_A3_TO_G_CM3)
    box = volume ** (1.0 / 3.0)

    placed: list[np.ndarray] = []
    all_pos = np.zeros((0, 3))
    attempts = 0
    while len(placed) < n_molecules:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Packing failed after {max_attempts} attempts ({len(placed)}/{n_molecules}); "
                "lower the density or min_distance"
            )
        rot = _random_rotation(rng)
        trial = mol_pos @ rot.T + rng.uniform(0, box, size=3)
        if len(all_pos):
            d = trial[:, None, :] - all_pos[None, :, :]
            d -= box * np.round(d / box)
            if np.min(np.linalg.norm(d, axis=-1)) < min_distance:
                continue
        placed.append(trial)
        all_pos = np.vstack([all_pos, trial])

    zs = list(molecule.atomic_numbers) * n_molecules
    mol_idx = [k for k in range(n_molecules) for _ in molecule.atomic_numbers]
    atoms = Atoms(numbers=zs, positions=all_pos, cell=np.eye(3) * box, pbc=True)
    use_task = task or molecule.task
    prov = {
        "builder": "liquid_box_random_packing",
        "seed": int(seed),
        "n_molecules": int(n_molecules),
        "initial_density_g_cm3": float(density_g_cm3),
        "min_distance_A": float(min_distance),
        "source_molecule": molecule.system_id,
        "note": "initial packing only; density must come from NPT sampling",
    }
    if use_task == OMOL_TASK:
        prov["periodic_omol_acknowledged"] = bool(acknowledge_periodic_omol)
    return System.from_ase(
        atoms,
        system_id=system_id or f"{molecule.system_id}_liquid{n_molecules}",
        task=use_task,
        charge=molecule.charge * n_molecules if use_task == OMOL_TASK else 0,
        spin=1 if use_task == OMOL_TASK else 0,
        mass_convention="standard_atomic_weight",
        molecule_index=mol_idx,
        provenance=prov,
    )


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([
        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d],
    ])
