"""Static description of a physical system handed to potentials and samplers.

A `System` fixes everything that does not change during sampling or training:
elements, atom ordering, masses (isotope-resolved when spectroscopy needs it),
total charge, the UMA spin descriptor, the UMA task, periodicity and molecule
membership. Positions and cells are *inputs* to the potential and are passed
separately so they can carry autograd graphs.

`atom_order_hash` identifies (elements, ordering, masses, charge, spin, task,
pbc); sample caches store it and refuse to be reused for a different system.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from ase import Atoms
from ase.data import chemical_symbols

from .isotopes import isotopologue_masses, standard_atomic_weight

# UMA task conventions verified against fairchem-core 2.23.0
# (units/mlip_unit/api/inference.py: validate_uma_atoms_data):
#   omol: total charge (int) and spin multiplicity (int, default 1)
#   other tasks: charge=0 and spin=0 expected
OMOL_TASK = "omol"
PERIODIC_TASKS = frozenset({"omat", "omc", "oc20", "oc25", "odac"})


@dataclass(frozen=True)
class System:
    system_id: str
    atomic_numbers: tuple[int, ...]
    masses: tuple[float, ...]
    charge: int = 0
    spin: int = 1
    task: str = OMOL_TASK
    pbc: tuple[bool, bool, bool] = (False, False, False)
    # Molecule id for every atom; used for COM-scaled barostat moves,
    # intra/intermolecular RDF splits and molecule counting.
    molecule_index: tuple[int, ...] | None = None
    # Reference coordinates (A) and cell (A); not used by potentials directly.
    positions: np.ndarray | None = field(default=None, compare=False, repr=False)
    cell: np.ndarray | None = field(default=None, compare=False, repr=False)
    mass_convention: str = "isotope"  # "isotope" or "standard_atomic_weight"
    smiles: str | None = None
    provenance: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        n = len(self.atomic_numbers)
        if len(self.masses) != n:
            raise ValueError("masses and atomic_numbers differ in length")
        if self.molecule_index is not None and len(self.molecule_index) != n:
            raise ValueError("molecule_index must have one entry per atom")
        if self.task != OMOL_TASK and (self.charge != 0 or self.spin != 0):
            raise ValueError(
                f"UMA task '{self.task}' expects charge=0 and spin=0; got "
                f"charge={self.charge}, spin={self.spin}"
            )
        if self.task == OMOL_TASK and self.spin < 1:
            raise ValueError("omol spin is the spin multiplicity (>= 1)")
        if any(self.pbc) and self.task == OMOL_TASK:
            # Allowed, but OMol25 training data are aperiodic (UMA docs):
            # callers must opt in through provenance to make the choice explicit.
            if not self.provenance.get("periodic_omol_acknowledged", False):
                raise ValueError(
                    "Periodic system with the aperiodic 'omol' task: set "
                    "provenance['periodic_omol_acknowledged']=True after "
                    "deciding this is scientifically justified."
                )

    # ------------------------------------------------------------------
    @property
    def n_atoms(self) -> int:
        return len(self.atomic_numbers)

    @property
    def is_periodic(self) -> bool:
        return any(self.pbc)

    @property
    def total_mass(self) -> float:
        return float(sum(self.masses))

    @property
    def composition(self) -> dict[str, int]:
        return dict(Counter(chemical_symbols[z] for z in self.atomic_numbers))

    @property
    def formula(self) -> str:
        comp = self.composition
        order = sorted(comp, key=lambda s: (s not in ("C", "H"), s != "C", s))
        return "".join(f"{s}{comp[s] if comp[s] > 1 else ''}" for s in order)

    @property
    def n_molecules(self) -> int:
        if self.molecule_index is None:
            return 1
        return len(set(self.molecule_index))

    def atom_order_hash(self) -> str:
        payload = {
            "Z": list(self.atomic_numbers),
            "m": [round(m, 8) for m in self.masses],
            "q": self.charge,
            "s": self.spin,
            "task": self.task,
            "pbc": list(self.pbc),
            "mol": list(self.molecule_index) if self.molecule_index else None,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    def z_tensor(self, device="cpu") -> torch.Tensor:
        return torch.as_tensor(self.atomic_numbers, dtype=torch.long, device=device)

    def mass_tensor(self, dtype=torch.float64, device="cpu") -> torch.Tensor:
        return torch.as_tensor(self.masses, dtype=dtype, device=device)

    def positions_tensor(self, dtype=torch.float64, device="cpu") -> torch.Tensor:
        if self.positions is None:
            raise ValueError(f"System {self.system_id} has no reference positions")
        return torch.as_tensor(np.asarray(self.positions), dtype=dtype, device=device)

    def cell_tensor(self, dtype=torch.float64, device="cpu") -> torch.Tensor:
        if self.cell is None:
            return torch.zeros(3, 3, dtype=dtype, device=device)
        return torch.as_tensor(np.asarray(self.cell), dtype=dtype, device=device)

    def with_isotopes(self, overrides: dict[int, int], system_id: str | None = None) -> System:
        """Isotopologue with atom-index -> mass-number overrides (same PES)."""
        masses = isotopologue_masses(self.atomic_numbers, overrides)
        return replace(
            self,
            masses=masses,
            mass_convention="isotope",
            system_id=system_id or f"{self.system_id}[{_iso_label(self, overrides)}]",
        )

    def with_positions(self, positions, cell=None) -> System:
        return replace(self, positions=np.asarray(positions, dtype=float),
                       cell=None if cell is None else np.asarray(cell, dtype=float))

    def to_ase(self, positions=None, cell=None) -> Atoms:
        pos = self.positions if positions is None else positions
        c = self.cell if cell is None else cell
        atoms = Atoms(numbers=list(self.atomic_numbers), positions=np.asarray(pos),
                      cell=None if c is None else np.asarray(c), pbc=list(self.pbc))
        atoms.set_masses(list(self.masses))
        atoms.info.update(charge=int(self.charge), spin=int(self.spin))
        return atoms

    @classmethod
    def from_ase(
        cls,
        atoms: Atoms,
        system_id: str,
        task: str = OMOL_TASK,
        charge: int | None = None,
        spin: int | None = None,
        mass_convention: str = "isotope",
        isotopes: dict[int, int] | None = None,
        molecule_index=None,
        provenance: dict | None = None,
        smiles: str | None = None,
    ) -> System:
        zs = tuple(int(z) for z in atoms.get_atomic_numbers())
        if mass_convention == "isotope":
            masses = isotopologue_masses(zs, isotopes)
        elif mass_convention == "standard_atomic_weight":
            masses = tuple(standard_atomic_weight(z) for z in zs)
        else:
            raise ValueError(mass_convention)
        if atoms.has("masses") and np.abs(np.asarray(atoms.get_masses()) - np.asarray(masses)).max() > 1e-6:
            # masses come from the isotope specification; never drop explicitly set masses silently
            raise ValueError("Atoms carry explicit masses that differ from the requested mass convention; "
                             "pass isotopes={atom_index: mass_number} (or use System.with_isotopes)")
        if charge is None:
            charge = int(atoms.info.get("charge", 0))
        if spin is None:
            spin = int(atoms.info.get("spin", 1 if task == OMOL_TASK else 0))
        pbc = tuple(bool(b) for b in atoms.pbc)
        cell = np.array(atoms.cell.array) if any(pbc) else None
        return cls(
            system_id=system_id,
            atomic_numbers=zs,
            masses=masses,
            charge=charge,
            spin=spin,
            task=task,
            pbc=pbc,
            molecule_index=None if molecule_index is None else tuple(int(i) for i in molecule_index),
            positions=np.array(atoms.get_positions()),
            cell=cell,
            mass_convention=mass_convention,
            smiles=smiles,
            provenance=dict(provenance or {}),
        )


def _iso_label(system: System, overrides: dict[int, int]) -> str:
    parts = [f"{a}:{m}{chemical_symbols[system.atomic_numbers[a]]}" for a, m in sorted(overrides.items())]
    return ",".join(parts)
