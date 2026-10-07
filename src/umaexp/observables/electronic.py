r"""Electronic-state observables that a charge/spin-conditioned UMA can address.

Implemented (gas phase, `omol` task, which takes total charge and spin
multiplicity as inputs):

    IE_ad = E(M+, R*_+) - E(M, R*_0)  [+ Delta ZPE]
    EA_ad = E(M, R*_0)  - E(M-, R*_-) [+ Delta ZPE]

Energies at each state's own optimized geometry; composition is unchanged,
so element references cancel. By the envelope theorem the electronic part
needs no implicit-geometry term; ZPE uses implicitly differentiated modes.
Experimental adiabatic IE/EA values (e.g. NIST WebBook ion energetics) are
evaluated quantities with method-specific caveats (vertical vs adiabatic,
0 K vs 298 K); record which one each label is.

Not implemented (need components beyond the selected UMA model):
  pKa, redox potentials in solution: a solvation model for every state and
    a reference-electrode / proton solvation-free-energy convention;
  excited states and spectra: excited-state surfaces are outside a
    ground-state potential.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from ..potential.base import Potential
from ..systems.system import System
from .base import ExtensionRequired, Prediction
from .thermochemistry import rrho
from .vibrational import attach_implicit_geometry, harmonic_modes, optimize_geometry


@dataclass
class AdiabaticIonization:
    neutral: System
    ion: System  # same atoms; charge and spin changed
    kind: str = "IE"  # "IE" (cation) or "EA" (anion)
    include_zpe: bool = True

    def __post_init__(self):
        if self.neutral.atomic_numbers != self.ion.atomic_numbers:
            raise ValueError("neutral and ion must have identical atoms")
        if self.neutral.task != "omol" or self.ion.task != "omol":
            raise ValueError("charge/spin-resolved energies require the omol task")
        dq = self.ion.charge - self.neutral.charge
        if (self.kind == "IE" and dq != 1) or (self.kind == "EA" and dq != -1):
            raise ValueError(f"{self.kind} needs ion charge = neutral {'+' if self.kind == 'IE' else '-'} 1")

    @staticmethod
    def ion_of(neutral: System, charge_delta: int, spin: int) -> System:
        return replace(neutral, charge=neutral.charge + charge_delta, spin=spin,
                       system_id=f"{neutral.system_id}{'+' if charge_delta > 0 else '-'}")

    def _state_energy(self, potential, system):
        x_star, info = optimize_geometry(potential, system, fmax=1e-5)
        if not self.include_zpe:
            return potential.energy(system, x_star), info
        x = attach_implicit_geometry(potential, system, x_star)
        nm = harmonic_modes(potential, system, x, create_graph=True)
        zpe = rrho(nm.wavenumbers.clamp_min(1.0), x, system.mass_tensor(), 298.15)["zpe"]
        return potential.energy(system, x) + zpe, info

    def predict(self, potential: Potential) -> Prediction:
        e0, i0 = self._state_energy(potential, self.neutral)
        e1, i1 = self._state_energy(potential, self.ion)
        val = (e1 - e0) if self.kind == "IE" else (e0 - e1)
        ok = i0["converged"] and i1["converged"]
        return Prediction(val.reshape(1), torch.zeros(1, dtype=torch.float64), valid=ok,
                          diagnostics={"neutral": i0, "ion": i1}, meta={"units": "eV"})


def solution_redox_or_pka(*_args, **_kwargs):
    raise ExtensionRequired(
        "Solution-phase pKa/redox needs a solvation free energy for every protonation/charge "
        "state (explicit-solvent cycles via free_energy.SolvationCycle or a validated implicit "
        "model) plus a reference convention (SHE absolute potential / proton solvation energy)."
    )
