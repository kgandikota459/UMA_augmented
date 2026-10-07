"""FreeSolv hydration free energies (Mobley & Guthrie, JCAMD 28, 711 (2014)).

Only the experimental fields are used: `expt`, `d_expt`, `expt_reference`.
`calc*` fields are GAFF alchemical results and are never mapped to
experimental labels. When the database notes that the experimental
uncertainty was *assigned a default value*, the record is flagged
`default_uncertainty_assigned`. Values are hydration free energies in
kcal/mol at 298 K for transfer at equal concentration (Ben-Naim standard
state), which is what a decoupling cycle computes.

License: data CC BY 4.0 (repository LICENSE, verified 2026-09-26).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json

from ... import units
from ..schema import (
    Conditions,
    Observation,
    Phase,
    Provenance,
    Redistribution,
    SplitGroups,
    TransformStep,
    Uncertainty,
    UncertaintyType,
    ValueKind,
)

URL = "https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.json"
PARSER_VERSION = "freesolv_v1"


def fetch(path: str) -> dict:
    import requests

    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return {"path": path, "sha256": hashlib.sha256(r.content).hexdigest(), "url": URL,
            "retrieval_date": dt.date.today().isoformat()}


def to_observations(database: dict, fetch_info: dict, elements: set[str] | None = None,
                    inchikey_block=None, scaffold=None) -> list[Observation]:
    from rdkit import Chem

    out = []
    for key, rec in sorted(database.items()):
        smi = rec["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if elements is not None and not {a.GetSymbol() for a in mol.GetAtoms()} <= elements:
            continue
        notes = " ".join(rec.get("notes") or [])
        flags = []
        if "default" in notes.lower() and "uncertainty" in notes.lower():
            flags.append("default_uncertainty_assigned")
        ik = Chem.MolToInchiKey(mol)
        out.append(Observation(
            observation_id=f"freesolv:{key}",
            system_ids=[f"freesolv:{key}"],
            family="solvation_free_energy",
            quantity="hydration_free_energy",
            definition="Free energy of transfer from gas to water at 298 K, equal concentrations "
                       "(Ben-Naim standard state), from FreeSolv 'expt'.",
            value=float(rec["expt"]) * units.KCAL_MOL_TO_EV,
            units="eV",
            labels={"smiles": smi, "iupac": rec.get("iupac"), "expt_kcal_mol": rec["expt"],
                    "groups": rec.get("groups")},
            uncertainty=Uncertainty(kind=UncertaintyType.STANDARD,
                                    value=float(rec["d_expt"]) * units.KCAL_MOL_TO_EV,
                                    source_text=notes or None, conversion_rule="d_expt taken as 1 sd"),
            value_kind=ValueKind.EVALUATED,
            conditions=Conditions(temperature_k=298.15, pressure_bar=None, phase=Phase.SOLUTION, solvent="O",
                                  protocol="compiled hydration free energy", missing=["pressure"]),
            compatible_uma_tasks=["omol"],
            forward_model={"name": "solvation_cycle", "requires": ["reference MM decoupling", "UMA endpoints"]},
            provenance=Provenance(
                source_name="FreeSolv", source_url=fetch_info["url"], doi="10.1007/s10822-014-9747-x",
                source_record_id=key, retrieval_date=dt.date.fromisoformat(fetch_info["retrieval_date"]),
                dataset_version="master (retrieved)", license="CC BY 4.0",
                license_url="https://creativecommons.org/licenses/by/4.0/",
                redistribution=Redistribution.ALLOWED,
                citation="Mobley & Guthrie, J. Comput. Aided Mol. Des. 28, 711 (2014)",
                transformation_history=[
                    TransformStep(step="fetch", tool="requests", version="-",
                                  parameters={"sha256": fetch_info["sha256"]}),
                    TransformStep(step="select_experimental_fields", tool="umaexp", version=PARSER_VERSION,
                                  parameters={"fields": ["expt", "d_expt", "expt_reference"]}),
                ],
            ),
            groups=SplitGroups(molecule_group=ik.split("-")[0], scaffold_group=scaffold(smi) if scaffold else None,
                               provenance_group="freesolv",
                               publication_group=(rec.get("expt_reference") or None)),
            quality_flags=flags,
        ))
    return out


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
