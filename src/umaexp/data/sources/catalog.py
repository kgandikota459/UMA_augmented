"""Candidate sources (D1-D11): role, access, licensing and curation rules.

This is a catalogue of *candidates*, not a claim that clean, mutually
compatible labels exist. `status` says what this repository does with each.
License/redistribution fields are what was verified on 2026-09-26; re-check
before any redistribution.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceInfo:
    key: str
    name: str
    url: str
    role: str  # experimental label | evaluated label | inferred model | computational anchor | infrastructure
    families: tuple[str, ...]
    access: str
    license: str
    redistribution: str
    curation: str
    status: str


CATALOG = {
    "D1": SourceInfo(
        "D1", "NIST Chemistry WebBook (SRD 69)", "https://webbook.nist.gov/chemistry/",
        "evaluated label", ("vibrational", "thermochemistry", "electronic_state"),
        "per-species HTML pages (cbook.cgi?ID=...&Mask=...), rate-limit politely",
        "NIST SRD, Standard Reference Data Act", "restricted (keep local; regenerate by script)",
        "Separate selected fundamentals (evaluated) from IR/Raman observed values, ratings, phases, "
        "isotopologues; flag Fermi resonance/inactive/Coriolis; ion energetics distinguish vertical/adiabatic.",
        "IMPLEMENTED: vibrational fundamentals parser (data/sources/nist_webbook.py)"),
    "D2": SourceInfo(
        "D2", "NIST TRC ThermoML Archive", "https://trc.nist.gov/ThermoML/",
        "experimental label", ("density", "fluctuation_response", "thermochemistry", "equation_of_state"),
        "per-article XML at trc.nist.gov/ThermoML/<DOI>.xml (through 2019)",
        "archive terms; data from journal articles", "unknown (keep local)",
        "Preserve (T, p, composition, phase), expanded uncertainty and confidence level, method, sample "
        "purity and DOI; pure vs mixture; direct vs derived presentation.",
        "IMPLEMENTED: generic parser + pure-liquid density selector (data/sources/thermoml.py)"),
    "D3": SourceInfo(
        "D3", "FreeSolv", "https://github.com/MobleyLab/FreeSolv",
        "evaluated label", ("solvation_free_energy",), "database.json on GitHub",
        "CC BY 4.0", "allowed with attribution",
        "Use expt/d_expt/expt_reference only; never calc*; flag default uncertainties; Ben-Naim standard state.",
        "IMPLEMENTED: loader (data/sources/freesolv.py)"),
    "D4": SourceInfo(
        "D4", "SAMPL challenge repositories", "https://github.com/samplchallenges",
        "experimental label", ("binding_free_energy", "solvation_free_energy"),
        "per-challenge GitHub repositories (experimental_data folders)",
        "per-repository licenses (verify each)", "per repository",
        "Separate experiment from participant submissions and reference calculations; keep challenge-specific "
        "conditions (buffer, pH, temperature) and splits.",
        "PLANNED: host-guest pilot after the solvation cycle is validated"),
    "D5": SourceInfo(
        "D5", "BindingDB", "https://www.bindingdb.org/",
        "experimental label", ("binding_free_energy",), "TSV downloads / web services",
        "BindingDB terms (verify)", "unknown",
        "Separate Kd/Ki/IC50/EC50; constructs, mutations, assay conditions, censoring ('>'), duplicates.",
        "INTERFACE: affinity_to_dg guards; no loader (advanced branch)"),
    "D6": SourceInfo(
        "D6", "BMRB", "https://bmrb.io/",
        "experimental label", ("nmr",), "NMR-STAR entries / API",
        "BMRB terms (open, verify)", "verify",
        "Select explicit observables (3J, RDC, NOE-derived restraints, shifts) with supported forward models; "
        "not a population dataset.",
        "INTERFACE: forward models in observables/nmr.py; loader planned"),
    "D7": SourceInfo(
        "D7", "SASBDB", "https://www.sasbdb.org/",
        "experimental label", ("scattering",), "entry downloads (curves + metadata)",
        "SASBDB terms (verify)", "verify",
        "Keep errors, buffer/background/solvent information; deposited models are not independent labels.",
        "INTERFACE: Debye forward model + nuisance profiling; loader planned"),
    "D8": SourceInfo(
        "D8", "RCSB PDB", "https://www.rcsb.org/",
        "inferred model", ("structure", "binding_free_energy"), "mmCIF files / APIs",
        "CC0 (PDB archive)", "allowed",
        "Initialization and measurement interpretation only; a deposited conformation is not a Boltzmann "
        "probability or an energy label.",
        "USED ONLY FOR STARTING STRUCTURES (no labels)"),
    "D9": SourceInfo(
        "D9", "OMol25 / FAIRChem data", "https://arxiv.org/abs/2505.08762",
        "computational anchor", ("anchor",), "FAIRChem/HF dataset downloads (large)",
        "CC BY 4.0 (dataset card; verify)", "allowed with attribution (verify)",
        "Anchor/replay only; must match the checkpoint's task (omol: wB97M-V/def2-TZVPD), charge/spin; "
        "never mislabeled as experiment.",
        "IMPLEMENTED: anchor loss interface (losses/anchors.py) accepting ASE-DB/extxyz subsets"),
    "D10": SourceInfo(
        "D10", "SPICE", "https://github.com/openmm/spice-dataset",
        "computational anchor", ("anchor",), "HDF5 releases",
        "public domain / per release (verify)", "verify per release",
        "Different level of theory (wB97M-D3BJ/def2-TZVPPD) than OMol25; use only energy differences/forces "
        "and only with an explicit theory-offset model, or not at all for omol anchoring.",
        "NOT USED BY DEFAULT (theory mismatch with omol task)"),
    "D11": SourceInfo(
        "D11", "QCArchive", "https://docs.qcarchive.molssi.org/",
        "infrastructure", ("anchor",), "QCPortal client",
        "per dataset", "per dataset",
        "A repository platform, not one dataset or level of theory; resolve each dataset's specification.",
        "NOT USED (infrastructure option for targeted anchor generation)"),
}

# Families without a suitable primary source in D1-D11 and what is needed.
GAPS = {
    "fluctuation_response": "Cp, kappa_T, alpha_P for pure liquids: ThermoML (D2) contains heat capacity, speed "
                            "of sound and PVT data; kappa_T usually derived (evaluated) - record as EVALUATED.",
    "kinetics_transport": "Self-diffusion and viscosity of pure liquids: ThermoML (D2) transport properties "
                          "(viscosity) and primary NMR/tracer diffusion papers; targeted extraction needed.",
    "electronic_state": "Adiabatic IE/EA: NIST WebBook ion energetics (D1) - parser planned; distinguish "
                        "vertical/adiabatic and 0 K/298 K.",
    "ir_raman_dielectric": "Absolute IR intensities: NIST quantitative IR spectra (D1, gas phase) for a "
                           "handful of molecules; dielectric constants: ThermoML (D2).",
}
