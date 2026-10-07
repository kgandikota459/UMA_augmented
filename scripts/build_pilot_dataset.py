"""Deterministic extraction of the pilot dataset (needs network access).

Outputs (git-ignored unless the source license allows redistribution):
  data/raw/<source>/...                      raw downloads (+ sha256 in manifest)
  data/processed/<family>.parquet            typed registries
  data/manifests/pilot_manifest.json         versions, hashes, licenses, splits, gaps

  python scripts/build_pilot_dataset.py [--skip-thermoml] [--skip-freesolv]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import sys

import requests
import yaml

from umaexp.data import DatasetManifest, ExperimentRegistry, MolecularIdentity, bemis_murcko_scaffold, group_split
from umaexp.data.sources import freesolv, nist_webbook, thermoml
from umaexp.data.sources.catalog import CATALOG, GAPS

log = logging.getLogger("build_pilot")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def inchikey(smiles: str) -> str:
    from rdkit import Chem

    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def build_vibrational(cfg_path: str, raw_dir: str) -> tuple[ExperimentRegistry, dict]:
    cfg = yaml.safe_load(open(cfg_path))
    reg = ExperimentRegistry()
    report = {}
    for m in cfg["molecules"]:
        try:
            info = nist_webbook.fetch(m["nist_id"], os.path.join(raw_dir, "nist_webbook"))
        except Exception as err:  # network or HTTP failure is recorded, not hidden
            report[m["system_id"]] = f"fetch failed: {err}"
            continue
        page = open(info["path"], encoding="utf-8", errors="replace").read()
        parsed = nist_webbook.parse_vibrational_page(page)
        if not parsed["has_table"]:
            report[m["system_id"]] = "no Shimanouchi table on the WebBook page"
            continue
        parent = m.get("parent_smiles", m["smiles"])
        ik = inchikey(m["smiles"])
        ident = MolecularIdentity(system_id=m["system_id"], smiles=m["smiles"], inchikey=ik,
                                  cas=m["nist_id"][1:], charge=0, spin_multiplicity=1)
        obs = nist_webbook.to_observations(parsed, ident, info, molecule_group=inchikey(parent).split("-")[0],
                                           scaffold_group=bemis_murcko_scaffold(parent))
        for o in obs:
            o.labels["symmetry_number_config"] = m["symmetry_number"]
            reg.add(o)
        report[m["system_id"]] = f"{len(obs)} fundamentals ({parsed['symmetry']}, sigma={parsed['symmetry_number']})"
    return reg, report


def build_freesolv(raw_dir: str) -> tuple[ExperimentRegistry, dict]:
    os.makedirs(os.path.join(raw_dir, "freesolv"), exist_ok=True)
    info = freesolv.fetch(os.path.join(raw_dir, "freesolv", "database.json"))
    db = freesolv.load(info["path"])
    obs = freesolv.to_observations(db, info, elements={"H", "C", "N", "O"}, scaffold=bemis_murcko_scaffold)
    reg = ExperimentRegistry(obs)
    n_default = sum("default_uncertainty_assigned" in o.quality_flags for o in obs)
    return reg, {"records_HCNO": len(obs), "of_total": len(db), "default_uncertainty_assigned": n_default}


def build_thermoml(cfg_path: str, raw_dir: str) -> tuple[ExperimentRegistry, dict]:
    cfg = yaml.safe_load(open(cfg_path))
    comp = cfg["compounds"]
    for ik, c in comp.items():  # verify the configured InChIKeys instead of trusting them
        got = inchikey(c["smiles"])
        if got != ik:
            raise ValueError(f"InChIKey mismatch for {c['system_id']}: config {ik} vs RDKit {got}")
    idx_path = os.path.join(raw_dir, "thermoml", "journal_issue.json")
    os.makedirs(os.path.dirname(idx_path), exist_ok=True)
    r = requests.get(thermoml.BASE + "jsons/journal_issue.json", timeout=120)
    r.raise_for_status()
    open(idx_path, "wb").write(r.content)
    index = r.json()
    cands = []
    for journal, years in index.items():
        for year, issues in years.items():
            for _, arts in issues.items():
                for a in arts:
                    t = a.get("title", "").lower()
                    if any(k in t for k in cfg["title_keywords"]) and any(n in t for n in cfg["name_keywords"]):
                        cands.append((journal, year, a["doi"], a["title"]))
    cands.sort(key=lambda x: (x[1], x[2]))
    reg = ExperimentRegistry()
    report = {"candidates": len(cands), "articles_with_pure_density": []}
    tlo, thi = cfg["temperature_window_K"]
    plo, phi = cfg["pressure_window_bar"]
    for _journal, _year, doi, _title in cands:
        if len(report["articles_with_pure_density"]) >= cfg["max_articles"]:
            break
        url = f"{thermoml.BASE}{doi}.xml"
        try:
            x = requests.get(url, timeout=60)
            x.raise_for_status()
        except Exception:
            continue
        path = os.path.join(raw_dir, "thermoml", doi.replace("/", "_") + ".xml")
        open(path, "wb").write(x.content)
        info = {"url": url, "sha256": hashlib.sha256(x.content).hexdigest(),
                "retrieval_date": dt.date.today().isoformat()}
        try:
            parsed = thermoml.parse(x.content)
        except Exception as err:
            report.setdefault("parse_errors", []).append(f"{doi}: {err}")
            continue
        obs = thermoml.pure_liquid_density(parsed, doi, info, lambda ik: comp.get(ik, {}).get("system_id"))
        keep = [o for o in obs if o.conditions.temperature_k is not None and tlo <= o.conditions.temperature_k <= thi
                and o.conditions.pressure_bar is not None and plo <= o.conditions.pressure_bar <= phi]
        if keep:
            for o in keep:
                reg.add(o)
            report["articles_with_pure_density"].append({"doi": doi, "n": len(keep),
                                                         "compounds": sorted({o.system_ids[0] for o in keep})})
    return reg, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-thermoml", action="store_true")
    ap.add_argument("--skip-freesolv", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    raw = os.path.join(ROOT, "data", "raw")
    proc = os.path.join(ROOT, "data", "processed")
    man = DatasetManifest("uma-exp-pilot", "0.1.0",
                          extraction={"script": "scripts/build_pilot_dataset.py", "python": sys.version.split()[0],
                                      "date": dt.date.today().isoformat(), "seed": a.seed})
    reports, splits = {}, {}
    vib, reports["vibrational"] = build_vibrational(os.path.join(ROOT, "data/pilot/vibrational_pilot.yaml"), raw)
    man.add_registry(vib, os.path.join(proc, "vibrational.parquet"), CATALOG["D1"].name,
                     CATALOG["D1"].license, CATALOG["D1"].redistribution)
    splits["vibrational"] = group_split(list(vib), seed=a.seed)
    if not a.skip_freesolv:
        fs, reports["solvation"] = build_freesolv(raw)
        man.add_registry(fs, os.path.join(proc, "solvation.parquet"), CATALOG["D3"].name,
                         CATALOG["D3"].license, CATALOG["D3"].redistribution)
        splits["solvation"] = group_split(list(fs), seed=a.seed)
    if not a.skip_thermoml:
        tm, reports["density"] = build_thermoml(os.path.join(ROOT, "data/pilot/density_pilot.yaml"), raw)
        man.add_registry(tm, os.path.join(proc, "density.parquet"), CATALOG["D2"].name,
                         CATALOG["D2"].license, CATALOG["D2"].redistribution)
        splits["density"] = group_split(list(tm), seed=a.seed)
    man.notes += [f"{k}: {json.dumps(v, default=str)}" for k, v in reports.items()]
    man.notes += [f"GAP {k}: {v}" for k, v in GAPS.items()]
    os.makedirs(os.path.join(ROOT, "data", "manifests"), exist_ok=True)
    man.save(os.path.join(ROOT, "data", "manifests", "pilot_manifest.json"))
    with open(os.path.join(ROOT, "data", "manifests", "pilot_splits.json"), "w") as f:
        json.dump(splits, f, indent=2)
    print(json.dumps(reports, indent=2, default=str))


if __name__ == "__main__":
    main()
