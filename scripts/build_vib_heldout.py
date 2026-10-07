"""Build Vib-v1's development extension (D_ext) and held-out test set (T), BOTH SEALED.

Implements experiments/vib_v1/PROTOCOL.md section 2 mechanically and fails closed. Nothing
identifying is printed or written inside the repository:

  * sealed material (T and D_ext labels, raw pages, lookup responses, the eligibility table, a
    manifest hashing every sealed file, the secret salt) goes to $UMAEXP_SEALED_DIR (default
    ~/.umaexp_sealed/vib_v1), under random, non-identifying tags;
  * the repository gets experiments/vib_v1/heldout_public.json: counts, salted commitments
    H(salt || content) for the T and D_ext files and for the sealed manifest, the salt's digest,
    software versions and per-file digests of the code tree at build time.

D_ext is unsealed only after the whole protocol is frozen (scripts/unseal_dev_ext.py).

  python scripts/build_vib_heldout.py            # offline: re-parse pilot pages; eligibility steps 1-4 (counts)
  python scripts/build_vib_heldout.py --fetch    # network build; requires the S2 freeze; run once
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import secrets
import shutil
import sys
import time
from collections import Counter

import numpy as np
import yaml

from umaexp.config import DataConfig
from umaexp.data import ExperimentRegistry, MolecularIdentity, bemis_murcko_scaffold
from umaexp.data.curation import usable
from umaexp.data.registry import sha256_file
from umaexp.data.sources import nist_webbook

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTOCOL = os.path.join(ROOT, "experiments", "vib_v1", "PROTOCOL.md")
PUBLIC = os.path.join(ROOT, "experiments", "vib_v1", "heldout_public.json")
RESERVED = os.path.join(ROOT, "experiments", "vib_v1", "reserved_blocks.json")
SEALED = os.environ.get("UMAEXP_SEALED_DIR", os.path.join(os.path.expanduser("~"), ".umaexp_sealed", "vib_v1"))
SEED = 20261006
HCNO = {"H", "C", "N", "O"}
# PROTOCOL 2.4 minimum size (label-only rules, before any model output exists)
MIN_T_MOLECULES = 15
MIN_T_LABEL_USABLE = 130  # >= 100 scored after the model-based rules of 3.2 (~85% retention on D_pilot)
T_FRACTIONS = (0.5, 0.6, 0.7)  # same permutation: T at a larger fraction contains T at a smaller one
# candidates with any real-weight model output before the draw (DECISIONS D-17): forced into D_ext
FORCED_DEV = {"CH3CH2OH"}
MAX_SEARCH_RESULTS = 50
MIN_FREE_BYTES = 1_000_000_000


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def perceive(atoms) -> tuple[str, str, str]:
    """SMILES, InChIKey and Hill formula from a neutral geometry via RDKit bond perception (fails on radicals)."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    xyz = f"{len(atoms)}\n\n" + "\n".join(
        f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.positions))
    mol = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineBonds(mol, charge=0)
    Chem.SanitizeMol(mol)
    if any(a.GetNumRadicalElectrons() or a.GetFormalCharge() for a in mol.GetAtoms()):
        raise ValueError("radical or charge-separated perception")
    return Chem.MolToSmiles(Chem.RemoveHs(mol)), Chem.MolToInchiKey(mol), CalcMolFormula(mol)


def pilot_blocks() -> set[str]:
    from rdkit import Chem

    pilot = yaml.safe_load(open(os.path.join(ROOT, "data", "pilot", "vibrational_pilot.yaml")))
    blocks = set()
    for m in pilot["molecules"]:
        for smi in {m["smiles"], m.get("parent_smiles", m["smiles"])}:
            blocks.add(Chem.MolToInchiKey(Chem.MolFromSmiles(smi)).split("-")[0])
    return blocks


def offline_candidates() -> tuple[list[dict], Counter]:
    """Eligibility steps 1-4 (no network)."""
    from ase.collections import g2

    from umaexp.data.pointgroup import find_point_group

    reasons, cands = Counter(), []
    dev_blocks = pilot_blocks()
    for name in g2.names:
        atoms = g2[name]
        if abs(atoms.get_initial_charges().sum()) > 1e-9 or np.abs(atoms.get_initial_magnetic_moments()).sum() > 0:
            reasons["1_not_neutral_closed_shell"] += 1
            continue
        if len(atoms) < 3:
            reasons["2_fewer_than_3_atoms"] += 1
            continue
        mom = np.sort(atoms.get_moments_of_inertia())
        if mom[0] < 1e-3 * mom[2]:
            reasons["2b_linear"] += 1
            continue
        try:
            smi, ik, formula = perceive(atoms)
        except Exception:  # noqa: BLE001 - recorded as a reason, never printed
            reasons["3_bond_perception_failed_or_radical"] += 1
            continue
        if ik.split("-")[0] in dev_blocks:
            reasons["4_in_pilot_list"] += 1
            continue
        pg = find_point_group(atoms.positions, atoms.get_masses(), atoms.get_atomic_numbers(), tol=0.02)
        cands.append({"g2_name": name, "smiles": smi, "inchikey": ik, "formula": formula,
                      "elements": sorted(set(atoms.get_chemical_symbols())), "n_atoms": len(atoms),
                      "g2_point_group": pg.name})
    return cands, reasons


def build_pilot_dev() -> str:
    """Re-parse the cached pilot WebBook pages (development data) with the current parser; no network."""
    import glob

    from rdkit import Chem

    pilot = yaml.safe_load(open(os.path.join(ROOT, "data", "pilot", "vibrational_pilot.yaml")))
    reg = ExperimentRegistry()
    for m in pilot["molecules"]:
        pages = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "nist_webbook", f"{m['nist_id']}_mask800_*.html")))
        if not pages:
            continue
        raw = open(pages[-1], "rb").read()
        parsed = nist_webbook.parse_vibrational_page(raw.decode("utf-8", errors="replace"))
        if not parsed["has_table"] or parsed["page_errors"]:
            continue
        parent = m.get("parent_smiles", m["smiles"])
        ik = Chem.MolToInchiKey(Chem.MolFromSmiles(m["smiles"]))
        ident = MolecularIdentity(system_id=m["system_id"], smiles=m["smiles"], inchikey=ik, cas=m["nist_id"][1:],
                                  charge=0, spin_multiplicity=1)
        info = {"url": f"{nist_webbook.CBOOK}?ID={m['nist_id']}&Mask=800", "sha256": hashlib.sha256(raw).hexdigest(),
                "retrieval_date": re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(pages[-1])).group(1)}
        obs = nist_webbook.to_observations(parsed, ident, info,
                                           molecule_group=Chem.MolToInchiKey(Chem.MolFromSmiles(parent)).split("-")[0],
                                           scaffold_group=bemis_murcko_scaffold(parent))
        for o in obs:
            o.labels.update(smiles=m["smiles"], symmetry_number_config=m["symmetry_number"])
            reg.add(o)
    out = os.path.join(ROOT, "data", "processed", "vib_v1_dev_pilot.parquet")
    reg.to_parquet(out)
    n_use = sum(usable(o, DataConfig().exclude_flags) for o in reg)
    print(f"dev pilot: {len(reg)} fundamentals ({n_use} usable under the frozen rule) -> {os.path.relpath(out, ROOT)}")
    return out


class FetchError(RuntimeError):
    pass


def _get(session, url: str, archive: str | None = None, tries: int = 4) -> str:
    """GET with retries and exponential backoff; any remaining failure aborts the whole build."""
    err = "unknown"
    for k in range(tries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200 and "<html" in r.text.lower():
                if archive:
                    with open(archive, "w", encoding="utf-8") as f:
                        f.write(r.text)
                time.sleep(1.0)
                return r.text
            err = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001 - type only
            err = type(e).__name__
        time.sleep(2.0 * 2**k)
    raise FetchError(err)


def _page_key(text: str) -> str | None:
    m = re.search(r"IUPAC Standard InChIKey:</strong>\s*<span[^>]*>([A-Z-]{27})<", text)
    return m.group(1) if m else None


def _title(text: str) -> str:
    m = re.search(r"<title>(.*?)</title>", text, re.S)
    return m.group(1).strip() if m else ""


def _page_formula(text: str) -> str | None:
    m = re.search(r"Formula</a>:</strong>\s*(.*?)</li>", text, re.S)
    return re.sub(r"<[^>]+>|\s+", "", m.group(1)) if m else None


def _own_vib_link(text: str) -> set[str]:
    return set(re.findall(r'cbook\.cgi\?ID=([A-Z]\w+)&amp;Units=SI&amp;Mask=800', text))


def nist_resolve(session, inchikey: str, archive_dir: str | None = None, tag: str = "",
                 formula: str | None = None) -> tuple[str | None, str]:
    """WebBook species ID whose page carries exactly this standard InChIKey (and formula), or (None, reason).

    Search-result pages are resolved by opening every listed species and keeping the records with the
    exact key, their own vibrational-data link and the candidate's formula; exactly one must remain
    (the WebBook contains duplicate and erroneous records with copied keys, e.g. for ethane and D2O).
    Raises FetchError (aborting the build) on any page that is neither a recognised result page nor a
    species page, and on search results longer than MAX_SEARCH_RESULTS.
    """

    def arch(name):
        return os.path.join(archive_dir, f"{tag}_{name}.html") if archive_dir else None

    text = _get(session, f"{nist_webbook.CBOOK}?InChI={inchikey}&Units=SI", arch("lookup"))
    title = _title(text)
    if "No Matching Species" in title:
        return None, "5a_not_in_webbook"
    if "Search Results" in title:  # several species share the key prefix: open each, keep the exact match
        ids = sorted(set(re.findall(r'cbook\.cgi\?ID=([A-Z]\w+)&amp;Units=SI"', text)))
        if not ids or len(ids) > MAX_SEARCH_RESULTS:
            raise FetchError(f"search results page with {len(ids)} entries")
        exact = []
        for i, sid in enumerate(ids):
            t = _get(session, f"{nist_webbook.CBOOK}?ID={sid}&Units=SI", arch(f"candidate{i}"))
            if _page_key(t) is None:
                raise FetchError("unrecognised species page")
            if _page_key(t) == inchikey and sid in _own_vib_link(t) and (formula is None or _page_formula(t) == formula):
                exact.append((sid, t))
        if len(exact) != 1:
            return None, "5b_no_unique_exact_species"
        sid, text = exact[0]
    else:
        key = _page_key(text)
        if key is None:
            raise FetchError("unrecognised lookup page")
        if key != inchikey:
            return None, "5c_page_inchikey_mismatch"
        if formula is not None and _page_formula(text) != formula:
            return None, "5e_formula_mismatch"
        own = sorted(set(re.findall(r'cbook\.cgi\?ID=([A-Z]\w+)&amp;Units=SI&amp;Mask=800', text)))
        if len(own) != 1:
            return None, "5d_no_unique_vibrational_link"
        sid = own[0]
    if f"ID={sid}&amp;Units=SI&amp;Mask=800" not in text:
        return None, "5d_no_unique_vibrational_link"
    return sid, "ok"


def _versions() -> dict:
    import ase
    import rdkit
    import requests

    return {"python": platform.python_version(), "numpy": np.__version__, "rdkit": rdkit.__version__,
            "ase": ase.__version__, "requests": requests.__version__, "parser": nist_webbook.PARSER_VERSION}


def _code_files() -> dict:
    out = {}
    for d in ("src", "scripts", "configs"):
        for base, _, names in os.walk(os.path.join(ROOT, d)):
            if "__pycache__" not in base:
                for n in names:
                    if not n.endswith(".pyc"):
                        p = os.path.join(base, n)
                        out[os.path.relpath(p, ROOT)] = sha256_file(p)
    return dict(sorted(out.items()))


def _label_usable(o, exclude) -> bool:
    """Label-only scoring rules (source flags + torsion text), used for the size rule before any model output."""
    return usable(o, exclude) and "torsion" not in (o.labels.get("approximate_type") or "").lower()


def _process(session, c: dict, staging: str, exclude) -> tuple[list, str | None]:
    tag = c["tag"]
    nist_id, why = nist_resolve(session, c["inchikey"], os.path.join(staging, "pages"), tag, formula=c["formula"])
    if nist_id is None:
        return [], why
    text = _get(session, f"{nist_webbook.CBOOK}?ID={nist_id}&Mask=800",
                os.path.join(staging, "pages", f"{tag}_mask800.html"))
    key = _page_key(text)
    if key is None:  # neither a species page nor a result page (PROTOCOL 2.3): abort, never a silent exclusion
        raise FetchError("unrecognised Mask=800 page")
    if key != c["inchikey"]:
        return [], "6f_mask800_inchikey_mismatch"
    parsed = nist_webbook.parse_vibrational_page(text)
    if not parsed["has_table"]:
        return [], "6a_no_shimanouchi_table"
    if parsed["page_errors"]:
        return [], "6b_page_errors"
    if parsed["symmetry"] != c["g2_point_group"]:
        return [], "6c_point_group_mismatch"
    ident = MolecularIdentity(system_id=c["g2_name"], smiles=c["smiles"], inchikey=c["inchikey"],
                              cas=nist_id[1:] if nist_id.startswith("C") else None, charge=0, spin_multiplicity=1)
    info = {"url": f"{nist_webbook.CBOOK}?ID={nist_id}&Mask=800", "sha256": _h(text),
            "retrieval_date": dt.date.today().isoformat()}
    obs = nist_webbook.to_observations(parsed, ident, info, molecule_group=c["inchikey"].split("-")[0],
                                       scaffold_group=bemis_murcko_scaffold(c["smiles"]))
    if sum(nist_webbook.species_degeneracy(o.labels["species"]) for o in obs) != 3 * c["n_atoms"] - 6:
        return [], "6d_incomplete_table"
    if sum(_label_usable(o, exclude) for o in obs) == 0:
        return [], "6e_no_usable_fundamental"
    for o in obs:
        o.labels.update(smiles=c["smiles"], g2_name=c["g2_name"], initial_geometry="ase.collections.g2")
    c.update(nist_id=nist_id, n_label_usable=sum(_label_usable(o, exclude) for o in obs))
    return obs, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="network build (requires the S2 freeze; run once)")
    a = ap.parse_args()
    build_pilot_dev()
    cands, reasons = offline_candidates()
    print(f"offline eligibility: {len(cands)} candidates after steps 1-4; excluded: {dict(sorted(reasons.items()))}")
    if not a.fetch:
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from freeze_vib_v1 import verify_s2

    verify_s2()  # section 2 and the data code must be frozen, and unchanged since
    reserved = set(json.load(open(RESERVED))["inchikey_blocks"])
    if {c["inchikey"][:14] for c in cands} != reserved:  # guards against RDKit/ASE/pilot-list drift
        raise SystemExit("offline candidate set differs from the frozen reserved list; aborting")
    final = os.path.join(SEALED, "build")
    staging = os.path.join(SEALED, "staging")
    if os.path.exists(final) or os.path.exists(PUBLIC):
        raise SystemExit("a sealed build already exists; refusing to rebuild (PROTOCOL.md 2.4)")
    if shutil.disk_usage(os.path.expanduser("~")).free < MIN_FREE_BYTES:
        raise SystemExit("less than 1 GB free disk; refusing to start")
    import requests

    exclude = DataConfig().exclude_flags
    shutil.rmtree(staging, ignore_errors=True)  # an unpublished partial build: nothing was seen, safe to redo
    os.makedirs(os.path.join(staging, "pages"), exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "umaexp research data curation"
    eligible = []
    try:
        for c in cands:
            c["tag"] = secrets.token_hex(6)  # random, non-identifying
            obs, why = _process(session, c, staging, exclude)
            if why:
                reasons[why] += 1
                c["reason"] = why
                continue
            c["_obs"] = obs
            eligible.append(c)
    except Exception as e:  # noqa: BLE001 - any failure aborts; only the exception type is printed
        shutil.rmtree(staging, ignore_errors=True)
        raise SystemExit(f"build aborted ({type(e).__name__}: network or page failure); nothing published") from None

    try:  # split and sealing: any failure aborts with the exception type only (red-team report #4)
        # split (PROTOCOL 2.4): forced-dev first; groups = InChIKey first blocks; strata in fixed order
        forced = [c for c in eligible if c["g2_name"] in FORCED_DEV]
        pool = [c for c in eligible if c["g2_name"] not in FORCED_DEV]
        by_block: dict[str, list[dict]] = {}
        for c in pool:
            by_block.setdefault(c["inchikey"].split("-")[0], []).append(c)
        below_minimum = False
        for k, frac in enumerate(T_FRACTIONS):
            rng = np.random.default_rng(SEED)
            test, ext, strata = [], list(forced), {}
            for stratum in ("HCNO_only", "other_elements"):
                blocks = sorted(b for b, cs in by_block.items()
                                if all((set(c["elements"]) <= HCNO) == (stratum == "HCNO_only") for c in cs))
                perm = rng.permutation(len(blocks))
                n_t = math.ceil(len(blocks) * frac)
                test += [c for i in perm[:n_t] for c in by_block[blocks[i]]]
                ext += [c for i in perm[n_t:] for c in by_block[blocks[i]]]
                strata[stratum] = {"eligible_groups": len(blocks), "test_groups": n_t}
            n_usable_t = sum(c["n_label_usable"] for c in test)
            if len(test) >= MIN_T_MOLECULES and n_usable_t >= MIN_T_LABEL_USABLE:
                break
            if k == len(T_FRACTIONS) - 1:
                below_minimum = True  # pre-registered fallback: proceed at the last fraction, reported as underpowered
        assert not ({c["inchikey"][:14] for c in test} & ({c["inchikey"][:14] for c in ext} | pilot_blocks()))

        # write everything into staging, hash it, publish the commitments, then rename staging into place
        salt = secrets.token_hex(32)
        ext_reg = ExperimentRegistry([o for c in ext for o in c["_obs"]])
        test_reg = ExperimentRegistry([o for c in test for o in c["_obs"]])
        ext_reg.to_parquet(os.path.join(staging, "dev_ext.parquet"))
        test_reg.to_parquet(os.path.join(staging, "test.parquet"))

        def mols(cs):
            return [{k: c[k] for k in ("g2_name", "nist_id", "smiles", "inchikey", "tag", "n_label_usable")}
                    for c in sorted(cs, key=lambda c: c["inchikey"])]

        json.dump(mols(ext), open(os.path.join(staging, "dev_ext_molecules.json"), "w"), indent=1)
        json.dump(mols(test), open(os.path.join(staging, "test_molecules.json"), "w"), indent=1)
        json.dump([{k: v for k, v in c.items() if k != "_obs"} for c in cands],
                  open(os.path.join(staging, "eligibility.json"), "w"), indent=1)
        manifest = {}
        for base, _, names in os.walk(staging):
            for n in names:
                p = os.path.join(base, n)
                manifest[os.path.relpath(p, staging)] = sha256_file(p)
        json.dump(dict(sorted(manifest.items())), open(os.path.join(staging, "MANIFEST.json"), "w"), indent=1)
        with open(os.path.join(staging, "SALT"), "w") as f:
            f.write(salt)

        def commit(name):
            return _h(salt + open(os.path.join(staging, name), "rb").read().hex())

        public = {
            "built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "protocol_sha256_at_build": sha256_file(PROTOCOL),
            "versions": _versions(),
            "code_files_at_build": _code_files(),
            "pool": {"source": "ase.collections.g2", "n_candidates_offline": len(cands)},
            "seed": SEED, "t_fraction_used": frac, "below_minimum_size": below_minimum, "exclude_flags": exclude,
            "excluded_by_reason": dict(sorted(reasons.items())),
            "n_eligible": len(eligible), "n_forced_dev": len(forced), "strata": strata,
            "dev_ext": {"n_molecules": len(ext), "n_observations": len(ext_reg),
                        "commit_parquet": commit("dev_ext.parquet"), "commit_molecules": commit("dev_ext_molecules.json")},
            "test": {"n_molecules": len(test), "n_observations": len(test_reg), "n_label_usable": n_usable_t,
                     "commit_parquet": commit("test.parquet"), "commit_molecules": commit("test_molecules.json")},
            "commit_manifest": commit("MANIFEST.json"),
            "salt_sha256": _h(salt),
        }
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(staging, ignore_errors=True)
        raise SystemExit(f"build aborted during split/sealing ({type(e).__name__}); nothing published") from None
    tmp = PUBLIC + ".tmp"
    with open(tmp, "w") as f:
        json.dump(public, f, indent=2)
    os.replace(tmp, PUBLIC)  # commitments published (atomic)
    os.rename(staging, final)  # sealed data in place (atomic)
    with open(os.path.join(final, "COMPLETE"), "w") as f:
        f.write(public["built"])
    print(json.dumps({k: public[k] for k in ("excluded_by_reason", "n_eligible", "n_forced_dev", "strata",
                                             "t_fraction_used", "below_minimum_size")}, indent=2))
    print(f"D_ext: {len(ext)} molecules, T: {len(test)} molecules ({n_usable_t} label-usable modes); "
          f"both SEALED outside the repository (not displayed)")


if __name__ == "__main__":
    main()
