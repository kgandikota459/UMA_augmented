import datetime as dt
import os

import pytest

from umaexp.data import (
    Conditions,
    DatasetManifest,
    ExperimentRegistry,
    MolecularIdentity,
    Observation,
    Phase,
    Provenance,
    SplitGroups,
    Uncertainty,
    UncertaintyType,
    ValueKind,
    assert_no_leakage,
    bemis_murcko_scaffold,
    group_split,
)
from umaexp.data.sources import nist_webbook, thermoml

HERE = os.path.dirname(__file__)


def _obs(i, mol, pub=None, kind=ValueKind.MEASURED, **kw):
    return Observation(
        observation_id=f"o{i}", system_ids=[mol], family="density", quantity="mass_density", definition="d",
        value=1.0, units="g/cm^3", uncertainty=Uncertainty(kind=UncertaintyType.STANDARD, value=0.001),
        value_kind=kind, conditions=Conditions(temperature_k=298.15, phase=Phase.LIQUID),
        provenance=Provenance(source_name="test"), groups=SplitGroups(molecule_group=mol, publication_group=pub),
        **kw)


def test_schema_guards():
    with pytest.raises(ValueError):
        _obs(0, "m", kind=ValueKind.COMPUTED)  # computed needs a method
    ok = _obs(1, "m", kind=ValueKind.COMPUTED, computed_method="wB97M-V/def2-TZVPD")
    assert not ok.is_experimental
    u = Uncertainty(kind=UncertaintyType.EXPANDED, value=0.2, coverage_factor=2.0)
    assert u.standard_sd() == pytest.approx(0.1)
    with pytest.raises(ValueError):
        Observation(**{**_obs(2, "m").model_dump(), "stoichiometry": [1.0, -1.0]})


def test_registry_parquet_roundtrip_and_manifest(tmp_path):
    reg = ExperimentRegistry([_obs(i, f"mol{i % 3}") for i in range(6)])
    man = DatasetManifest("pilot", "0.1")
    path = str(tmp_path / "density.parquet")
    man.add_registry(reg, path, source="test", license="CC0", redistribution="allowed")
    back = ExperimentRegistry.from_parquet(path)
    assert [o.model_dump() for o in back] == [o.model_dump() for o in reg]
    man.save(str(tmp_path / "manifest.json"))
    assert DatasetManifest.load(str(tmp_path / "manifest.json")).verify() == []
    with open(path, "ab") as f:
        f.write(b"x")
    assert man.verify() == [path]


def test_group_split_has_no_leakage():
    # mol0 and mol1 share a publication -> must land together
    obs = [_obs(0, "mol0", pub="doiA"), _obs(1, "mol1", pub="doiA"), _obs(2, "mol0"),
           _obs(3, "mol2"), _obs(4, "mol3"), _obs(5, "mol4"), _obs(6, "mol5"), _obs(7, "mol6")]
    split = group_split(obs, seed=3)
    assert_no_leakage(split, obs)
    where = {i: k for k, ids in split.items() for i in ids}
    assert where["o0"] == where["o1"] == where["o2"]
    assert sum(len(v) for v in split.values()) == len(obs)


def test_scaffold():
    assert bemis_murcko_scaffold("CCc1ccccc1") == "c1ccccc1"
    assert bemis_murcko_scaffold("CCO") == "acyclic"


def test_nist_parser_on_synthetic_layout():
    page = open(os.path.join(HERE, "fixtures", "synthetic_webbook_vib.html")).read()
    parsed = nist_webbook.parse_vibrational_page(page)
    assert parsed["has_table"] and parsed["symmetry_number"] == 3
    assert parsed["notes"]["A"].startswith("0~1")
    ident = MolecularIdentity(system_id="SYN", cas="0-00-0")
    info = {"url": "file://synthetic", "sha256": "0", "retrieval_date": dt.date.today().isoformat()}
    obs = nist_webbook.to_observations(parsed, ident, info, molecule_group="SYN")
    assert len(obs) == 5  # doubled a1 level merged
    a1, e, f2, rock, bend = obs
    assert "inversion_doubling" in a1.quality_flags and a1.value == 3000.0
    assert a1.uncertainty.standard_sd() == pytest.approx(0.5)
    # inactive in IR only (Raman observed): not silent
    assert e.labels["degeneracy"] == 2 and "silent_mode" not in e.quality_flags
    assert "observed_via_coriolis_interaction" in e.quality_flags
    # inactive in both columns, value from an overtone (OC code in the Comments column)
    assert {"silent_mode", "not_directly_observed"} <= set(rock.quality_flags)
    # FR code in the Comments column
    assert "fermi_resonance" in bend.quality_flags and "silent_mode" not in bend.quality_flags
    assert f2.labels["degeneracy"] == 3 and "condensed_phase_only" in f2.quality_flags
    assert all(o.value_kind == ValueKind.EVALUATED for o in obs)
    assert all(o.provenance.redistribution.value == "restricted" for o in obs)


def test_nist_parser_restores_dropped_double_primes():
    rows = "".join(
        f'<tr valign="top"><td align="center">{sp}</td><td align="center">{n}</td><td>m</td>'
        f'<td align="right">{v}</td><td>&nbsp;<a href="#ESpecA">A</a></td><td align="right">{v}</td><td>gas</td>'
        f'<td align="right"></td><td></td><td></td></tr>'
        for sp, n, v in [("a'", 1, 3000), ("a'", 2, 1500), ("a", 3, 900)])
    page = (f'<html><body><h3>Symmetry: &nbsp; C<sub>s</sub> &nbsp; Symmetry Number &#963; = 1</h3>'
            f'<table summary="Vibrational data">{rows}</table>'
            f'<table aria-label="List of notes"><tr><td><a id="ESpecA">A</a></td><td>0~1 cm-1 uncertainty</td></tr>'
            f'</table></body></html>')
    parsed = nist_webbook.parse_vibrational_page(page)
    assert [r["species"] for r in parsed["rows"]] == ["a'", "a'", "a''"]
    assert parsed["rows"][2]["species_raw"] == "a"


def test_thermoml_parser_on_synthetic_layout():
    raw = open(os.path.join(HERE, "fixtures", "synthetic_thermoml.xml"), "rb").read()
    parsed = thermoml.parse(raw)
    info = {"url": "file://synthetic", "sha256": "0", "retrieval_date": dt.date.today().isoformat()}
    obs = thermoml.pure_liquid_density(parsed, "10.0000/synthetic.1", info, lambda ik: "methanol")
    assert [o.value for o in obs] == [0.8, 0.79]
    assert obs[0].conditions.temperature_k == 298.15 and obs[0].conditions.pressure_bar == pytest.approx(1.01)
    assert obs[0].uncertainty.standard_sd() == pytest.approx(0.0001)
    assert obs[0].value_kind == ValueKind.MEASURED
