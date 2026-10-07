"""Golden regression test of the NIST parser on the cached pilot pages (+ fail-closed checks).

The golden file stores flag sets, counts and a hash of the values (no values:
NIST SRD 69 redistribution terms). Regenerate deliberately when the parser
changes, and record the change in experiments/vib_v1/DECISIONS.md.
"""

import glob
import hashlib
import json
import os

import pytest

from umaexp.config import DataConfig
from umaexp.data import MolecularIdentity
from umaexp.data.sources import nist_webbook as nw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = json.load(open(os.path.join(ROOT, "tests", "fixtures", "nist_pilot_golden_v3.json"), encoding="utf-8"))
INFO = {"url": "-", "sha256": "-", "retrieval_date": "2026-09-26"}


def _page(name):
    path = os.path.join(ROOT, "data", "raw", "nist_webbook", name)
    if not os.path.exists(path):
        pytest.skip("cached NIST pilot pages not present (regenerate with scripts/build_pilot_dataset.py)")
    return open(path, encoding="utf-8", errors="replace").read()


@pytest.mark.parametrize("sid", sorted(GOLDEN["molecules"]))
def test_pilot_page_parses_exactly_as_recorded(sid):
    g = GOLDEN["molecules"][sid]
    assert GOLDEN["parser"] == nw.PARSER_VERSION, "parser changed: regenerate the golden file deliberately"
    p = nw.parse_vibrational_page(_page(g["page"]))
    assert p["has_table"] == g["has_table"] and p["page_errors"] == g["page_errors"] and p["symmetry"] == g["symmetry"]
    if not p["has_table"]:
        return
    obs = nw.to_observations(p, MolecularIdentity(system_id=sid), INFO, "g")
    modes = {f"{o.labels['species']}{o.labels['mode_number']}":
             sorted(x for x in o.quality_flags if x != "primary_reference_not_resolved") for o in obs}
    assert modes == g["modes"]
    ex = DataConfig().exclude_flags
    assert sum(not any(x in o.quality_flags for x in ex) and o.uncertainty.standard_sd() is not None
               for o in obs) == g["n_usable"]
    vals = "\n".join(f"{o.labels['species']}{o.labels['mode_number']}={o.value}" for o in obs)
    assert hashlib.sha256(vals.encode()).hexdigest() == g["values_sha256"]


def test_known_hard_cases_fail_closed():
    m = GOLDEN["molecules"]
    assert {"free_text_comment", "nonnumeric_value", "conflicting_duplicate_rows"} <= set(m["CH3OH"]["modes"]["a''12"])
    assert sum("overlapped_band" in f for f in m["CH3OH"]["modes"].values()) == 3
    assert m["NH3"]["n_usable"] == 0
    assert sum(r.get("n_usable", 0) for r in m.values()) == 72


def _row(sp, n, sel, rating="A", ir="", ir_phase="gas", comment=""):
    return (f'<tr valign="top"><td align="center">{sp}</td><td align="center">{n}</td><td>m</td>'
            f'<td align="right">{sel}</td><td>&nbsp;<a href="#ESpec{rating}">{rating}</a></td>'
            f'<td align="right">{ir}</td><td>{ir_phase}</td><td align="right"></td><td></td><td>{comment}</td></tr>')


HEADER = ('<tr><th>&nbsp;Sym.&nbsp;</th><th>&nbsp;No&nbsp;</th><th>&nbsp;Approximate&nbsp;</th>'
          '<th colspan="2">Selected Freq.</th><th colspan="2">Infrared</th><th colspan="2">Raman</th><th>Comments</th></tr>')


def _page_with(rows, source=True):
    notes = "".join(f'<tr><td><a id="ESpec{c}">{c}</a></td><td>{t}</td></tr>' for c, t in
                    [("A", "0~1 cm-1 uncertainty"), ("E", "15~30 cm-1 uncertainty"), ("ZZ", "mystery")])
    return (f'<html><body><h3>Symmetry: &nbsp; C<sub>2v</sub> &nbsp; Symmetry Number &#963; = 2</h3>'
            f'<table summary="Vibrational data">{HEADER}{"".join(rows)}</table>'
            f'{"Source: Shimanouchi, 1972" if source else "Source: Jacox, 1998"}'
            f'<table aria-label="List of notes">{notes}</table></body></html>')


def test_synthetic_fail_closed_cases():
    rows = [_row("a1", 1, 3000, ir="3000"),
            _row("a1", 2, 1500, ir="80~300"),                                  # range in a value cell
            _row("b1", 3, 1000, ir="1000"), _row("b1", 3, 900, rating="E", ir="900"),  # conflicting duplicates
            _row("b2", 4, 800, ir="801.5"),                                      # selected not at a gas value
            _row("b2", 5, 700, rating="ZZ", ir="700"),                          # unknown note code
            _row("a2", 6, 600, ir="600", comment="tentative assignment")]       # free text
    p = nw.parse_vibrational_page(_page_with(rows))
    assert p["page_errors"] == []
    obs = {o.labels["mode_number"]: set(o.quality_flags)
           for o in nw.to_observations(p, MolecularIdentity(system_id="S"), INFO, "g")}
    assert not obs[1] - {"primary_reference_not_resolved"}
    assert "nonnumeric_value" in obs[2]
    assert "conflicting_duplicate_rows" in obs[3]
    assert "selected_not_gas_column" in obs[4]
    assert "unknown_note_code" in obs[5]
    assert "free_text_comment" in obs[6]
    assert "table not attributed to Shimanouchi (1972)" in nw.parse_vibrational_page(_page_with(rows, False))["page_errors"]


def test_more_fail_closed_cases_from_red_team_report_3():
    rows = [_row("a1", 1, 3000, ir="", ir_phase="gas"),                    # gas column empty: no gas value at all
            _row("a1", 2, 1500, ir="1500", comment="symmetric level"),     # single tunnelling level, not merged
            _row("b1", 3, "", ir="1000")]                                   # empty selected cell
    p = nw.parse_vibrational_page(_page_with(rows))
    obs = {o.labels["mode_number"]: set(o.quality_flags)
           for o in nw.to_observations(p, MolecularIdentity(system_id="S"), INFO, "g")}
    assert "selected_not_gas_column" in obs[1]
    assert "inversion_doubling" in obs[2]
    assert "nonnumeric_value" in obs[3]
