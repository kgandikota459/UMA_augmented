"""Known-answer tests for the Vib-v1 torsion rule and the pre-registered analysis (no UMA)."""

import numpy as np
import pytest
from ase.build import molecule

from umaexp.observables.internal import torsion_fraction


def _ethane_vectors():
    a = molecule("C2H6")
    x, m, z = a.get_positions(), a.get_masses(), a.get_atomic_numbers()
    c0, c1 = 0, 1
    u = (x[c1] - x[c0]) / np.linalg.norm(x[c1] - x[c0])
    frag = [i for i in range(len(z)) if z[i] == 1 and np.linalg.norm(x[i] - x[c1]) < 1.3]
    rot = np.zeros_like(x)
    for i in frag:  # rigid rotation of one methyl group about the C-C axis
        rot[i] = np.cross(u, x[i] - x[c1])
    stretch = np.zeros_like(x)
    h = frag[0]  # one C-H stretch
    stretch[h] = (x[h] - x[c1]) / np.linalg.norm(x[h] - x[c1])
    from umaexp.observables.internal import _external

    ext = _external(x, m)
    vecs = []
    for d in (rot, stretch):  # like real normal modes: rigid-body motions projected out
        q = (np.sqrt(m)[:, None] * d).ravel()
        q = q - ext @ (ext.T @ q)
        vecs.append(q / np.linalg.norm(q))
    return x, m, z, np.stack(vecs, 1)


def test_torsion_fraction_separates_internal_rotation_from_stretch():
    x, m, z, V = _ethane_vectors()
    f = torsion_fraction(x, m, z, V)
    assert f[0] > 0.9 and f[1] < 0.1


def _rows(delta_ft, n=24, modes=4, invalid_ft=(), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        sid = f"m{k}"
        bias = rng.normal(0, 8)
        for j in range(modes):
            y = 1000.0 + 500 * j
            eb = bias + rng.normal(0, 4)
            base = {"system": sid, "mode": f"x{j}", "experiment": y, "omega_base": y, "delta": -40.0,
                    "delta_coupled": -41.0, "delta_nocoriolis": -39.5, "rating": "A", "crossing_risk": False,
                    "resonance": False, "xh": j == 3, "merged_class": False, "stratum": "HCNO_only", "n_scored": modes}
            rows.append({**base, "model": "B-VPT2", "prediction": y + eb})
            rows.append({**base, "model": "B-scale", "prediction": y + eb + 3.0})
            if sid in invalid_ft:
                continue
            rows.append({**base, "model": "FT", "prediction": y + eb * (1 - delta_ft)})
        if sid in invalid_ft:
            rows.append({"system": sid, "model": "FT", "invalid": "did not converge"})
    return rows


def test_analysis_declares_a_large_improvement_meaningful_and_no_harm():
    from umaexp.vib_v1 import analyse

    out = analyse(_rows(0.6), [])
    assert out["H1_meaningful_improvement"] and out["any_improvement"] and not out["harm"]
    assert out["H1_FT_vs_BVPT2"]["n_dropped_unpaired"] == 0
    assert "ratings_A_to_C" in out["sensitivity"] and "vpt2_delta_coupled" in out["sensitivity"]


def test_analysis_null_effect_is_not_an_improvement():
    from umaexp.vib_v1 import analyse

    out = analyse(_rows(0.0), [])
    assert not out["H1_meaningful_improvement"] and not out["any_improvement"] and not out["harm"]
    assert out["H1_FT_vs_BVPT2"]["delta"] == pytest.approx(0.0, abs=1e-9)


def test_invalid_ft_predictions_are_imputed_with_the_worst_error_not_dropped():
    from umaexp.vib_v1 import analyse

    rows = _rows(0.6, invalid_ft=("m0", "m1", "m2"))
    out = analyse(rows, [])
    assert out["n_invalid"]["FT"] == 3 and out["n_molecules"] == 24
    assert out["H1_FT_vs_BVPT2"]["n_molecules"] == 24  # imputed, not dropped
    assert out["sensitivity"]["complete_case"]["n_molecules"] == 21


def test_failing_on_hard_molecules_never_counts_as_improvement():
    # critic report #3 P3: FT identical to base, but invalid on the 4 hardest molecules
    from umaexp.vib_v1 import analyse

    rows = _rows(0.0, n=20)
    import pandas as pd

    df = pd.DataFrame(rows)
    b = df[df.model == "B-VPT2"]
    hardest = (b.assign(r2=(b.prediction - b.experiment) ** 2).groupby("system").r2.mean()
               .sort_values(ascending=False).index[:4])
    rows = [r for r in rows if not (r["model"] == "FT" and r["system"] in set(hardest))]
    rows += [{"system": s, "model": "FT", "invalid": "did not converge"} for s in hardest]
    out = analyse(rows, [])
    assert out["H1_FT_vs_BVPT2"]["delta"] >= -1e-12 and not out["any_improvement"]
