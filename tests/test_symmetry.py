"""Known-answer tests for symmetry-constrained mode assignment (toy potential; no UMA).

The decomposition of the vibrational representation into irreps depends only on
the geometry and the point group, not on the potential, so a pair potential
evaluated at the (symmetric) G2/ASE geometries gives the true per-species mode
counts. These must agree with the NIST/Shimanouchi species of each development
molecule.
"""

import os
from collections import defaultdict

import numpy as np
import pytest
import torch
from ase.build import molecule
from toy_helpers import PairMorse

from umaexp.observables.symmetry import (
    GROUP_ORDER,
    assign_by_symmetry,
    find_point_group,
    parse_species,
    symmetry_mode_groups,
)
from umaexp.observables.vibrational import modes_from_hessian
from umaexp.potential.derivatives import hessian
from umaexp.systems import System

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = os.path.join(ROOT, "data", "processed", "vib_v1_dev_pilot.parquet")
ASE_NAME = {"H2S": "SH2", "D2O": "H2O"}


def _toy_modes(name, deuterate=False):
    a = molecule(ASE_NAME.get(name, name))
    hs = [i for i, z in enumerate(a.get_atomic_numbers()) if z == 1]
    s = System.from_ase(a, system_id=name, isotopes={i: 2 for i in hs} if deuterate else None)
    if deuterate:
        assert abs(s.masses[hs[0]] - 2.014) < 1e-3
    x = torch.tensor(a.get_positions())
    H = hessian(PairMorse(1.0, 1.2, 1.3), s, x, create_graph=False).detach()
    nm = modes_from_hessian(H, x, s.mass_tensor(dtype=torch.float64))
    pg = find_point_group(a.get_positions(), np.asarray(s.masses, float), np.asarray(s.atomic_numbers))
    groups = symmetry_mode_groups(pg, nm.eigenvalues.numpy(), nm.modes.numpy(), nm.wavenumbers.detach().numpy())
    return pg, groups, nm


def test_parse_species_reads_only_convention_free_features():
    assert parse_species("a1g") == {"letter": "a", "dim": 1, "sub": 1, "parity": "g", "primes": 0, "sign": None}
    assert parse_species("a''")["primes"] == 2 and parse_species("a'")["primes"] == 1
    assert parse_species("e2u")["dim"] == 2 and parse_species("f2")["dim"] == 3
    assert parse_species("σg+")["letter"] == "sigma" and parse_species("πu")["dim"] == 2
    with pytest.raises(ValueError):
        parse_species("x1")


@pytest.mark.parametrize("name,order", [("H2O", 4), ("NH3", 6), ("CH4", 24), ("C2H4", 8), ("C2H6", 12),
                                        ("C6H6", 24), ("CH3OH", 2), ("HCOOH", 2), ("H2CO", 4), ("CH3Cl", 6),
                                        ("CH3CN", 6), ("H2S", 4)])
def test_point_group_orders_of_development_geometries(name, order):
    a = molecule(ASE_NAME.get(name, name))
    pg = find_point_group(a.get_positions(), a.get_masses(), a.get_atomic_numbers())
    assert pg.order == order


def _dev_tables():
    from umaexp.data import ExperimentRegistry

    if not os.path.exists(DEV):
        pytest.skip("development registry not built (scripts/build_vib_heldout.py)")
    t = defaultdict(list)
    for o in ExperimentRegistry.from_parquet(DEV):
        t[o.system_ids[0]].append(o)
    return t


@pytest.mark.parametrize("name", ["H2O", "D2O", "NH3", "CH4", "H2CO", "HCN", "C2H2", "C2H4", "C2H6", "CH3OH",
                                  "H2S", "CH3Cl", "HCOOH", "C6H6", "CH3CN"])
def test_symmetry_assignment_reproduces_every_tabulated_species(name):
    obs = sorted(_dev_tables()[name], key=lambda o: o.labels["mode_number"])
    pg, groups, nm = _toy_modes(name, deuterate=(name == "D2O"))
    pg_label = obs[0].labels["point_group"]
    if pg_label in GROUP_ORDER:
        assert pg.order == GROUP_ORDER[pg_label]
    res = assign_by_symmetry(pg, groups, [o.labels["species"] for o in obs], [o.labels["mode_number"] for o in obs],
                             [o.value for o in obs], expected_point_group=pg_label)
    assert res.point_group_ok and not res.excluded_exp, res.notes
    assert sorted(e for e, _ in res.matches) == list(range(len(obs)))  # every tabulated mode matched once
    used = [i for _, idx in res.matches for i in idx]
    assert len(used) == len(set(used)) == nm.modes.shape[1]  # every computed mode used once
    for e, idx in res.matches:
        assert len(idx) == parse_species(obs[e].labels["species"])["dim"]
    assert not res.merged_classes  # axis conventions resolved from the species counts (no value-based merging)


def test_near_degenerate_modes_of_different_irreps_are_separated():
    pg, groups, nm = _toy_modes("C2H6")
    nd = [g for g in groups if len(g.indices) == 1]
    g1 = next(g for g in nd if g.characters[pg.idx_inversion] > 0)   # a g mode
    g2 = next(g for g in nd if g.characters[pg.idx_inversion] < 0)   # a u mode
    lam = nm.eigenvalues.numpy().copy()
    V = nm.modes.numpy().copy()
    i, j = g1.indices[0], g2.indices[0]
    lam[i] = lam[j] = 0.5 * (lam[i] + lam[j])  # exact accidental degeneracy ...
    c = np.cos(0.6)
    vi, vj = V[:, i].copy(), V[:, j].copy()
    V[:, i], V[:, j] = c * vi + np.sin(0.6) * vj, -np.sin(0.6) * vi + c * vj  # ... with mixed eigenvectors
    nu = np.sign(lam) * np.sqrt(np.abs(lam)) * 521.4709
    out = symmetry_mode_groups(pg, lam, V, nu)
    pure = [g for g in out if set(g.indices) <= {i, j}]
    assert len(pure) == 2 and all(len(g.indices) == 1 and g.irreducible for g in pure)
    assert sorted(round(g.characters[pg.idx_inversion]) for g in pure) == [-1, 1]


def test_assignment_never_uses_measured_values_outside_merged_classes():
    obs = sorted(_dev_tables()["CH4"], key=lambda o: o.labels["mode_number"])
    pg, groups, _ = _toy_modes("CH4")
    sp, num = [o.labels["species"] for o in obs], [o.labels["mode_number"] for o in obs]
    ref = assign_by_symmetry(pg, groups, sp, num, [o.value for o in obs], "Td")
    rng = np.random.default_rng(0)
    for _ in range(5):
        shuffled = list(rng.uniform(100, 4000, len(obs)))
        assert assign_by_symmetry(pg, groups, sp, num, shuffled, "Td").matches == ref.matches


def test_missing_species_is_resolved_by_counts_and_reported_as_orphan():
    # an incomplete table can still be matched by counts, which is why the builder requires complete tables
    obs = sorted(_dev_tables()["H2CO"], key=lambda o: o.labels["mode_number"])
    keep = [o for o in obs if o.labels["species"] != "b2"]
    pg, groups, _ = _toy_modes("H2CO")
    res = assign_by_symmetry(pg, groups, [o.labels["species"] for o in keep], [o.labels["mode_number"] for o in keep],
                             [o.value for o in keep], "C2v")
    assert not res.excluded_exp and any("absent from the table" in n for n in res.notes)


def test_inconsistent_counts_exclude_the_class_instead_of_value_matching():
    obs = sorted(_dev_tables()["H2CO"], key=lambda o: o.labels["mode_number"])
    sp = [o.labels["species"] for o in obs] + ["b2"]  # one tabulated mode too many
    pg, groups, _ = _toy_modes("H2CO")
    res = assign_by_symmetry(pg, groups, sp, [o.labels["mode_number"] for o in obs] + [99],
                             [o.value for o in obs] + [500.0], "C2v")
    b = [i for i, x in enumerate(sp) if x.startswith("b")]
    assert set(b) <= set(res.excluded_exp)
    assert all(sp[e] == "a1" for e, _ in res.matches)


def test_wrong_point_group_is_flagged_not_assigned():
    obs = sorted(_dev_tables()["H2O"], key=lambda o: o.labels["mode_number"])
    pg, groups, _ = _toy_modes("H2O")
    res = assign_by_symmetry(pg, groups, [o.labels["species"] for o in obs], [o.labels["mode_number"] for o in obs],
                             [o.value for o in obs], expected_point_group="C3v")
    assert not res.point_group_ok and not res.matches and len(res.excluded_exp) == len(obs)


def _toy_from_xyz(z, x, name="syn"):
    from umaexp.observables.vibrational import modes_from_hessian

    s = System(system_id=name, atomic_numbers=list(z), masses=[{1: 1.008, 6: 12.0, 7: 14.003, 8: 15.995}[k] for k in z],
               charge=0, spin=1)
    xt = torch.tensor(np.asarray(x, float))
    H = hessian(PairMorse(1.0, 1.2, 1.3), s, xt, create_graph=False).detach()
    nm = modes_from_hessian(H, xt, s.mass_tensor(dtype=torch.float64))
    pg = find_point_group(np.asarray(x, float), np.asarray(s.masses, float), np.asarray(z))
    groups = symmetry_mode_groups(pg, nm.eigenvalues.numpy(), nm.modes.numpy(), nm.wavenumbers.detach().numpy())
    return pg, groups


def _synthetic():
    a, c = 1.1, 0.55  # flattened tetrahedron (c != a / sqrt 2): D2d
    d2d = ([6, 1, 1, 1, 1], [[0, 0, 0], [a, 0, c], [-a, 0, c], [0, a, -c], [0, -a, -c]])
    c2h = ([7, 7, 1, 1], [[0.62, 0, 0], [-0.62, 0, 0], [1.05, 0.9, 0], [-1.05, -0.9, 0]])
    t = np.deg2rad(22.5)
    hs = [[0.92 * np.cos(p + s * t), 0.92 * np.sin(p + s * t), s * 1.2] for s in (1, -1) for p in (0, np.pi)]
    d2 = ([6, 6, 1, 1, 1, 1], [[0, 0, 0.67], [0, 0, -0.67]] + hs)
    eth = []
    for s, phase in ((1, 0.0), (-1, np.deg2rad(37.0))):  # generic torsion: D3
        eth += [[1.02 * np.cos(phase + k * 2 * np.pi / 3), 1.02 * np.sin(phase + k * 2 * np.pi / 3), s * 1.15]
                for k in range(3)]
    d3 = ([6, 6, 1, 1, 1, 1, 1, 1], [[0, 0, 0.77], [0, 0, -0.77]] + eth)
    return {"D2d": d2d, "C2h": c2h, "D2": d2, "D3": d3}


@pytest.mark.parametrize("name", ["D2d", "C2h", "D2", "D3"])
def test_point_group_names_distinguish_groups_of_equal_order(name):
    z, x = _synthetic()[name]
    assert find_point_group(np.asarray(x, float), np.ones(len(z)) * np.asarray(z), np.asarray(z)).name == name


def test_d2d_uses_the_s4_axis_and_assigns_every_species():
    # flattened XY4 (Td -> D2d): Gamma_vib = 2 A1 + B1 + 2 B2 + 2 E
    z, x = _synthetic()["D2d"]
    pg, groups = _toy_from_xyz(z, x)
    assert pg.name == "D2d" and pg.unique_principal and not pg.d2_type
    sp = ["a1", "a1", "b1", "b2", "b2", "e", "e"]
    res = assign_by_symmetry(pg, groups, sp, list(range(1, 8)), [3000, 1500, 1300, 3100, 1400, 3050, 1200], "D2d")
    assert res.point_group_ok and not res.excluded_exp and not res.merged_classes, res.notes
    assert sorted(e for e, _ in res.matches) == list(range(7))


def test_td_t2_species_are_symmetric_under_sigma_d():
    pg, groups, _ = _toy_modes("CH4")
    assert pg.name == "Td" and pg.cubic
    t_groups = [g for g in groups if len(g.indices) == 3]
    assert t_groups and all(all(np.sign(g.characters[k]) > 0 for k in pg.idx_sigma_all) for g in t_groups)
