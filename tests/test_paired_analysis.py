"""Known-answer tests for the molecule-level paired analysis."""

import numpy as np
import pytest
import pandas as pd

from umaexp.validation.paired import cluster_bootstrap_metric, paired_bootstrap, per_molecule_rmse


def _frame(n_mol=30, n_modes=6, shift_b=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_mol):
        bias = rng.normal(0, 10)  # shared within a molecule: modes are not independent
        for k in range(n_modes):
            y = 1000 + 300 * k
            err = bias + rng.normal(0, 5)
            rows.append({"system": f"m{m}", "experiment": y, "A": y + err, "B": y + err + shift_b})
    return pd.DataFrame(rows)


def test_per_molecule_rmse_matches_hand_computation():
    df = pd.DataFrame({"system": ["a", "a", "b"], "experiment": [0.0, 0.0, 0.0], "p": [3.0, 4.0, 2.0]})
    e = per_molecule_rmse(df, "p")
    assert np.isclose(e["a"], np.sqrt(12.5)) and np.isclose(e["b"], 2.0)


def test_paired_bootstrap_identical_models_give_zero_delta():
    df = _frame()
    e = per_molecule_rmse(df, "A")
    r = paired_bootstrap(e, e, n_boot=500)
    assert r["delta"] == 0.0 and r["ci"] == [0.0, 0.0] and r["n_a_better"] == 0


def test_paired_bootstrap_detects_a_real_improvement_and_brackets_it():
    df = _frame(shift_b=8.0)  # B carries an extra +8 cm-1 error on every mode
    ea, eb = per_molecule_rmse(df, "A"), per_molecule_rmse(df, "B")
    r = paired_bootstrap(ea, eb, n_boot=2000)
    assert r["ci"][1] < 0 and r["ci"][0] < r["delta"] < r["ci"][1]


def test_paired_bootstrap_ci_covers_zero_under_the_null_at_about_nominal_rate():
    hits = 0
    n_rep = 200
    for s in range(n_rep):
        rng = np.random.default_rng(1000 + s)
        ea = pd.Series(rng.normal(30, 6, 25), index=[f"m{i}" for i in range(25)])
        eb = pd.Series(rng.normal(30, 6, 25), index=ea.index)
        r = paired_bootstrap(ea, eb, n_boot=1000, seed=s)
        hits += r["ci"][0] <= 0 <= r["ci"][1]
    assert 0.88 <= hits / n_rep <= 0.99  # percentile bootstrap, n=25: close to 95%


def test_cluster_bootstrap_is_wider_than_naive_mode_bootstrap_for_correlated_modes():
    df = _frame(n_mol=20, n_modes=10)
    clus = cluster_bootstrap_metric(df, "A", n_boot=1000)
    rng = np.random.default_rng(0)
    err = (df["A"] - df["experiment"]).to_numpy()
    naive = [np.sqrt((err[rng.integers(0, len(err), len(err))] ** 2).mean()) for _ in range(1000)]
    assert clus["ci"][1] - clus["ci"][0] > np.subtract(*np.quantile(naive, [0.975, 0.025]))


def test_sign_flip_test_is_exact_for_small_n_and_detects_a_shift():
    from umaexp.validation.paired import sign_flip_test

    idx = [f"m{i}" for i in range(10)]
    a = pd.Series(np.arange(10.0), index=idx)
    r = sign_flip_test(a, a, n_perm=100_000)
    assert r["exact"] and r["p_two_sided"] == 1.0
    r2 = sign_flip_test(a - 3.0, a)
    assert r2["p_two_sided"] == pytest.approx(2 / 2**10)  # only all-negative / all-positive reach |mean| = 3


def test_paired_bootstrap_reports_unpaired_molecules():
    a = pd.Series([1.0, 2.0, 3.0], index=["x", "y", "z"])
    b = pd.Series([1.0, 2.0], index=["x", "y"])
    assert paired_bootstrap(a, b, n_boot=100)["n_dropped_unpaired"] == 1
