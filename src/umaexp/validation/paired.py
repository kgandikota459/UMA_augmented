"""Molecule-level paired comparison of models (the pre-registered Vib-v1 analysis).

The molecule is the unit of analysis: modes of one molecule share a geometry,
a Hessian and systematic errors, so they are not independent observations.

  e_m(model) = RMS over usable distinct fundamentals of molecule m of (prediction - experiment)
  delta      = mean_m [e_m(A) - e_m(B)]        (negative = A better)

Confidence intervals are percentile bootstraps that resample molecules; pooled
mode-level metrics resample molecules as clusters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def per_molecule_rmse(df: pd.DataFrame, pred: str, exp: str = "experiment", group: str = "system") -> pd.Series:
    r = (df[pred] - df[exp]) ** 2
    return r.groupby(df[group]).mean().pow(0.5)


def paired_bootstrap(e_a: pd.Series, e_b: pd.Series, n_boot: int = 10_000, seed: int = 20261006,
                     alpha: float = 0.05) -> dict:
    """delta = mean(e_a - e_b) over molecules present in both, with a percentile CI."""
    idx = e_a.index.intersection(e_b.index)
    n_dropped = int(len(e_a.index.union(e_b.index)) - len(idx))  # reported, never silent
    if len(idx) < 2:
        raise ValueError("need at least two paired molecules")
    d = (e_a.loc[idx] - e_b.loc[idx]).to_numpy(float)
    base = e_b.loc[idx].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[draws].mean(axis=1)
    rel_boot = boot / base[draws].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    rlo, rhi = np.quantile(rel_boot, [alpha / 2, 1 - alpha / 2])
    return {"n_molecules": int(len(d)), "delta": float(d.mean()), "ci": [float(lo), float(hi)],
            "relative": float(d.mean() / base.mean()), "relative_ci": [float(rlo), float(rhi)],
            "mean_e_b": float(base.mean()), "n_a_better": int((d < 0).sum()), "n_boot": n_boot, "seed": seed,
            "n_dropped_unpaired": n_dropped}


def sign_flip_test(e_a: pd.Series, e_b: pd.Series, n_perm: int = 100_000, seed: int = 20261006) -> dict:
    """Two-sided paired sign-flip permutation test of mean(e_a - e_b) = 0 (exact when 2^n <= n_perm)."""
    idx = e_a.index.intersection(e_b.index)
    d = (e_a.loc[idx] - e_b.loc[idx]).to_numpy(float)
    n = len(d)
    obs = abs(d.mean())
    if 2**n <= n_perm:
        signs = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(n_perm, n))
    stats = np.abs((signs * d).mean(axis=1))
    return {"n": n, "p_two_sided": float((stats >= obs - 1e-12).mean()), "exact": bool(2**n <= n_perm)}


def wilcoxon_paired(e_a: pd.Series, e_b: pd.Series) -> dict:
    from scipy.stats import wilcoxon

    idx = e_a.index.intersection(e_b.index)
    res = wilcoxon(e_a.loc[idx].to_numpy(float), e_b.loc[idx].to_numpy(float))
    return {"statistic": float(res.statistic), "p_two_sided": float(res.pvalue), "n": int(len(idx))}


def cluster_bootstrap_metric(df: pd.DataFrame, pred: str, exp: str = "experiment", group: str = "system",
                             metric: str = "rmse", n_boot: int = 10_000, seed: int = 20261006,
                             alpha: float = 0.05) -> dict:
    """Pooled mode-level RMSE/MAE with molecules resampled as clusters."""
    err = (df[pred] - df[exp]).to_numpy(float)
    codes, uniq = pd.factorize(df[group])
    per = [err[codes == k] for k in range(len(uniq))]

    def stat(chunks):
        e = np.concatenate(chunks)
        return float(np.sqrt((e**2).mean())) if metric == "rmse" else float(np.abs(e).mean())

    rng = np.random.default_rng(seed)
    boot = [stat([per[k] for k in rng.integers(0, len(per), len(per))]) for _ in range(n_boot)]
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"metric": metric, "value": stat(per), "ci": [float(lo), float(hi)], "n_modes": int(len(err)),
            "n_molecules": int(len(per))}
