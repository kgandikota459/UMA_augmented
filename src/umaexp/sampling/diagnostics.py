"""Time-series and sampling diagnostics.

statistical_inefficiency follows the estimator of Chodera et al. (JCTC 2007)
as implemented in pymbar.timeseries (same defaults; `fast` enables adaptive lag increments), written
out here so it can be applied to reweighted series; tests compare it with
pymbar. detect_equilibration follows Chodera (JCTC 2016): choose the burn-in
t0 maximizing the number of effectively uncorrelated samples.
"""

from __future__ import annotations

import numpy as np


def statistical_inefficiency(x: np.ndarray, y: np.ndarray | None = None, mintime: int = 3,
                             fast: bool = False) -> float:
    x = np.asarray(x, dtype=float)
    y = x if y is None else np.asarray(y, dtype=float)
    n = x.size
    if n < 2:
        return 1.0
    dx, dy = x - x.mean(), y - y.mean()
    sigma2 = float((dx * dy).mean())
    if sigma2 == 0:
        return 1.0
    g = 1.0
    t = 1
    increment = 1
    while t < n - 1:
        c = float(np.sum(dx[: n - t] * dy[t:] + dy[: n - t] * dx[t:])) / (2.0 * (n - t) * sigma2)
        if c <= 0.0 and t > mintime:
            break
        g += 2.0 * c * (1.0 - t / n) * increment
        t += increment
        if fast:
            increment += 1
    return max(g, 1.0)


def detect_equilibration(x: np.ndarray, nskip: int | None = None) -> tuple[int, float, float]:
    """Return (t0, g, N_eff) maximizing N_eff = (T - t0) / g(t0).

    Scans burn-in candidates every `nskip` frames (default T//200) using the
    fast lag schedule to keep the scan O(T) per candidate.
    """
    x = np.asarray(x, dtype=float)
    T = x.size
    nskip = nskip or max(1, T // 200)
    best = (0, 1.0, 0.0)
    for t0 in range(0, T - 1, nskip):
        g = statistical_inefficiency(x[t0:], fast=True)
        neff = (T - t0) / g
        if neff > best[2]:
            best = (t0, g, neff)
    return best


def block_standard_error(x: np.ndarray, n_blocks: int = 10) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size // n_blocks * n_blocks
    if n < n_blocks * 2:
        return float("nan")
    means = x[:n].reshape(n_blocks, -1).mean(axis=1)
    return float(means.std(ddof=1) / np.sqrt(n_blocks))


def replica_consistency(means: np.ndarray, sems: np.ndarray) -> float:
    """Chi^2/dof of independent-replica means about their weighted mean."""
    means, sems = np.asarray(means, float), np.asarray(sems, float)
    w = 1.0 / np.maximum(sems, 1e-300) ** 2
    mu = float((w * means).sum() / w.sum())
    return float(((means - mu) ** 2 * w).sum() / max(len(means) - 1, 1))
