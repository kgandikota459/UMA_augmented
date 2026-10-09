"""Mode tracking by subspace projection (DEVIATIONS.md DV-03)."""

import numpy as np

from umaexp.observables.vibrational import ModeMatch, track_modes


def _basis(n=6, seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return q


def _rot(q, i, j, theta):
    out = q.copy()
    c, s = np.cos(theta), np.sin(theta)
    out[:, i], out[:, j] = c * q[:, i] + s * q[:, j], -s * q[:, i] + c * q[:, j]
    return out


def test_identity_gives_full_overlap():
    q = _basis()
    m = [ModeMatch(0, [0]), ModeMatch(1, [1, 2]), ModeMatch(2, [4])]
    out = track_modes(q, m, q, np.arange(6.0))
    assert [t.comp_indices for t in out] == [[0], [1, 2], [4]]
    assert all(abs(float(t.note.split("=")[1]) - 1) < 1e-9 for t in out)


def test_degenerate_pair_rotated_and_split_by_grouping_is_still_tracked():
    q = _basis()
    new = _rot(q, 1, 2, 0.7)
    m = [ModeMatch(0, [1, 2])]
    # the supplied grouping splits the pair into singletons: must not matter
    out = track_modes(q, m, new, np.array([0, 1.0, 3.2, 5, 6, 7]), [[0], [1], [2], [3], [4], [5]])
    assert out[0].comp_indices == [1, 2] and abs(float(out[0].note.split("=")[1]) - 1) < 1e-9


def test_reordered_modes_are_followed():
    q = _basis()
    new = q[:, [3, 1, 2, 0, 5, 4]]
    m = [ModeMatch(0, [0]), ModeMatch(1, [4]), ModeMatch(2, [1, 2])]
    out = track_modes(q, m, new, np.arange(6.0))
    assert [t.comp_indices for t in out] == [[3], [5], [1, 2]]


def test_mixed_mode_reports_low_overlap():
    q = _basis()
    new = _rot(q, 0, 3, np.pi / 4)  # tracked mode 0 mixes 50/50 with an untracked mode
    out = track_modes(q, [ModeMatch(0, [0])], new, np.arange(6.0))
    assert abs(float(out[0].note.split("=")[1]) - 0.5) < 1e-9 and out[0].ambiguous


def test_matches_never_share_a_mode():
    q = _basis()
    new = _rot(q, 0, 1, np.pi / 4)
    out = track_modes(q, [ModeMatch(0, [0]), ModeMatch(1, [1])], new, np.arange(6.0))
    assert sorted(out[0].comp_indices + out[1].comp_indices) == [0, 1]
