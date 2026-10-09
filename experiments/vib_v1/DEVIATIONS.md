# Vib-v1 deviations from the frozen protocol

Each entry: what changed, why, the evidence (Dev only), the files, and the
critic's verdict. T is still sealed for every entry unless stated.

## DV-01 (2026-10-09): species not numbered in descending frequency are excluded

* **Found on:** D_ext molecule cyclobutane, in the first Dev precompute after
  the PROTOCOL freeze (DECISIONS D-31, D-32). No T information is involved.
* **Problem.** PROTOCOL §3.3 pairs mode numbers ascending with computed
  frequencies descending within a species. That assumes the source numbers each
  species in descending frequency. It held for all 50 pilot species, but not for
  cyclobutane, whose a1, b1, b2 and e species are numbered otherwise (a1 runs
  2895, 1443, 1001, 2975, 741, 197). The result was a silent misassignment:
  base RMS error 922 cm⁻¹ against a Dev median of 13.8, with one residual of
  −2105 cm⁻¹.
* **Change.** Within a species matched by number, if the tabulated values ever
  increase with mode number, or a value is missing, the whole species is
  excluded for every model and noted. The check uses the tabulated values of
  the source table only (labels, never model output), is mechanical, and fails
  closed. Ties are allowed.
* **Effect on Dev (labels only):** 4 of 97 species have an inversion, all in
  cyclobutane. Its 6 scored modes are removed, so it has an empty scored set.
  Scored modes go from 187 to 181, over 28 molecules. No other molecule
  changes.
* **Why not rank by measured value instead:** that would be value-based
  matching, which the protocol avoids; excluding is the conservative choice.
* **Files:** src/umaexp/observables/symmetry.py (`assign_by_symmetry`), with a
  test in tests/test_symmetry.py.
* **Consequences:** the cross-validation launched under D-31 was stopped
  within its first 3 steps, and its outputs set aside unread
  (runs/vib_v1/cv_aborted/). The Dev precompute is re-run before CV restarts.
* **Critic verdict:** accepted (red-team report #8, 2026-10-09), as written:
  whole-species exclusion, exact ties allowed, no tolerance. When exclusions
  are reported by reason, DV-01 cases are counted from the `notes` field.

## DV-03 (2026-10-09): mode tracking by projection over modes, not over groups

* **Found on:** Dev, by the critic at the MODEL-stage review (red-team report
  #9, NO-GO). No T information is involved; T is still sealed.
* **Problem.** PROTOCOL §3 anchors every non-base model to the base assignment
  by eigenvector-subspace tracking with overlap ≥ 0.5. The implementation
  matched each base subspace to a *group* of the new model's modes and added a
  penalty of 1 when the group size differed. When the new model's grouping
  split a degenerate pair (ethane e_g: an FT pair 1.6 cm⁻¹ from an e_u pair;
  allene e: a pair 2.2 cm⁻¹ apart), a pair whose squared projection onto the
  new modes is 1.00 was scored 0.00 or −0.50 and the molecule declared invalid.
  The unchanged base model failed its own tracking on allene. Invalid molecules
  drop out of training steps and are imputed conservatively when scored, so
  the error biased training, CV selection and the endpoint, always against FT
  (ethane +41 cm⁻¹, allene +9 cm⁻¹ on Dev).
* **Change.** `track_modes` assigns individual modes: a base match of
  dimension d takes d new modes, one-to-one across matches (Hungarian),
  maximizing the squared projection onto its base subspace; the overlap is the
  sum of those projections divided by d. No grouping of the new modes and no
  size penalty enter. The 0.5 threshold and everything else are unchanged.
* **Acceptance (Dev, real weights; runs/vib_v1/dv03_acceptance.json):**
  - FT = base: 0 invalid of 28; predictions equal B-VPT2 to 8e-4 cm⁻¹ on 27
    molecules and to 0.19 cm⁻¹ on aziridine, whose base assignment and ω come
    from the group-symmetrized Hessian (self-overlap 0.85 there, 1.00
    elsewhere).
  - Final FT of D-34 (trained under the faulty rule): 0 invalid of 28; ethane
    overlap 0.996, allene 0.999; lowest overlap 0.974.
* **Files:** src/umaexp/observables/vibrational.py (`track_modes`), with
  tests/test_track_modes.py.
* **Consequences:** the CV, selection, final FT and rehearsals of D-34 are
  void. They are set aside (runs/vib_v1/void_dv03/) and rerun as registered,
  saving one CV-fold checkpoint for the real-path rehearsal.
* **Critic verdict:** accepted (red-team report #10, 2026-10-09). The 0.19 cm⁻¹
  aziridine difference is left as a known limit; residual mixing of numerically
  degenerate modes of different symmetry fails closed and is acceptable.

## Considered and declined (no code change)

* **DV-02 (proposed by the critic, declined 2026-10-09).** Oxirane loses 6
  usable modes because two B modes of different symmetry lie 4.4 cm⁻¹ apart and
  are slightly mixed, so the class counts fail and the class is excluded. The
  proposal was to trigger the symmetrized-Hessian fallback whenever a character
  differs from an integer by more than 0.05. Declined: the current behaviour
  already fails closed, and the threshold would sit just below oxirane's own
  value (0.054), which makes it a rule tuned to one Dev molecule after the
  freeze. Such cases stay excluded, for Dev and T alike.
