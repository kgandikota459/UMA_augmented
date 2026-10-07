# Vib-v1: pre-registered protocol (DRAFT r3, not yet frozen)

**Question (narrowed after red-team report #2).** Fine-tune the readout weights
of UMA-S-1.1 against experimental gas-phase fundamentals, scored with a fixed
base-model VPT2 anharmonic correction. Does this improve the VPT2-corrected
fundamentals of small molecules the procedure has never seen, relative to the
unchanged model with the same correction?

The claim is about VPT2-corrected fundamentals. Fine-tuning can absorb
systematic errors of the fixed correction Δ_base into the harmonic force
constants (§8), so a positive result is not by itself evidence that the
electronic potential improved.

**Status.** DRAFT r3, 2026-10-07. Replaces r2 after red-team report #3.
Decisions are logged in `DECISIONS.md` (append-only). Changes after a freeze go
to `DEVIATIONS.md`.

**Staged freezes** (`scripts/freeze_vib_v1.py`; each stage refuses to overwrite
itself and checks the earlier ones):

| Stage | What is frozen | Required before |
|---|---|---|
| S2 | §2 text plus the builder's full run-time import closure (computed by `freeze_vib_v1.s2_import_closure`: builder, data package incl. parser, curation and point-group modules, config, observables/base), the freeze script, the pilot list and the reserved set | The network fetch |
| PROTOCOL | This whole file, the code tree, and the decision log (append-only from here) | Unsealing D_ext |
| MODEL | Dev data, the sealed T digests, the CV selection, the final FT checkpoint and its summary | Unblinding |

Each freeze needs the critic's written sign-off.

## 1. Exposure statement

* **The 15 pilot molecules are development data only.** All were scored with the
  unchanged model on 2026-10-06. Per-split RMSE was printed (old test = C6H6, 5
  modes; old val = C2H2 and C2H4), along with a per-mode residual table. Both
  of the largest old residuals were later shown to be assignment errors.
* **`runs/vib_stage2_uma_s_1p1*` is quarantined.** Only 3 model-loading log
  lines from it were ever seen.
* **Lr calibration pilot (D-16, D-18, D-19):** it ran on D_pilot; its held-out
  values were never read and inform nothing.
* **Ethanol (a G2 candidate):** its real-weight energy was computed in the
  Stage-0 audit, before the draw (no vibrational quantity). It is therefore
  forced into D_ext (§2.4).
* **No other model output exists** for any G2 candidate.
* **Prior knowledge cannot be removed.** The researchers, including the AI
  agents, know literature fundamentals of common molecules. Mitigations:
  - T and D_ext are sealed until the whole protocol is frozen, so no design
    choice can be informed by working out T from the D_ext names;
  - no model output exists for a T molecule before unblinding.

## 2. Data (frozen at stage S2)

### 2.1 Parser and curation rule
* **Parser `nist_shimanouchi_v3.1`, failing closed.**
  - On a page:
    - the table header must match the expected layout;
    - when there are several tables, a single "Gas" table is required;
    - the table must be attributed to "Shimanouchi, 1972".
  - Note codes are read from the IR, Raman and Comments columns. A code outside
    the allow-list (the 18 codes seen on curated pages) becomes
    `unknown_note_code`.
  - Comments must match the FR/OC/OV grammar, the Coriolis sentence or the
    inversion-level phrases; anything else becomes `free_text_comment`.
  - Value cells must be plain numbers (`nonnumeric_value`).
  - Duplicate rows that conflict in value or rating are flagged
    `conflicting_duplicate_rows`.
  - The selected value must lie within 1 cm⁻¹ of some gas-phase column value
    (`selected_not_gas_column`). A mode with no gas-phase value at all fails
    this check, and a missing selected value is `nonnumeric_value`.
  - Any "(anti)symmetric level" comment sets `inversion_doubling`, whether or
    not its rows were merged.
  - `silent_mode` is set only when both the IR and Raman columns are inactive.
  - Dropped double primes are restored.
* **Usable fundamental:** none of these flags is set:
  - fermi_resonance, not_directly_observed, observed_via_coriolis_interaction,
    silent_mode;
  - condensed_phase_only, phase_not_stated (so the gas phase must be explicit);
  - inversion_doubling, rating_uncertainty_gt_15cm (so ratings A–D only);
  - overlapped_band, unknown_note_code, free_text_comment, nonnumeric_value,
    conflicting_duplicate_rows, selected_not_gas_column.
* A golden regression test pins the parser on the 15 cached pilot pages: 110
  fundamentals, 72 usable.

### 2.2 Pools
* **D_pilot:** the 15 pilot molecules (development only). Linear molecules (HCN,
  C2H2) are dropped from training and scoring; the VPT2 forward model covers
  nonlinear molecules only.
* **Candidates:** `ase.collections.g2` (ASE 3.29.0, 162 species).

### 2.3 Eligibility (mechanical; counts per reason are published)
1. Neutral and closed shell.
2. At least 3 atoms and not linear.
3. RDKit bond perception (charge 0) succeeds, and no atom carries radical
   electrons or a formal charge.
4. The InChIKey first block is not in the pilot list (parents included).

   Offline steps 1–4 leave **73 candidates**. This list is the public reserved
   set (§2.5).
5. Exactly one WebBook record matches. The record must have:
   - this standard InChIKey;
   - its own vibrational-data (Mask=800) link;
   - the candidate's Hill formula.

   "Search Results" pages (at most 50 entries) are resolved by opening every
   listed record. The WebBook contains duplicate and erroneous records that
   copy a key; the live dry run on pilot keys found such records for ethane and
   D2O.
6. The page check passes:
   - the fetched Mask=800 page carries the candidate's InChIKey;
   - its Shimanouchi table parses with no page errors;
   - the page's point group equals the G2 geometry's point group by name
     (Schoenflies symbol computed from the symmetry operations);
   - the table is complete (Σ degeneracies = 3N−6);
   - at least one fundamental is usable under §2.1 with no "torsion" in its
     approximate-type text.

Network failures are retried with backoff. The whole build aborts, and nothing
is published (fail closed), on any of:
- a failure that remains after the retries;
- an unrecognised page (neither a result page nor a species page);
- search results longer than 50 entries;
- any exception during per-candidate processing (only its type is printed).

At fetch time the builder also asserts that its offline candidate set equals
the frozen reserved list.

### 2.4 Split, minimum size and sealing
* **Forced D_ext:** eligible molecules with any real-weight model output before
  the draw (ethanol only).
* **Groups:** the remaining eligible molecules are grouped by InChIKey first
  block and split into two strata, H/C/N/O only versus other elements.
* **Draw:** one `numpy.random.default_rng(20261006)` permutes each stratum's
  sorted groups, in that stratum order. The first ceil(n·f) groups go to **T**,
  the rest to **D_ext**.
* **Minimum size, applied before any model output exists:**
  - f = 0.5. T must have at least 15 molecules and at least 130 label-usable
    modes (§2.3 step 6 rule; the margin allows for the §3.2 losses, about 85%
    retention on D_pilot).
  - If that fails, f = 0.6, and then f = 0.7, with the same permutation, so a
    larger T contains every smaller one.
  - If f = 0.7 still misses the minimum, the build proceeds at f = 0.7 and
    records `below_minimum_size: true`. The primary analysis stands and is
    reported as underpowered. No revision of the rule is allowed.
* **Sealing:**
  - **Both T and D_ext** go outside the repository (`$UMAEXP_SEALED_DIR/build`,
    under random, non-identifying tags), along with:
    - the archived lookup responses;
    - the eligibility table;
    - a manifest hashing every sealed file;
    - a secret salt.
  - The repository gets only `heldout_public.json`, which contains:
    - counts;
    - commitments H(salt ‖ content) for the T and D_ext files and the manifest;
    - the salt's digest;
    - software versions;
    - per-file digests of the code tree at build time.

    It is written atomically before the staging directory is renamed into
    place.
  - D_ext is unsealed by `scripts/unseal_dev_ext.py` only after the PROTOCOL
    freeze; it verifies the commitments first. T stays sealed until
    unblinding, when the salt is disclosed.

### 2.5 Reserved set (cross-family)
* `reserved_blocks.json` lists the 73 offline candidates.
* The vibrational, density and Vib-v1 loaders refuse them for any training or
  development use until unblinding. A missing list raises an error, and there
  is no bypass (the unblinding script builds T molecules directly).
* D_ext blocks are released when D_ext is unsealed.
* Any future data family must call `assert_not_reserved`.

### 2.6 Structures
* Every model optimizes its own minimum from the G2 geometry, which fixes the
  conformer: fmax 1e-5 eV/Å, at most 5000 steps.
* Pilot molecules use the G2 geometry of their InChIKey block, with
  SMILES-uniform isotopes (D2O uses H2O with D masses). Otherwise RDKit
  embedding is used.

## 3. Forward model and assignment (identical for every model)

### 3.1 Forward model
* **Prediction:** ν_pred = ω_model + Δ_base.
* **Correction:** Δ_base = ν_GVPT2 − ω, computed once per molecule from the
  unchanged UMA-S-1.1 at its own minimum and held fixed (no gradient).
* **Force field:** cubic and semi-diagonal quartic constants from central
  differences of analytic Hessians along rectilinear normal coordinates (step
  0.025 in dimensionless coordinates).
* **Anharmonic constants:** Mills' asymmetric-top formulas, including the
  Coriolis terms. Degenerate modes are treated through their Cartesian
  components.
* **Averaging:** components are averaged only within irrep-pure symmetry
  mode groups, never across species that happen to be near-degenerate.
* **Fermi resonances:**
  - Detected by Martin's test, with |Δ| < 200 cm⁻¹ (an exact zero gap counts
    as resonant) and:
    - type I: φ_iik⁴/(256Δ³) > 1 cm⁻¹;
    - type II: φ_ijk⁴/(64Δ³) > 1 cm⁻¹.
  - The detected terms are removed (DVPT2), and the fundamental is
    diagonalized with its partner states (GVPT2). Couplings are φ_iik/4 and
    φ_ijk/(2√2).
* **Decoupling (primary):** modes with ω < 400 cm⁻¹, or with torsion fraction
  ≥ 0.5 (§3.2), are removed from every VPT2 sum and get no correction.
* **Validation, done before freezing:**
  - Exact-spectrum tests: a Morse oscillator (exact), coupled 2-D models with
    λ⁴ error scaling, degenerate 2-D Hénon–Heiles (including at realistic
    strength), a triply degenerate F-type model, and type I and type II Fermi
    resonances (both dyad energies).
  - Finite-difference force field against autograd third derivatives.
  - Real-weight invariance to random rotations inside degenerate subspaces
    (CH4, CH3Cl): within 0.05 cm⁻¹ at step 0.025, and shrinking like step², so
    the residual is numerical.
* **Degenerate-mode scope rule (not pre-registered):** had that invariance
  check failed, molecules with degenerate modes would have been excluded. The
  rule was written into this protocol only after the check had passed
  (DECISIONS D-21), so it is not a pre-registered safeguard, and no claim
  relies on it.
* **Robustness checks:** coupled (no decoupling); no Coriolis terms; Δ scaled
  by 0.85 and 1.15; Δ recomputed self-consistently from FT (FT-selfconsistent).
  FT was trained under the primary Δ, so these checks are biased against FT.
  They are reported as robustness checks, not as alternative primary analyses.

### 3.2 Scored modes S_m
Rules are label-free, fixed per molecule from the base model, and the same for
every model. A fundamental is scored if all of these hold:
* it is usable (§2.1) and matched by the base assignment (§3.3);
* ω_base ≥ 400 cm⁻¹;
* it is not a torsion: the source type text does not contain "torsion", and
  its torsion fraction is < 0.5;
* |Δ_base| ≤ 0.08·ω_base (R = 0.08, set from Dev: maximum 0.048, median 0.025
  over the 60 scored D_pilot modes; DECISIONS D-20, D-22);
* its GVPT2 fundamental weight is ≥ 0.7.

Definitions and edge cases:
* **Torsion fraction:** the largest squared overlap of the mode with a rigid
  internal rotation of a fragment about an acyclic bond (mass-weighted,
  rigid-body motions projected out).
* **Empty S_m:** the molecule is excluded and counted.

### 3.3 Assignment
* **Done once, on the base model, using the full table** (flagged modes keep
  their slots).
* **Symmetry:**
  - Symmetry operations are found from the minimum geometry (atoms told apart
    by element and mass) and closed into a group.
  - Near-degenerate eigenvectors (within 1 cm⁻¹) are separated into irreps with
    a random class-sum operator.
* **Convention-free label constraints:** dimension, g/u, ′/″, A vs B, the
  subscript of A, the subscript of E, and in D2d the B1/B2 split (B1 is
  symmetric under C2′).
* **B1/B2/B3 subscripts:** the possible axis conventions are enumerated (2 for
  groups with two σv or C2′ classes, 6 for D2-type groups). Only conventions
  consistent with every species count at once are kept.
* **Matching:** within a species, mode numbers ascending are paired with
  computed frequencies descending (value-free). Species are merged only when
  the consistent conventions disagree; a merged class is matched by
  measured-value rank (a sensitivity analysis excludes those modes).
* **Failures:**
  - A class whose counts disagree is excluded.
  - An unmatched usable mode raises an error.
  - If the irreps are not clean, assignment falls back to the group-symmetrized
    Hessian.
* **Point group:** the base minimum must have the page's point group (by
  Schoenflies name), or the molecule is excluded.
* **Groups without a unique proper principal axis:**
  - D2d and S4 use the S4 axis as the principal axis.
  - Td tells A1 from A2, and T2 from T1, by the sign of the σd character.
* **Other models (FT)** are never re-ranked. At every evaluation, and at every
  training step, their modes are mapped to the fixed base assignment by
  eigenvector-subspace tracking:
  - the new modes are grouped by irrep, not by frequency closeness;
  - matching is Hungarian on subspace overlaps.

  A prediction is invalid if a scored mode's overlap is below 0.5, or if the
  model's minimum has a different point group. During training, an invalid
  prediction makes the molecule abstain for that step.

## 4. Models
* **B-VPT2:** the unchanged model with the §3 forward model. This is the primary
  comparator. Nothing is fitted, so it equals a frozen-weights arm trained with
  the same objective.
* **B-scale (descriptive only):** unchanged harmonic wavenumbers times one
  least-squares scale fitted on the Dev S_m modes.
* **FT:** the readout preset fine-tuned on Dev (§6).
* **FT-selfconsistent (sensitivity):** FT with Δ recomputed from FT.

## 5. Endpoints and analysis (molecule = unit of analysis)
* **Per-molecule error:** e_m = RMS over S_m of (ν_pred − ν_exp) in cm⁻¹.
  Degenerate modes count once.
* **Molecules analysed:** those with a valid base model and a non-empty S_m.
* **Invalid FT predictions:**
  - the optimization does not converge;
  - any vibrational wavenumber is below 0, or any scored mode is below
    50 cm⁻¹ ("soft");
  - the point group changes;
  - a tracking overlap is below 0.5.

  Each invalid prediction gets e_m = max(e_m^B-VPT2, the model's worst valid
  e_m), so a failure can never count as an improvement (conservative
  imputation; **primary**). Counts are reported per model.
* **Primary decision (H1, one-sided α = 2.5%):**
  - Δ = mean_m(e_m^FT − e_m^B-VPT2), with a percentile bootstrap over molecules
    (10,000 resamples, seed 20261006).
  - **Meaningful improvement** if the upper 95% bound is below −MEI, with
    MEI = 2.5 cm⁻¹.
  - Also reported:
    - any improvement (upper bound below 0);
    - harm (lower bound above 0);
    - meaningful harm (lower bound above +MEI).
* **Secondary:**
  - sign-flip permutation test (exact when 2^n ≤ 10⁵);
  - Wilcoxon signed-rank test;
  - FT vs B-scale;
  - pooled mode RMSE and MAE with molecule-cluster bootstrap;
  - X–H stretch vs other modes;
  - invalid counts.
* **Sensitivity:**
  - complete-case analysis;
  - ratings A–C only;
  - excluding crossing-risk modes (a same-irrep base neighbour within 30 cm⁻¹);
  - excluding modes in any detected resonance;
  - excluding merged-class modes;
  - molecules with at least 3 scored modes;
  - per stratum;
  - coupled VPT2;
  - no Coriolis terms;
  - Δ × 0.85 and × 1.15;
  - FT-selfconsistent;
  - leave-one-molecule-out range.

### 5.1 Drift guardrail (absolute bounds, fixed now; reported with H1)
* **Bounds:**
  - Force: RMS over samples, atoms and Cartesian components of
    F_FT − F_base, at most 0.05 eV/Å (= 1.15 kcal/mol/Å).
  - Energy: RMS over samples of the relative-energy deviation, with each
    molecule's mean offset removed, at most 0.02 eV per molecule
    (= 0.46 kcal/mol).

  Both are set a priori, on the conventional 1 kcal/mol scale.
* **Sampling:** 20 300 K normal-mode displacements per T molecule (seed fixed),
  with amplitudes of modes below 200 cm⁻¹ capped at the 200 cm⁻¹ amplitude.
* **Also reported:** per-molecule energy offsets per atom.
* **Exceeding a bound** is reported as harm next to H1.

### 5.2 Development-only diagnostics
* H2O/D2O isotope consistency of the residuals.
* Cross-validated Δ.
* Power estimate. It is informational; no decision depends on it.

## 6. Development procedure (Dev = D_pilot ∪ D_ext, after the PROTOCOL freeze)
* **Objective:** the mean over molecules of the mean over S_m of
  (ν_pred − ν_exp)² / (10 cm⁻¹)². Molecules count equally and modes uniformly
  within a molecule, matching the per-molecule endpoint.
  - Implementation: the data term is the sum of per-molecule NLLs, each with
    per-mode variance n_m·(10 cm⁻¹)², divided by the number of valid molecules
    in the batch (`FamilyWeight(1.0, "n_targets")`).
* **Regularization:**
  - L2-SP 1.0;
  - baseline-drift λ 1.0, using a replay of displaced Dev geometries only.
* **Optimizer:** Adam with lr 1e-3, mini-batches of 8 molecules drawn without
  replacement (seeded). The trust-region cap sets the step (D-18).
* **Cross-validation:**
  - Grouped 3-fold over Dev, fold seed 20261006.
  - Grid: cap ∈ {3e-4, 1e-3, 3e-3}, up to 60 steps, held-out folds scored
    every 10 steps with anchored evaluation.
  - Held-out values are written to trace files, not printed.
* **Selection:**
  - Criterion: the CV mean of e_m. Invalid held-out predictions get
    max(e_m^B-VPT2, the configuration's worst valid e_m).
  - One-standard-error rule on paired per-molecule differences to the best
    configuration: a configuration is eligible if its mean difference to the
    best is at most the standard error of those differences. Among eligible
    configurations, take the smallest cap, then the fewest steps.
* **Final FT:**
  - Retrained on all Dev with the selected cap.
  - Steps are epoch-matched: CV steps × |Dev| / mean training-fold size.
  - Trainable tensors are saved, and in-sample Dev diagnostics are recorded.
* **Before the MODEL freeze:** a full dress rehearsal of the unblinding code on
  Dev that exercises the real path:
  - the base precompute, recomputed from the parquet;
  - an FT model taken from a CV fold, so that tracking sees modes that have
    moved;
  - environment and digests recorded.

## 7. Unblinding
1. `scripts/unblind_vib_v1.py` refuses to run unless the MODEL-stage digests
   all match and no lock file exists. The digests cover:
   - the protocol and code files;
   - the Dev data and the Dev precompute;
   - the sealed build manifest and T files;
   - the CV selection and the final summary;
   - the base `uma-s-1p1` checkpoint and the FT checkpoint.

   The lock file is written before the first held-out computation. Code
   changes made after the PROTOCOL freeze must each be named in
   `DEVIATIONS.md`, and their diff is written at the MODEL freeze.
2. It then:
   - computes the base precompute (§3) for T, which is the first model output
     for any T molecule;
   - scores B-VPT2, B-scale, FT and FT-selfconsistent once;
   - runs the drift guardrail;
   - writes results and a lock file.
3. A re-run is allowed only for a bug fix recorded in `DEVIATIONS.md`. Only the
   code digest may differ, and results from both before and after the fix are
   reported.
4. The salt is disclosed so the commitments can be checked. All outcomes are
   reported.

## 8. Threats to validity (declared in advance)
* **Fixed Δ_base.** It ignores FT's change in anharmonicity (second order), and
  FT can absorb systematic VPT2 errors into harmonic force constants. That is
  why the claim is narrowed and the FT-selfconsistent sensitivity analysis
  exists. Experimental harmonic constants (ω_e) are not available in a curated
  source within scope, so the harmonic surface itself is not tested.
* **VPT2 accuracy.** On a DFT-quality surface, residuals are about 10 cm⁻¹ for
  semi-rigid molecules. Decoupling large-amplitude modes is an approximation.
* **Coriolis convention.** The Mills asymmetric-top Coriolis terms are used for
  every molecule. For perpendicular bands of light symmetric tops, the band-origin
  convention can differ by a few cm⁻¹; this is covered by the no-Coriolis
  sensitivity analysis. It is common to both models, so it largely cancels in Δ.
* **Scope.** Nonlinear, neutral, closed-shell G2 molecules (about ≤ 14 atoms),
  semi-rigid modes only (ω ≥ 400 cm⁻¹, no torsions or resonance-mixed states).
* **Labels.** Compiled "selected" values; the curation rule is strict and fails
  closed.
* **Inferability of T.** It is mitigated by sealing D_ext until the PROTOCOL
  freeze. After that, T is partly inferable from the public lists, but every
  design choice is already frozen.
* **Geometry and conformer.** The G2 (MP2) conformer is assumed to be the one
  measured.
* **Design limits.** One base model, one trainable subset, one random split.
