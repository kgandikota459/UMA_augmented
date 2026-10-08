# Vib-v1 decision log

Every decision that can affect an analysis. For each: when it was made (UTC),
what had been seen at that time, and whether information from a test split
could have influenced it. Append only. Corrections are added as new entries.

## Made before any real-weight outcome existed (2026-09-26)

- **D-01.** The molecule list (`data/pilot/vibrational_pilot.yaml`), the
  original split (`group_split`, seed 0), the exclusion flags, the forward model
  (`fundamental_scaled`, prior 0.955 ± 0.02, rel_model_sd 0.015) and the
  assignment algorithm (`degeneracy_sorted`) were all fixed before the gated
  weights were available. The critic verified this from file times and history.
  These choices are untainted by outcomes, but several are now known to be
  defective (see D-09 to D-12).

## 2026-10-06: real weights; the old test split was exposed

- **D-02, 03:13–03:14Z.** `baseline_vibrations.py` printed per-split RMSE,
  including the old test split (C6H6 only), and a per-mode residual table
  (C6H6 e1u14 at −67.9 cm⁻¹ among the rows). **Exposure.**
- **D-03, 03:15Z.** Test-split evaluation was added to `scripts/train.py`.
  This was done after D-02. **Tainted.** Reverted under Vib-v1 (D-13).
- **D-04, 03:16Z.** Stage 2 was launched with lr 1e-4 and 40 steps (the config
  said 1e-5, "tune on validation"). The choice had no validation procedure and
  came after D-02. **Tainted.**
  - The run (`runs/vib_stage2_uma_s_1p1*`) is quarantined and excluded.
  - The only content anyone saw from it: the last 3 lines of its log at
    03:18:56Z, all model-loading messages (HTTP HEAD for the checkpoint, a
    `dataset_list` deprecation warning, a redirect notice).
  - A further progress check was rejected by the user before it ran.
- **D-05.** The diagnostic "X–H stretch residual +10.6 vs others −15.9 cm⁻¹"
  pooled the old val and test splits. **Withdrawn.** On train alone it is
  +10.5 vs −17.6 (critic).
- **D-06.** The narrative that the largest residuals were "physically
  plausible" (CH3OH 2844 Fermi; C2H2 ν4 and HCN ν3 as real model errors) is
  **withdrawn**:
  - it was written after seeing residuals;
  - it omitted the two largest residuals, one of them a verified misassignment
    (C2H4 au4);
  - the CH3OH 2844 mode is not flagged as Fermi-resonant in the source data.

## 2026-10-06: Vib-v1 design (no G2 molecule fetched or evaluated yet)

- **D-07.** All 15 pilot molecules become development data, and the old splits
  are retired. Nothing measured on them is evidence of generalization.
- **D-08.** The held-out pool is `ase.collections.g2`, with eligibility and the
  split fixed in PROTOCOL §2 before any fetch.
  - The offline steps (1–4) were run and printed counts only, plus the names of
    the 4 molecules where bond perception failed (ClF3, AlF3, SiH2 singlet,
    AlCl3) and of the 17 that match the pilot list. None of these can be in T.
- **D-09.** The scientific target is the transferable electronic (BO) potential
  (spec line 63). Anharmonicity enters the forward model explicitly: VPT2 on the
  base model, held fixed (stop-gradient). There is no free global scale.
  - Reason: critic C3 (H2O and D2O need different scales; the readout absorbed
    the global scale in the smoke run) and the spec's warning against folding
    missing physics into the weights.
- **D-10.** Mode assignment is constrained to symmetry species from
  eigenvector characters, with no value-based matching anywhere, and any
  unmatched usable mode is an error.
  - Reason: critic C2 (verified misassignments in C2H4 and C2H6, and silent
    drops).
- **D-11.** Parser and curation fixes are frozen before T is fetched:
  - a″ species: NIST prints them as "a" in Cs tables;
  - `inactive_in_ir_or_raman` currently fires on every mode of
    centrosymmetric molecules; it is replaced by `silent_mode`, set only when
    both the IR and Raman columns are inactive;
  - phase-not-stated values are excluded, the same as condensed-phase values.
  - Reason: critic M5.
- **D-12.** Comparators are:
  - the base model with VPT2 (no fitting);
  - the base model with a least-squares global scale (reference only);
  - a frozen-weights arm trained with the identical objective, so the effect of
    changing weights is isolated (critic M1).
  
  Invalid predictions count as failures, and a fixed set of modes is scored.
- **D-13.** Test evaluation is removed from `train.py` and
  `baseline_vibrations.py`. The held-out set can only be scored by the
  one-shot unblinding script (critic C1).
- **D-14.** One molecule-to-split map applies across data families. T
  molecules are held out of every family's training data, enforced by a salted
  hash check (critic M4).
- **D-15 (open, density family).** `pipeline.density_tasks` sets the starting
  box density from the measured densities, so labels enter the simulation
  setup. NPT equilibration should forget it, but this has not been shown. It
  must be replaced by a label-free estimate before any density evaluation. Found
  while removing test-split loading from the test suite. That test had also
  built test-split density tasks (labels loaded but not printed); it now checks
  packing directly.
- **D-16 (2026-10-07, Dev only).** The base-model precompute on the 13
  nonlinear pilot molecules finished.
  - All converged with no imaginary modes, and the point groups match the
    source.
  - Merged classes appear only for convention-dependent B pairs (C2H4 b2u/b3u,
    C6H6 b1u/b2u).
  - S_m totals 61 scored modes over 12 molecules. Exclusions: 33 by source
    flags, 4 by resonance weight < 0.7, 3 with ω < 400 cm⁻¹, 1 torsion. NH3 has
    0 scored modes (phase not stated, inversion doubling).
  - A learning-rate calibration pilot follows: fold 0 of a 3-fold grouped split
    of these 12 molecules, lr ∈ {1e-4, 1e-3}, 10 steps, batch 8. It only sets the
    CV grid (PROTOCOL 6) and is not part of any reported result.
- **D-17 (2026-10-07; red-team report #2 B1–B8; before any fetch).**
  - **Parser `nist_shimanouchi_v3` fails closed.** These conditions now exclude
    a mode, as do overlapped bands (OV):
    - note codes outside an allow-list (the 18 codes seen on the curated pages);
    - comments outside the FR/OC/OV, Coriolis and inversion-level grammar;
    - value cells that are not plain numbers;
    - duplicate rows with conflicting values or ratings;
    - selected values more than 1 cm⁻¹ from every gas-phase column value.

    On a page, the gas table is chosen explicitly, the header layout is
    validated, and the Shimanouchi (1972) source is required. A golden test
    over the 15 cached pilot pages records the result: 110 fundamentals, 72
    usable (was 76; CH3OH ν12 and three OV modes are now out).
  - **Eligibility.**
    - Step 3 also excludes radical and charge-separated perceptions.
    - Ethanol is forced into D_ext, because a real-weight energy was computed
      for it before the draw.
    - Linear molecules are dropped from Dev training and scoring as well
      (HCN, C2H2).
    - The offline candidate count is now 73.
  - **Builder.**
    - Fails closed: retries with backoff, and any remaining network error aborts
      with nothing published.
    - Validates every page: exact standard InChIKey (including "Search Results"
      disambiguation; the resolver dry run on pilot keys exposed this for
      ethane), Shimanouchi source, page point group against the G2 geometry,
      complete table (Σ degeneracies = 3N−6).
    - Draws the split over InChIKey-block groups, implements the §2.4 size rule
      (≥ 15 molecules and ≥ 130 label-usable modes; fraction 0.5 then 0.6; then
      a terminal abort), and records versions.
  - **Sealing.**
    - Both T and D_ext are written outside the repository
      ($UMAEXP_SEALED_DIR), under non-identifying file names.
    - Commitments are H(secret salt ‖ content).
    - D_ext stays sealed until the whole protocol is frozen. Development
      continues on D_pilot only.
  - **Reserved set.** The public reserved set (`reserved_blocks.json`, 73 InChIKey
    blocks) is enforced in the vibrational, density and Vib-v1 loaders.
  - **Lr pilot (D-16).** Its held-out-fold evaluations are not read and not used
    for any decision. Only training-loss and update behaviour inform the grid
    (critic M7).
- **D-18 (2026-10-07; lr pilot, training behaviour only; no held-out values
  read).** With max_relative_update = 1e-3, the first four steps at lr 1e-4 and
  at lr 1e-3 are identical (same loss and gradient norm). The trust-region cap,
  not the learning rate, sets the step, so a learning-rate grid is degenerate.
  Training loss is still falling at step 10 (1.23 → 0.94–0.96), at about 42 s
  per 8-molecule step.

  Decided:
  - Adam lr is fixed at 1e-3, and the CV grid is over the cap
    max_relative_update ∈ {3e-4, 1e-3, 3e-3};
  - step budget up to 60, scored every 10 steps;
  - the trainer now logs the proposed and applied relative step and the trust
    scale.
- **D-19 (2026-10-07).** The background waiter for the lr pilot printed the
  pilot log's tail, which includes held-out-fold lines, into a harness output
  file. That file has not been opened, and the pilot's held-out numbers stay
  unread (D-17).
- **D-20 (2026-10-07, D_pilot only).** Results of the precompute under parser
  v3 and the r2 rules (decoupled GVPT2, step 0.025, axis-convention
  assignment):
  - 12 molecules with 59 scored modes, and no merged classes.
  - Exclusions: 37 by source flags, 3 by resonance weight, 2 with ω < 400 cm⁻¹,
    1 torsion.
  - Over the scored modes, |Δ_base|/ω has maximum 0.048 (HCOOH ν1, O–H
    stretch) and median 0.025.
  - The largest torsion fraction among scored modes is 0.011.
  - Crossing-risk flags: 1 mode. Resonance flags: 10 modes.

  Decided: R = 0.08, about 1.7 times the Dev maximum, replacing the
  placeholder 0.25 (critic M6). It leaves the Dev scored set unchanged.
- **D-21 (2026-10-07; red-team report #3; still before any fetch). Corrections
  to my own earlier entries:**
  - **D-17 overstated four things.** These were true of the builder as written
    then:
    1. A 200-status page that was neither a result page nor a species page was
       silently excluded, not aborted on.
    2. The fetched Mask=800 page was never checked for its InChIKey.
    3. Point groups were compared by order only.
    4. The "non-identifying" tags were deterministic candidate indices.

    All four are fixed in this round.
  - **D-20 contained an artifact.** One of its three "resonance weight < 0.7"
    exclusions was caused by averaging Δ and weights across near-degenerate
    modes of different symmetry: C2H6 a1g1 at 3056.68 and a2u5 at 3056.21
    cm⁻¹, both within 0.5 cm⁻¹ (P1). Averaging is now done within irrep-pure
    groups only. The Dev precompute is re-run (D-22).
  - **The degenerate-mode scope rule (PROTOCOL 3.1) was written into the
    protocol only after the invariance test had passed.** It is therefore not
    a pre-registered safeguard, and no claim relies on it.
- **D-21b (S2 fixes).**
  - **Parser v3.1.**
    - A mode with no gas-phase value fails `selected_not_gas_column`.
    - A single "(anti)symmetric level" row is flagged `inversion_doubling`.
    - An empty selected cell is `nonnumeric_value` (it no longer crashes).
    - The golden file was regenerated deliberately: usable count still 72, and
      only already-excluded modes changed.
  - **Resolver.**
    - Unrecognised pages, and search results over 50 entries, abort the build.
    - A record needs the exact key, its own Mask=800 link, and the Hill
      formula.
    - The live dry run on non-test keys (15 pilot molecules, the 3 pilot
      species without tables, and one invalid key; responses in
      runs/vib_v1/resolver_dryrun/) was 19/19 consistent with the cached
      build. It also showed that the WebBook holds a second record with
      ethane's key and formula (R633468, no vibrational link) and two "Deuterium
      oxide" records. Requiring the record's own vibrational link resolves
      both.
  - **Builder.**
    - The fetched page's InChIKey is checked.
    - Point groups are compared by Schoenflies name.
    - Every exception during per-candidate processing aborts.
    - At fetch time the offline candidates must equal the reserved list.
    - Tags are random; a manifest hashes every sealed file and is committed.
    - Per-file code digests are recorded at build time.
    - Publication is atomic: PUBLIC is written via os.replace, then the staging
      directory is renamed.
    - The size-rule sequence is 0.5 → 0.6 → 0.7, then proceed with the result
      flagged as underpowered.
  - **Code layout and freeze scope.**
    - Point-group detection moved to `umaexp/data/pointgroup.py`, frozen at S2.
      D2d and S4 use the S4 axis; Td subscripts come from σd.
    - The curation rule moved to `umaexp/data/curation.py`.
    - `reserved.py` fails closed, with no bypass.
    - The S2 freeze scope is now the builder's import closure, computed by
      importing it fresh (16 files), plus the freeze script, the pilot list and
      the reserved list.
- **D-21c (protocol-stage fixes from report #3).**
  - **P1.** VPT2 corrections are averaged within irrep groups only.
  - **P2.** Tracking:
    - every model is tracked from the fixed base assignment, never from the
      previous step;
    - new modes are grouped by irrep;
    - overlap must be ≥ 0.5 in evaluation and in training (otherwise the
      molecule abstains for that step).
  - **P3.** Imputation is e = max(e_B-VPT2, worst valid e), in analyse() and in
    the CV selection.
  - **P4.** The one-SE rule uses paired differences to the best configuration.
  - **P5.** Per-molecule normalization (`n_targets`).
  - **P6.** D2d and Td handling, and point groups by name.
  - **P7.** Freezes:
    - the PROTOCOL stage stores per-file digests and a code archive;
    - at the MODEL freeze, every changed file must be named in
      DEVIATIONS.md, and its diff is written;
    - the MODEL stage digests the base checkpoint and the Dev precompute;
    - the unblinding lock is written before the first held-out computation.
  - **P8.** The dress rehearsal must use the real path; still to do before the
    MODEL freeze.
  - **P9.** The drift text is corrected (per-component RMS; 0.05 eV/Å =
    1.15 kcal/mol/Å).
  - **P10.**
    - Defaults are aligned: 60 steps, scored every 10.
    - The invalid-prediction rules are written out in the protocol.
    - Martin's test has a zero-gap guard.
    - The Δ variants are labelled robustness checks.
- **D-22 (2026-10-07; D_pilot only).** Dev precompute re-run after the P1 fix:
  - 12 molecules, 60 scored modes. C2H6 a1g1 is now scored, as the critic
    predicted.
  - Exclusions: 37 by source flags, 2 by resonance weight, 2 with ω < 400 cm⁻¹,
    1 torsion. No merged classes, and every point group equals the page's
    group by name.
  - |Δ|/ω on scored modes: maximum 0.048, median 0.025. R = 0.08 stands.
- **D-23 (2026-10-07; D_pilot only; code exercise, not evidence).** The
  real-path dress rehearsal (PROTOCOL 6, critic P8) needs an FT whose modes
  have moved. It is trained on fold 0 of a 3-fold grouped split of D_pilot for
  10 steps at cap 1e-3.
  - Its held-out trace is not read.
  - From the rehearsal outputs only structural fields are inspected: validity,
    tracking overlaps, invalid counts and the presence of every analysis field.
    No error value informs any decision.
- **D-24 (2026-10-07). Data-loss incident and reconstruction.**
  - **What happened.** About 16:13Z the whole project directory was deleted to
    free disk space. It was not in the Trash, and Time Machine was not
    configured.
  - **How it was rebuilt.** Every Write, Edit and Bash file operation in the
    session transcript was replayed in order into an empty directory:
    - Text-edit scripts ran under system python3, with an import blocker for
      heavy modules.
    - Computations, tests, network calls and process control were skipped.
    - The memory directory was never touched.
  - **Verification:**
    - **(a) Claude Code file-history snapshots:** 94 of 99 checkpoints are
      byte-identical. Of the other 5:
      - 4 are documentation test counts that had been filled in from live pytest
        output;
      - 1 is a `run_equilibrium_pilot.py` edit chained behind a `cp` of a run
        artifact.

      All 5 were re-synced to the backed-up content.
    - **(b) The 16 S2 digests printed before the deletion:** all byte-identical.
      That includes `reserved_blocks.json`, regenerated with the recorded script
      under the rebuilt environment.
    - **(c)** Every edit command whose replay failed was checked by hand against
      its original output.
  - **Environment.** Python 3.12.13, with the key versions recovered from the
    transcript: torch 2.13.0, fairchem-core 2.23.0, ase 3.29.0, numpy 2.4.6,
    rdkit 2026.3.6, scipy 1.18.1, pandas 3.0.6, pyarrow 25.0.1, pydantic
    2.13.5, pint 0.26.1, omegaconf 2.3.1, hydra-core 1.3.7, e3nn 0.6.0,
    pymbar 4.0.3, huggingface-hub 2.0.0. Transitive dependencies were
    re-resolved; the full original lock was not recoverable.
  - **Regenerated:** the public pilot data (re-fetched), the derived parquets,
    the golden parser fixture (from the re-fetched pages) and the Dev
    precompute.
  - **Lost:**
    - `docs/results/*` (copies of earlier pilot outputs);
    - all of `runs/`, including the quarantined Stage-2 run (never read) and
      the lr-pilot traces (never read);
    - rehearsal outputs.
  - **Integrity.** No sealed data existed yet, and no held-out information was
    involved.
- **D-25 (2026-10-07; red-team report #4, after the D-24 rebuild).**
  - **Builder fixes.**
    - A 200-status Mask=800 response that is not a species page now aborts the
      build (FetchError) instead of being silently excluded as `6f`.
    - The split and sealing steps after the candidate loop are wrapped: any
      exception aborts with its type only and deletes staging.
  - **Freeze tool.** `verify_s2` is strict at fetch time. At later stages, an S2
    code file may differ only if it is named in DEVIATIONS.md; §2 text changes
    are always fatal.
  - **Assignment.**
    - D2d B1/B2 are resolved convention-free from χ(C2′).
    - A merged class containing a non-numeric tabulated value is excluded
      (fail closed).
  - **Protocol text (outside §2).**
    - The degenerate-mode scope rule is now marked as not pre-registered.
    - R's basis is updated to 60 modes (D-22).
    - The S2 row of the staged-freeze table names the import closure.
  - **Housekeeping.**
    - Regenerated `requirements.lock` from the rebuilt environment (it was
      0 bytes after the replay).
    - Deleted the replay-created 0-byte `runs/vib_stage2_uma_s_1p1.log`.
    - Deleted unread the harness output files of the lr-pilot waiters (D-19),
      the only remaining copy of those held-out lines.
    - Re-ran the T-free resolver dry run against the live WebBook: 19/19
      consistent, archived in runs/vib_v1/resolver_dryrun/ (summary sha256
      8f6995f73bdbb277…).
- **D-26 (2026-10-07T19:48:19Z). Stage S2 frozen.**
  - The critic's sign-off (red-team report #5) is recorded verbatim in
    experiments/vib_v1/FROZEN_S2.json, including its condition for later
    stages.
  - FROZEN_S2.json sha256: efce129cc400dbcd69263d8ce9680bc70ed05471b502c619a0a9e98b9a56f4c2.
    It is also printed in the session conversation.
  - The 16 S2 files are archived in experiments/vib_v1/s2_frozen_files.tar.gz
    (sha256 f27d8c3e94579fa84001047620ccca0c5ca44bde6f881c0f4123e05def3e39b9), so any later change can be diffed against the frozen
    versions.
  - P8 real-path dress rehearsal, redone after D-24. The base precompute was
    recomputed from the parquet, and the FT came from D_pilot fold 0 after 10
    steps at cap 1e-3. Structural fields only: 12 molecules; 0 invalid
    predictions for B-VPT2, B-scale, FT and FT-selfconsistent; 60 scored rows
    per model; all analysis fields and all 14 sensitivity entries present.
    No error value was read.
  - The user chose a git repository with a private GitHub remote, and an
    encrypted disk image of the sealed build held by the user.
- **D-27 (2026-10-07/08). Held-out fetch completed under the frozen S2.**
  - **Eligibility.** 34 of 73 offline candidates are eligible. Excluded:
    - 30 have no unique vibrational (Mask=800) link;
    - 1 has no unique exact species;
    - 1 has a formula mismatch;
    - 3 have page errors;
    - 1 has a point-group mismatch;
    - 3 have no usable fundamental.

    No forced-dev candidate was eligible (ethanol had no qualifying page).
  - **Split.** Fraction 0.5; below_minimum_size false.
    - HCNO_only stratum: 16 groups, 8 to T.
    - Other-elements stratum: 18 groups, 9 to T.
  - **Result.**
    - T: 17 molecules, 151 label-usable modes, against a minimum of 15
      molecules and 130 modes.
    - D_ext: 17 molecules.
    - Both are sealed in $UMAEXP_SEALED_DIR/build. Only counts were displayed.
  - **Public record.** experiments/vib_v1/heldout_public.json sha256 8777a2fe61de31159fd96c73df800104c79ecf6a6f3f0b1075cf826c60305fcb, also
    printed in the session conversation.
