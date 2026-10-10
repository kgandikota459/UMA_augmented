# Vib-v1 results (unblinded 2026-10-10)

Single unblinding run `results/20261010T215353Z`, under `FROZEN.json` (record
sha256 12ccd5c5…e109). No re-run, no bug fix after unblinding. Deviations from
the frozen protocol: DV-01 and DV-03 (both before unblinding, Dev evidence
only; see DEVIATIONS.md).

## Primary outcome (H1): not supported

| | value |
|---|---|
| Test molecules analysed | 16 of 17 (1 excluded, below) |
| Scored modes | 125 |
| Mean per-molecule RMS error, base + VPT2 (B-VPT2) | 25.07 cm⁻¹ |
| Mean per-molecule RMS error, fine-tuned (FT) | 24.59 cm⁻¹ |
| **Δ = mean(e_FT − e_B-VPT2)** | **−0.48 cm⁻¹, 95% CI [−4.89, +4.76]** |
| Relative | −1.9% [−19.7%, +22.5%] |
| Molecules where FT is better | 12 of 16 |
| Sign-flip test (exact, two-sided) | p = 0.86 |
| Wilcoxon signed-rank (two-sided) | p = 0.21 |

The pre-registered criterion was an upper CI bound below −2.5 cm⁻¹. The upper
bound is +4.76. **No meaningful improvement, no improvement, no harm and no
meaningful harm can be claimed**; the interval covers both −2.5 and +2.5, so
the test is inconclusive rather than evidence of no effect.

The Dev cross-validated estimate (−3.63 cm⁻¹, n = 28) did not carry over.

## Harm guardrail (§5.1): force-drift bound exceeded

* Max force RMS deviation from the base model: 0.076 eV/Å (NF3), above the
  0.05 bound; all other test molecules below it (next 0.042, propyne).
  Median 0.015.
* Relative-energy bound met (max 0.0051 eV against 0.02).
* On Dev the bound was already exceeded for F2O (0.133), known before
  unblinding.

Per the protocol this is reported as harm next to H1.

## Secondary and sensitivity analyses

* FT vs B-scale (one least-squares scale, 0.9599, fitted on Dev):
  Δ = −5.78 [−10.98, −0.06]. B-VPT2 itself is 5.3 cm⁻¹ better than B-scale,
  so this reflects the VPT2 correction, not the fine-tuning.
* Pooled over modes: RMSE 33.86 (B-VPT2) vs 33.10 (FT); MAE 17.95 vs 17.62.
* By class (pooled RMSE): X–H stretches 19.1 (B-VPT2) → 28.1 (FT), worse;
  other modes 37.3 → 34.5, better.
* FT-selfconsistent (Δ_anh from FT's own force field): Δ = +0.06
  [−5.11, +6.80].
* Leave-one-molecule-out range of Δ: −2.54 to +0.68.
* Strata: H/C/N/O-only (7 molecules) −2.04 [−5.28, +0.41]; other elements
  (9 molecules) +0.73 [−6.70, +9.49].
* Other sensitivity analyses (ratings A–C, no crossing-risk modes, no
  resonance modes, no merged-class modes, ≥ 3 modes, coupled Δ, no-Coriolis Δ,
  Δ × 0.85, Δ × 1.15): Δ between −2.84 and +1.52, every interval contains 0.
* No invalid predictions for any model; no imputation was needed.

## Exclusions

* Molecule: acetone (CH3COCH3), empty scored set. The base model's minimum
  has point group C2, the source table assumes C2v, so all 12 label-usable
  modes are excluded (pre-registered rule; no model-vs-experiment value
  involved).
* Modes (of 214 tabulated on T): 125 scored; 60 source flags; 12 point-group
  mismatch (acetone); 8 with ω_base < 400 cm⁻¹; 6 resonance weight < 0.7;
  3 torsions. No DV-01 exclusions occurred on T.

## Per-molecule RMS errors (cm⁻¹)

| molecule | modes | B-VPT2 | FT | FT − B-VPT2 |
|---|---|---|---|---|
| NF3 | 4 | 27.8 | 9.9 | −18.0 |
| OCHCHO | 4 | 39.4 | 28.4 | −11.0 |
| C2F4 | 6 | 20.9 | 12.7 | −8.3 |
| COF2 | 6 | 17.0 | 13.6 | −3.4 |
| H2CCl2 | 6 | 21.4 | 18.3 | −3.0 |
| C3H8 | 18 | 13.5 | 11.1 | −2.4 |
| CH3CHO | 11 | 78.8 | 76.4 | −2.4 |
| CH3COF | 14 | 17.7 | 15.8 | −2.0 |
| CF4 | 3 | 8.3 | 6.9 | −1.4 |
| CH3COOH | 13 | 52.5 | 52.0 | −0.6 |
| H3CNH2 | 10 | 10.8 | 10.4 | −0.5 |
| CH3OCH3 | 8 | 8.1 | 8.1 | −0.0 |
| HCF3 | 5 | 22.8 | 24.8 | +1.9 |
| C3H4 (propyne) | 6 | 37.9 | 40.5 | +2.6 |
| CH3SiH3 | 8 | 13.8 | 24.1 | +10.3 |
| SiH4 | 3 | 10.3 | 40.7 | +30.4 |

## Exploratory observations (post hoc, not pre-registered, no claim)

* The two molecules with large degradation both contain Si–H bonds. Dev had
  no Si–H bond (its silicon molecules are SiCl4 and SiF4). The leave-one-out
  minimum (−2.54) is the value without SiH4. This suggests the fitted change
  does not transfer to bond types absent from the training molecules; it
  would need a new pre-registered test to establish.
* CH3CHO and CH3COOH have base errors of 79 and 52 cm⁻¹ that FT does not
  change; whether these are model errors or label/assignment problems of the
  kind seen for aziridine on Dev was not investigated.

## Limits

* 16 molecules; the interval half-width (≈ 4.8 cm⁻¹) is about twice the
  minimum effect of interest, so the test could not have detected an effect
  of the size seen on Dev.
* Label quality: source tables are decades old; symmetry labels and mode
  numbering are taken as given (PROTOCOL §8).
* Δ_anh comes from the base model for both arms.
* Two post-freeze code deviations (DV-01, DV-03) were needed; the first CV
  selection and final model (D-34) were voided and their Dev numbers seen
  before the rerun.

## Integrity

* Salt disclosed in `SALT_DISCLOSED.txt`; its digest and the three published
  commitments (test parquet, test molecule list, sealed manifest) verify
  against the files copied to `results/unsealed/`.
* `results/UNBLINDED.lock`: one start, one finish.
