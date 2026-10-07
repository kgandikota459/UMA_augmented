# umaexp: fine-tuning UMA's weights through experimental observables

`umaexp` is a PyTorch research codebase for updating the weights of Meta FAIR's
[UMA](https://arxiv.org/abs/2506.23971) interatomic potentials so that physical
predictions made from UMA's energy surface match experimental measurements.
Examples of such predictions are vibrational frequencies, NPT densities,
fluctuation responses, structure, NMR, free energies and ionization energies.

    U_w (UMA) ──► differentiable physics / statistical mechanics ──► ŷ_k ──► uncertainty-aware loss
       ▲                                                                            │
       └────────────── gradients through Hessians, implicit geometries, ◄───────────┘
                       reweighted ensembles and free-energy estimators

The full design, prior work and verification record are in
**[docs/DESIGN.md](docs/DESIGN.md)**.

## Status (2026-10-07)

| | |
|---|---|
| UMA integration | Native FAIRChem 2.23.0 model with MoLE unmerged, kept in eval mode and trainable. **Real `uma-s-1p1` weights have been in use since 2026-10-06.** Against FAIRChem's own predictor under the same settings, energies agree to 1.6e-7 eV at most (a strained NH4+ test geometry; the reason is not established) and to 2–24 neV for the other parity molecules, which sit at ASE geometries rather than UMA minima (`runs/audit_uma_s_1p1/audit.json`). |
| Derivatives | Forces, Hessians, parameter gradients and mixed third-order derivatives (∂/∂w of Hessian-derived losses) pass finite-difference checks on the real weights. The Hessian-vector-product gradient, including the implicit-geometry response, equals the full-graph gradient to < 1e-6 relative on real weights (`pytest -m uma_real`, run 2026-10-07). Cell-strain derivatives were checked only with random weights. |
| Observables | All 14 families are present in the registry, each with a stated status (`umaexp/observables/base.py::REGISTRY`). |
| Estimators | DiffTRe reweighting in full-graph and memory-bounded surrogate modes. Differentiable EXP, BAR and MBAR, and target-state free energies. |
| Known-truth check | On a Lennard-Jones liquid, a joint density + energy fit with observation-space Levenberg–Marquardt (overlap-limited line search) recovers the true (ε, σ) to < 0.1% in 7 steps. |
| Vibrational experiment | **No generalization result yet.** The pilot's old test split (one molecule, benzene) was scored and seen during development, so all 15 pilot molecules are now development data. A pre-registered protocol (`experiments/vib_v1/`) sets out the rest: a sealed held-out set from the G2 list, value-free symmetry assignment, a base-model GVPT2 anharmonic forward model, and a one-shot unblinding. An adversarial reviewer agent audits it. |
| Data | NIST parser v3.1 (2026-10-07) fails closed (allow-listed note codes, comment grammar, numeric cells, gas-phase cross-check), pinned by a golden test on the 15 cached pilot pages. The original pilot splits were made per family and were **not** leakage-controlled across families (e.g. benzene was vibrational test but FreeSolv train). Vib-v1 instead keeps a public reserved set that every loader refuses. |
| Tests | 203 tests collected, of which 180 run in the fast suite and pass (`pytest -m "not uma and not slow and not network"`). Known-answer tests cover VPT2 on model Hamiltonians (against exact spectra) and symmetry assignment on dev molecules and synthetic geometries. Both modules have had real bugs found by review: VPT2 averaging across near-degenerate species, and D2d handling. Coverage has not been re-measured since 2026-09-26. |

## Quick start

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.lock && uv pip install -e ".[dev]"

pytest -m "not uma"                       # fast suite (toys, estimators, data)
pytest -m uma                             # UMA-S architecture tests (CPU, ~1 min)

python scripts/audit_uma.py               # Stage-0 audit (random architecture)
python scripts/worked_example.py          # equations with numbers (docs section 6)
python scripts/build_pilot_dataset.py     # fetch + curate pilot data (network)
python scripts/run_equilibrium_pilot.py   # Stage-3 machinery on a known-truth liquid
python scripts/train.py configs/vib_pilot.yaml potential.checkpoint=null  # Stage-2 mechanics
```

### Using the real UMA checkpoint

The `facebook/UMA` weights are gated. To use them:

1. Request access at <https://huggingface.co/facebook/UMA>.
2. Run `hf auth login`.
3. Run:

```bash
python scripts/audit_uma.py --checkpoint uma-s-1p1 --task omol
pytest -m uma_real                        # tests on the real weights (skipped if not cached)
```

Held-out data are scored only by `scripts/unblind_vib_v1.py` under the Vib-v1
protocol; `scripts/train.py` scores train and validation splits only.

## Layout

```
src/umaexp/
  potential/    UMAPotential (trainable FAIRChem model), parameter presets, adapters,
                derivative engine, additive-correction comparator, toy potentials
  systems/      System spec (isotopes, charge/spin, task, PBC) and builders
  data/         Pydantic schema, Parquet registry + manifest, leakage-aware splits,
                sources: NIST WebBook, ThermoML, FreeSolv, catalog (D1-D11)
  observables/  one module per family + registry and statuses
  sampling/     BAOAB Langevin, MC barostat, NVE, sample caches, diagnostics
  reweighting/  DiffTRe estimator, controller, differentiable EXP/BAR/MBAR
  losses/       likelihoods, combined objective, anchors, priors
  training/     tasks, cache manager, trainer (8-step controlled loop), checkpoints
  validation/   gradcheck, physics checks, fresh-simulation evaluation, audit
configs/  scripts/  tests/  data/{pilot,manifests}  docs/
```

Raw downloads and derived tables from NIST sources are git-ignored. They are
covered by the Standard Reference Data Act, so regenerate them with the build
script rather than redistributing them.
