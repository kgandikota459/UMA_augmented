"""Structured experiment configuration (OmegaConf-validated dataclasses).

YAML files in configs/ are merged onto these defaults, so a typo in a key is
an error rather than a silently ignored setting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import OmegaConf


@dataclass
class AdapterConfig:
    enabled: bool = False
    patterns: list[str] = field(default_factory=lambda: [r"blocks\.3\..*fc_m0\.weights$"])
    rank: int = 4
    per_expert: bool = False


@dataclass
class PotentialConfig:
    checkpoint: str | None = None  # registry name (uma-s-1p1) or path; None -> random architecture (testing only)
    task: str = "omol"
    dtype: str = "float64"
    device: str = "cpu"
    preset: str | None = "readout"
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    adapters: AdapterConfig = field(default_factory=AdapterConfig)


@dataclass
class DataConfig:
    manifest: str = "data/manifests/pilot_manifest.json"
    splits: str = "data/manifests/pilot_splits.json"
    processed_dir: str = "data/processed"
    # Vib-v1 curation rule (experiments/vib_v1/DECISIONS.md D-11), frozen before the held-out fetch:
    # explicit gas phase; directly observed band origins only; no resonance-perturbed, silent or
    # tunnelling-split levels; rating uncertainty <= 15 cm-1.
    exclude_flags: list[str] = field(default_factory=lambda: [
        "fermi_resonance", "not_directly_observed", "condensed_phase_only", "phase_not_stated",
        "observed_via_coriolis_interaction", "silent_mode", "inversion_doubling", "rating_uncertainty_gt_15cm",
        # v3 fail-closed flags (critic report #2 C3) and overlapped bands (D-17)
        "overlapped_band", "unknown_note_code", "free_text_comment", "nonnumeric_value",
        "conflicting_duplicate_rows", "selected_not_gas_column"])


@dataclass
class VibrationalConfigSchema:
    enabled: bool = True
    systems: list[str] | None = None  # None -> all train-split systems
    target_kind: str = "fundamental_scaled"  # harmonic | fundamental_scaled | fundamental_corrected
    scale_prior_mean: float = 0.955  # planning value; calibrate on validation systems
    scale_prior_sd: float = 0.02
    fit_scale: bool = True
    rel_model_sd: float = 0.015
    geometry_response: str = "implicit"
    fmax: float = 1e-5
    max_opt_steps: int = 5000
    memory_mode: str = "graph"  # "hvp" for larger molecules (memory-bounded, same gradient)


@dataclass
class DensityConfigSchema:
    enabled: bool = False
    systems: list[str] = field(default_factory=list)
    n_molecules: int = 32
    temperature_k: float = 298.15
    pressure_bar: float = 1.01325
    acknowledge_periodic_omol: bool = False
    dt_fs: float = 0.5
    n_equilibration: int = 2000
    n_production: int = 10000
    stride: int = 20


@dataclass
class SamplingConfig:
    device: str = "cpu"  # "mps" for Apple GPUs (float32)
    dtype: str = "float64"
    cache_dir: str | None = "caches"


@dataclass
class TrainingConfig:
    steps: int = 20
    lr: float = 1e-5
    lr_decay: float = 1.0
    weight_decay: float = 0.0
    grad_clip: float | None = 10.0
    max_relative_update: float | None = 1e-3
    lambda_baseline: float = 0.0
    lambda_quantum: float = 0.0
    l2sp: float = 0.0  # 0.5 * l2sp * ||w - w0||^2 (anchor to pretrained weights, not to zero)
    optimizer: str = "adamw"  # or "lm": observation-space Levenberg-Marquardt + overlap line search
    nuisance_lr: float | None = 2.0e-3  # calibrated nuisances (e.g. frequency scale) learn faster than UMA
    lm_lambda: float = 1.0e-3
    ess_refresh: float = 0.5
    ess_reject: float = 0.1
    checkpoint_every: int = 10
    family_priority: dict[str, float] = field(default_factory=lambda: {"vibrational": 1.0, "density": 1.0})


@dataclass
class ExperimentConfig:
    name: str = "pilot"
    seed: int = 0
    out_dir: str = "runs"
    potential: PotentialConfig = field(default_factory=PotentialConfig)
    data: DataConfig = field(default_factory=DataConfig)
    vibrational: VibrationalConfigSchema = field(default_factory=VibrationalConfigSchema)
    density: DensityConfigSchema = field(default_factory=DensityConfigSchema)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(*paths: str, overrides: list[str] | None = None) -> ExperimentConfig:
    cfg = OmegaConf.structured(ExperimentConfig)
    for p in paths:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(p))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_object(cfg)
