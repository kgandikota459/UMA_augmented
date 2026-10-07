"""Canonical record schema for experimental and reference observations.

Design rules (docs/DESIGN.md, section "Canonical data schema"):
  * every record says whether its value is MEASURED, EVALUATED (critically
    assessed/derived from measurements), INFERRED (a model built from data,
    e.g. a deposited structure) or COMPUTED (quantum/simulation labels);
  * missing conditions stay missing (None) and are listed in
    `conditions.missing`; nothing is imputed silently;
  * uncertainty keeps the source's statement and its type; conversion to a
    standard deviation is an explicit, recorded curation rule;
  * split-group identifiers are part of the record so leakage control does
    not depend on side tables.
"""

from __future__ import annotations

import datetime as dt
import math
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..observables.base import Family


class ValueKind(str, Enum):
    MEASURED = "measured"
    EVALUATED = "evaluated"
    INFERRED = "inferred"
    COMPUTED = "computed"


class UncertaintyType(str, Enum):
    STANDARD = "standard"  # 1 sigma
    EXPANDED = "expanded"  # coverage factor k given
    RANGE_RATING = "range_rating"  # source gives a class/range (e.g. "0~1 cm-1")
    REPORTED_PRECISION = "reported_precision"  # only digits reported
    UNKNOWN = "unknown"


class Phase(str, Enum):
    GAS = "gas"
    LIQUID = "liquid"
    SOLID = "solid"
    CRYSTAL = "crystal"
    SOLUTION = "solution"
    INTERFACE = "interface"
    UNKNOWN = "unknown"


class Redistribution(str, Enum):
    ALLOWED = "allowed"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class TransformStep(_Model):
    step: str
    tool: str
    version: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    timestamp: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))


class Provenance(_Model):
    source_name: str
    source_url: str | None = None
    doi: str | None = None
    source_record_id: str | None = None
    retrieval_date: dt.date | None = None
    dataset_version: str | None = None
    license: str | None = None
    license_url: str | None = None
    redistribution: Redistribution = Redistribution.UNKNOWN
    citation: str | None = None
    transformation_history: list[TransformStep] = Field(default_factory=list)


class Uncertainty(_Model):
    kind: UncertaintyType
    value: float | list[float] | None = None
    coverage_factor: float | None = None
    range: tuple[float, float] | None = None  # for RANGE_RATING
    covariance_ref: str | None = None  # file with a full covariance for array values
    correlation_group: str | None = None  # records sharing a systematic error
    n_replicates: int | None = None
    source_text: str | None = None  # verbatim statement from the source
    conversion_rule: str | None = None  # how standard_sd() was obtained

    def standard_sd(self) -> float | list[float] | None:
        if self.kind == UncertaintyType.STANDARD:
            return self.value
        if self.kind == UncertaintyType.EXPANDED:
            if self.coverage_factor is None:
                raise ValueError("expanded uncertainty without coverage factor")
            k = self.coverage_factor
            return [v / k for v in self.value] if isinstance(self.value, list) else self.value / k
        if self.kind == UncertaintyType.RANGE_RATING:
            if self.range is None:
                return None
            # upper bound of the rating class treated as a ~95% bound
            return self.range[1] / 2.0
        if self.kind == UncertaintyType.REPORTED_PRECISION:
            # half the last reported digit as a uniform half-width -> sd
            return None if self.value is None else self.value / math.sqrt(3.0)
        return None


class Censoring(_Model):
    lower: float | None = None  # true value > lower
    upper: float | None = None  # true value < upper

    @model_validator(mode="after")
    def _check(self):
        if self.lower is None and self.upper is None:
            raise ValueError("censoring needs a bound")
        return self


class Grid(_Model):
    name: str
    values: list[float]
    units: str


class MolecularIdentity(_Model):
    system_id: str
    name: str | None = None
    smiles: str | None = None
    inchi: str | None = None
    inchikey: str | None = None
    cas: str | None = None
    formula: str | None = None
    charge: int | None = None
    spin_multiplicity: int | None = None
    protonation_state: str | None = None
    tautomer: str | None = None
    stereo_defined: bool | None = None
    isotopologue: dict[str, int] | None = None  # element or "atom:<i>" -> mass number
    atom_mapping: list[int] | None = None
    coordinates_ref: str | None = None  # path#sha256
    preparation_recipe: dict[str, Any] | None = None
    protein_construct: str | None = None
    mutations: list[str] | None = None


class Conditions(_Model):
    temperature_k: float | None = None
    temperature_sd_k: float | None = None
    pressure_bar: float | None = None
    phase: Phase = Phase.UNKNOWN
    solvent: str | None = None
    composition: dict[str, float] | None = None  # mole fractions by system_id
    ions: dict[str, float] | None = None  # mol/L
    ph: float | None = None
    buffer: str | None = None
    boundary_conditions: str | None = None
    protocol: str | None = None  # measurement technique
    missing: list[str] = Field(default_factory=list)  # fields absent from the source


class SplitGroups(_Model):
    molecule_group: str  # e.g. InChIKey first block
    scaffold_group: str | None = None
    provenance_group: str | None = None  # dataset / compilation
    publication_group: str | None = None  # primary reference
    system_group: str | None = None  # protein family/sequence cluster, etc.


class Observation(_Model):
    observation_id: str
    system_ids: list[str]
    stoichiometry: list[float] | None = None
    family: Family
    quantity: str
    definition: str
    value: float | list[float]
    units: str
    grid: Grid | None = None
    labels: dict[str, Any] = Field(default_factory=dict)
    uncertainty: Uncertainty
    censoring: Censoring | None = None
    value_kind: ValueKind
    computed_method: str | None = None
    energy_reference: str | None = None
    conditions: Conditions
    forward_model: dict[str, Any] | None = None
    nuisance: dict[str, Any] | None = None
    simulation_protocol: str | None = None
    sample_cache_ref: str | None = None
    compatible_uma_tasks: list[str] = Field(default_factory=list)
    provenance: Provenance
    groups: SplitGroups
    quality_flags: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("system_ids")
    @classmethod
    def _nonempty(cls, v):
        if not v:
            raise ValueError("system_ids must not be empty")
        return v

    @model_validator(mode="after")
    def _consistency(self):
        if self.value_kind == ValueKind.COMPUTED and not self.computed_method:
            raise ValueError("computed values need computed_method (and energy_reference if energies)")
        if self.value_kind != ValueKind.COMPUTED and self.computed_method:
            raise ValueError("computed_method set on a non-computed record")
        if self.stoichiometry is not None and len(self.stoichiometry) != len(self.system_ids):
            raise ValueError("stoichiometry must match system_ids")
        if isinstance(self.value, list) and self.grid is not None and len(self.value) != len(self.grid.values):
            raise ValueError("array value length must match grid")
        return self

    @property
    def is_experimental(self) -> bool:
        return self.value_kind in (ValueKind.MEASURED, ValueKind.EVALUATED)
