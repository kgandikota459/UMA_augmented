"""Trainable UMA potential built on the native FAIRChem PyTorch model.

Verified against fairchem-core 2.23.0 source (2026-09-26):

* `MLIPPredictUnit` (the ASE-calculator path) calls
  `model.requires_grad_(False)` and, in its "default"/"turbo" settings,
  *merges* the mixture-of-linear-experts (MoLE) weights into a new plain
  backbone. Neither is usable for training: gradients would reach a merged
  copy, not UMA's expert weights. We therefore instantiate the model with
  `load_inference_model(..., use_ema=True)` ourselves, keep MoLE unmerged,
  uncompiled, in eval mode (no MoE/composition dropout: the energy surface
  must be deterministic) and enable gradients on selected parameters.

* Energy convention (MLIPPredictUnit._process_outputs):
      E = rmsd * E_head + mean + sum_i ref[Z_i]
  where (mean, rmsd) is the task's Normalizer and ref the task's
  ElementReferences (composition-only constants, no gradient). Forces are
  -dE/dR computed here by autograd; the head's own force branch is switched
  off (`regress_config.forces = False`) to avoid duplicate work.

* Inputs: total charge and spin (omol: multiplicity) go into AtomicData
  `charge`/`spin`; the task name goes into `dataset`. `AtomicData.from_ase`
  silently zeroes charge/spin unless r_data_keys includes them, so we build
  AtomicData directly from tensors, which also keeps positions/cell on the
  autograd graph.

* MPS (Apple Metal): UMA accumulates node energies in float64
  (outputs.reduce_node_to_system), which MPS does not support. On MPS we
  swap in a same-dtype (float32) reduction for the duration of the call.
  Measured on an M1 Pro: forces agree with CPU float64 to ~1e-7 relative,
  and MPS float32 is ~1.6x faster than CPU float32 for 60-190 atoms.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import logging
import os
from dataclasses import dataclass, field

import torch
from torch import nn

from ..systems.system import System
from .base import Potential, as_batch, wrap_positions

log = logging.getLogger(__name__)

# UMA-S architecture as published in FAIRChem's training-release configs
# (configs/uma/training_release/backbone/K4L2.yaml + uma_sm_direct_pretrain.yaml
#  + uma_sm_conserve_finetune.yaml overrides). Used ONLY to build a
# randomly initialised network for offline testing when the gated checkpoint
# is unavailable. A real checkpoint always supplies its own model_config.
UMA_S_ARCH = dict(
    backbone=dict(
        model="fairchem.core.models.uma.escn_moe.eSCNMDMoeBackbone",
        moe_dropout=0.05,
        moe_layer_type="pytorch",
        num_experts=32,
        use_composition_embedding=True,
        use_global_embedding=False,
        max_num_elements=100,
        sphere_channels=128,
        lmax=2,
        mmax=2,
        otf_graph=True,
        max_neighbors=300,
        use_pbc=True,
        use_pbc_single=True,
        cutoff=6.0,
        edge_channels=128,
        distance_function="gaussian",
        num_distance_basis=64,
        regress_forces=True,
        regress_stress=True,
        direct_forces=False,
        num_layers=4,
        hidden_channels=128,
        norm_type="rms_norm_sh",
        act_type="gate",
        ff_type="spectral",
        chg_spin_emb_type="rand_emb",
        cs_emb_grad=True,
        dataset_list=["oc20", "omol", "omat", "odac", "omc"],
        model_version=1.1,
    ),
    heads=dict(
        energyandforcehead=dict(
            module="fairchem.core.models.uma.escn_moe.DatasetSpecificSingleHeadWrapper",
            head_cls="fairchem.core.models.uma.escn_md.MLP_EFS_Head",
            head_kwargs={"wrap_property": False},
            dataset_names=["omol", "omat", "oc20", "omc", "odac"],
        )
    ),
    pass_through_head_outputs=True,
    otf_graph=True,
)
UMA_NORMALIZER_RMSD = 1.423  # normalizer_rmsd in the UMA-S training configs


@dataclass
class EnergyConvention:
    """How a task's raw head output maps to total energy in eV."""

    task: str
    mean: float
    rmsd: float
    element_references: torch.Tensor | None  # float64 (max_Z+1,) or None
    source: str = ""

    def reference_offset(self, atomic_numbers: tuple[int, ...]) -> float:
        if self.element_references is None:
            return 0.0
        refs = self.element_references
        return float(sum(float(refs[z]) for z in atomic_numbers))


@dataclass
class UMAProvenance:
    checkpoint: str
    checkpoint_sha256: str | None
    model_id: str | None
    fairchem_version: str
    random_init: bool
    notes: list[str] = field(default_factory=list)


def _sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _reduce_same_dtype(node_values, batch, num_systems):
    """MPS replacement for fairchem's float64 node->system reduction."""
    shape = (num_systems,) + tuple(node_values.shape[1:])
    out = torch.zeros(shape, device=node_values.device, dtype=node_values.dtype)
    if node_values.dim() == 1:
        out = out.index_add(0, batch, node_values)
    else:
        flat = node_values.reshape(node_values.shape[0], -1)
        out = out.reshape(num_systems, -1).index_add(0, batch, flat).reshape(shape)
    return out, out


@contextlib.contextmanager
def _mps_reduction(active: bool):
    if not active:
        yield
        return
    import fairchem.core.models.uma.outputs as outs

    orig = outs.reduce_node_to_system
    outs.reduce_node_to_system = _reduce_same_dtype
    try:
        yield
    finally:
        outs.reduce_node_to_system = orig


class UMAPotential(Potential):
    updates_uma_weights = True

    def __init__(
        self,
        model: nn.Module,
        convention: EnergyConvention,
        provenance: UMAProvenance,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__()
        self.model = model
        self.convention = convention
        self.provenance = provenance
        self.task = convention.task
        self.energy_key = f"{convention.task}_energy"
        self.dtype = dtype
        self.device_name = device
        self.base_fingerprint = provenance.checkpoint_sha256 or "random-init"
        self._configure_for_training_graph()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        name_or_path: str,
        task: str = "omol",
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
        hash_checkpoint: bool = True,
    ) -> UMAPotential:
        """Load a FAIRChem UMA inference checkpoint for training.

        `name_or_path` is a local .pt path or a registry name such as
        'uma-s-1p1' (downloaded from the gated Hugging Face repo; requires an
        accepted license and a token, e.g. via `hf auth login`).
        """
        import fairchem.core
        import hydra
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
        from fairchem.core.units.mlip_unit.utils import (
            get_backbone_class_from_checkpoint,
            load_inference_model,
        )

        path = name_or_path
        if not os.path.isfile(path):
            from fairchem.core.calculate.pretrained_mlip import pretrained_checkpoint_path_from_name

            path = pretrained_checkpoint_path_from_name(name_or_path)

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        settings = InferenceSettings(
            tf32=False,
            activation_checkpointing=False,
            merge_mole=False,
            compile=False,
            external_graph_gen=False,
            internal_graph_gen_version=2,
            execution_mode="general",
            base_precision_dtype=dtype,
        )
        backbone_cls = get_backbone_class_from_checkpoint(ckpt)
        overrides = {"backbone": backbone_cls.build_inference_settings(settings)}
        prev = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            averaged, ckpt = load_inference_model(
                path, overrides=overrides, use_ema=True, preloaded_checkpoint=ckpt
            )
        finally:
            torch.set_default_dtype(prev)
        model = averaged.module  # AveragedModel wrapper -> HydraModel with EMA weights

        tasks = [hydra.utils.instantiate(t) for t in ckpt.tasks_config]
        conv = _convention_from_tasks(tasks, task, source=f"{name_or_path}:tasks_config")
        prov = UMAProvenance(
            checkpoint=str(name_or_path),
            checkpoint_sha256=_sha256_file(path) if hash_checkpoint else None,
            model_id=ckpt.model_config.get("model_id"),
            fairchem_version=fairchem.core.__version__,
            random_init=False,
        )
        pot = cls(model.to(dtype), conv, prov, dtype=dtype, device=device)
        pot._checkpoint_path = path
        return pot

    @classmethod
    def random_architecture(
        cls,
        task: str = "omol",
        seed: int = 0,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
        arch: dict | None = None,
        element_references: torch.Tensor | None = None,
    ) -> UMAPotential:
        """UMA-S-shaped network with RANDOM weights, for offline testing only.

        Scientifically meaningless energies; used to verify gradient paths,
        parity of the loading code and training mechanics while the gated
        checkpoint is unavailable. Never report physical results from it.
        """
        import fairchem.core
        from fairchem.core.models.base import HydraModel

        cfg = copy.deepcopy(arch or UMA_S_ARCH)
        cfg["backbone"].update(
            always_use_pbc=False, activation_checkpointing=False, otf_graph=True
        )
        torch.manual_seed(seed)
        prev = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            model = HydraModel(
                backbone=cfg["backbone"],
                heads=cfg["heads"],
                pass_through_head_outputs=cfg.get("pass_through_head_outputs", True),
                otf_graph=True,
                model_id=cfg.get("model_id"),
            )
        finally:
            torch.set_default_dtype(prev)
        conv = EnergyConvention(
            task=task, mean=0.0, rmsd=UMA_NORMALIZER_RMSD,
            element_references=element_references,
            source="random-architecture (UMA-S training-config normalizer)",
        )
        prov = UMAProvenance(
            checkpoint="random-architecture",
            checkpoint_sha256=None,
            model_id=None,
            fairchem_version=fairchem.core.__version__,
            random_init=True,
            notes=[f"seed={seed}", "UMA-S architecture from public training configs"],
        )
        pot = cls(model.to(dtype), conv, prov, dtype=dtype, device=device)
        pot._arch_config = cfg
        return pot

    def _configure_for_training_graph(self):
        m = self.model
        m.eval()  # disables MoE/composition dropout: deterministic surface
        rc = m.backbone.regress_config
        # We differentiate the energy ourselves; the head's own gradient
        # branches would otherwise run autograd.grad inside forward.
        rc.forces = False
        rc.stress = False
        rc.hessian = False
        if rc.direct_forces:
            raise ValueError(
                "Direct-force checkpoint: forces are not energy derivatives. "
                "Use a conserving UMA checkpoint."
            )
        for p in m.parameters():
            p.requires_grad_(False)
        m.to(device=self.device_name)
        if self.device_name == "mps" and self.dtype == torch.float64:
            raise ValueError("MPS does not support float64; use dtype=float32 on MPS")

    # ------------------------------------------------------------------
    # energy
    # ------------------------------------------------------------------
    def _atomic_data(self, system: System, pos: torch.Tensor, cell: torch.Tensor):
        from fairchem.core.datasets.atomic_data import AtomicData

        B, N, _ = pos.shape
        dev = pos.device
        long = dict(dtype=torch.long, device=dev)
        z = torch.as_tensor(system.atomic_numbers, **long).repeat(B)
        pbc = torch.as_tensor(system.pbc, dtype=torch.bool, device=dev).view(1, 3).expand(B, 3).contiguous()
        return AtomicData(
            pos=pos.reshape(B * N, 3),
            atomic_numbers=z,
            cell=cell,
            pbc=pbc,
            natoms=torch.full((B,), N, **long),
            edge_index=torch.empty((2, 0), **long),
            cell_offsets=torch.empty((0, 3), dtype=pos.dtype, device=dev),
            nedges=torch.zeros(B, **long),
            charge=torch.full((B,), int(system.charge), **long),
            spin=torch.full((B,), int(system.spin), **long),
            fixed=torch.zeros(B * N, **long),
            tags=torch.zeros(B * N, **long),
            batch=torch.arange(B, device=dev).repeat_interleave(N),
            sid=[system.system_id] * B,
            dataset=[self.task] * B,
        )

    def head_energy(self, system: System, positions: torch.Tensor, cell: torch.Tensor | None = None):
        """Normalized head output (before denormalization), shape (B,)."""
        if system.task != self.task:
            raise ValueError(f"system task {system.task!r} != potential task {self.task!r}")
        pos, c, _ = as_batch(positions, cell)
        dev = self.device_name
        pos = pos.to(self.dtype).to(dev)  # cast on the source device, then move
        B = pos.shape[0]
        if system.is_periodic:
            if c is None:
                raise ValueError("periodic system requires a cell")
            c = c.to(self.dtype).to(dev)
            if c.shape[0] != B:
                c = c.expand(B, 3, 3)
            pos = wrap_positions(pos, c)
        else:
            c = torch.zeros(B, 3, 3, device=dev, dtype=self.dtype)
        data = self._atomic_data(system, pos, c)
        with torch.enable_grad(), _mps_reduction(dev == "mps"):
            out = self.model(data)
        v = out[self.energy_key]
        return v["energy"] if isinstance(v, dict) else v

    def energy(self, system, positions, cell=None):
        """Total energy in eV, always returned as a CPU float64 tensor.

        When the model runs on MPS, gradients still flow back to the
        (device-resident) positions through the dtype/device casts.
        """
        single = positions.dim() == 2
        e_head = self.head_energy(system, positions, cell)
        # move first, then cast: MPS tensors cannot be cast to float64 on-device
        e = e_head.to("cpu").to(torch.float64) * self.convention.rmsd + self.convention.mean
        e = e + self.convention.reference_offset(system.atomic_numbers)
        return e[0] if single else e

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------
    def frozen_copy(self) -> UMAPotential:
        """Unchanged copy of the current weights (baseline / sampling reference)."""
        clone = copy.deepcopy(self)
        for p in clone.parameters():
            p.requires_grad_(False)
        return clone

    def to_device(self, device: str, dtype: torch.dtype | None = None) -> UMAPotential:
        """Copy of this potential on another device/dtype (e.g. MPS for sampling)."""
        clone = copy.deepcopy(self)
        clone.dtype = dtype or clone.dtype
        clone.device_name = device
        clone.model.to(device=device, dtype=clone.dtype)
        if device == "mps" and clone.dtype == torch.float64:
            raise ValueError("MPS does not support float64")
        return clone

    def export_inference_checkpoint(self, out_path: str, base_checkpoint: str | None = None,
                                    dtype: torch.dtype = torch.float32) -> str:
        """Write a FAIRChem-loadable inference checkpoint with the current weights.

        Adapters must be merged first (`adapters.merge_adapters`). The
        checkpoint keeps the base model_config/tasks_config, so it loads with
        `fairchem.core.calculate.pretrained_mlip.load_predict_unit` and the
        FAIRChemCalculator exactly like the original UMA checkpoint.
        """
        from .adapters import has_adapters

        if has_adapters(self.model):
            raise RuntimeError("merge adapters before exporting")
        base = base_checkpoint or getattr(self, "_checkpoint_path", None)
        if base is None:
            raise ValueError("need the base checkpoint whose configs are reused")
        ckpt = torch.load(base, map_location="cpu", weights_only=False)
        sd = {k: v.detach().to("cpu") for k, v in self.model.state_dict().items()}
        sd = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in sd.items()}
        ckpt.model_state_dict = sd
        ema = {f"module.{k}": v for k, v in sd.items()}
        ema["n_averaged"] = torch.tensor(1, dtype=torch.long)
        ckpt.ema_state_dict = ema
        torch.save(ckpt, out_path)
        return out_path


def _convention_from_tasks(tasks, task: str, source: str) -> EnergyConvention:
    for t in tasks:
        if t.property == "energy" and task in t.datasets:
            refs = None
            if t.element_references is not None:
                refs = t.element_references.element_references.detach().to("cpu", torch.float64)
            return EnergyConvention(
                task=task,
                mean=float(t.normalizer.mean),
                rmsd=float(t.normalizer.rmsd),
                element_references=refs,
                source=f"{source}:{t.name}",
            )
    raise KeyError(f"No energy task for dataset {task!r} in checkpoint tasks")


def write_synthetic_checkpoint(pot: UMAPotential, path: str, omol_refs: list[float] | None = None) -> str:
    """Serialize a random-architecture UMAPotential as a FAIRChem inference checkpoint.

    Lets tests exercise FAIRChem's own loader/predictor (the exact code path
    used for the real checkpoint) against this package's energy wrapper.
    """
    from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint

    if not pot.provenance.random_init:
        raise ValueError("only for random-architecture potentials")
    cfg = copy.deepcopy(pot._arch_config)
    cfg["backbone"].pop("always_use_pbc", None)
    cfg["backbone"].pop("activation_checkpointing", None)
    model_config = {"_target_": "fairchem.core.models.base.HydraModel", **cfg}
    model_config["backbone"]["otf_graph"] = False  # as trained; inference overrides it
    datasets = cfg["heads"]["energyandforcehead"]["dataset_names"]
    refs = pot.convention.element_references
    tasks = []
    for ds in datasets:
        norm = {"_target_": "fairchem.core.modules.normalization.normalizer.Normalizer",
                "mean": 0.0, "rmsd": pot.convention.rmsd}
        er = None
        if ds == pot.task and refs is not None:
            er = {"_target_": "fairchem.core.modules.normalization.element_references.ElementReferences",
                  "element_references": {"_target_": "torch.DoubleTensor", "_args_": [refs.tolist()]}}
        tasks.append({"_target_": "fairchem.core.units.mlip_unit.mlip_unit.Task",
                      "name": f"{ds}_energy", "level": "system", "property": "energy",
                      "out_spec": {"dim": [1], "dtype": "float32"}, "normalizer": norm,
                      "element_references": er, "datasets": [ds]})
        tasks.append({"_target_": "fairchem.core.units.mlip_unit.mlip_unit.Task",
                      "name": f"{ds}_forces", "level": "atom", "property": "forces",
                      "out_spec": {"dim": [3], "dtype": "float32"}, "normalizer": dict(norm),
                      "datasets": [ds]})
    sd = {k: v.detach().to("cpu").to(torch.float32) if v.is_floating_point() else v.detach().cpu()
          for k, v in pot.model.state_dict().items()}
    ema = {f"module.{k}": v for k, v in sd.items()}
    ema["n_averaged"] = torch.tensor(1, dtype=torch.long)
    torch.save(MLIPInferenceCheckpoint(model_config=model_config, model_state_dict=sd,
                                       ema_state_dict=ema, tasks_config=tasks), path)
    return path
