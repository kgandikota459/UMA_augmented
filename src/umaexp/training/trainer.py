"""Controlled update loop for fitting UMA's weights to experimental observables.

Per step:
  1. evaluate due tasks (direct observables with graph; ensemble observables
     through their caches, in memory-bounded surrogate mode by default);
  2. consult the sampling controller: accept / extend / refresh / abstain,
     recording why;
  3. combine experimental, anchor, baseline and prior terms;
  4. back-propagate: graph parts by autograd, surrogate parts through their
     chain-rule callbacks; check that trainable gradients are finite and not
     all zero;
  5. apply a clipped, optionally trust-region-limited update: AdamW, or
     observation-space Levenberg-Marquardt dw = -J^T (J J^T + lam I)^-1 r
     (few observables, many weights);
  6. re-check reweighting overlap with the new weights; if it collapsed,
     revert the update (parameters AND optimizer state) and shrink the step
     (AdamW: lr shrink; LM: halve the step along the Gauss-Newton direction);
  7. log everything and checkpoint.
Held-out evaluation with FRESH simulations is separate (validation/evaluation.py).
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from ..losses.objective import FamilyWeight, combine_objective
from ..reweighting.controller import ControllerConfig, Decision, assess
from ..reweighting.difftre import normalized_log_weights
from .cache_manager import CacheManager
from .checkpointing import save_training_state
from .tasks import DirectTask, EnsembleTask

log = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    steps: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: float | None = 10.0
    max_relative_update: float | None = 1e-2  # ||dw|| / max(||w||, floor) cap per step (trust region)
    trust_region_norm_floor: float = 1.0  # keeps the cap meaningful for parameters near zero
    lr_decay: float = 1.0  # multiplicative per accepted step
    # Separate learning rate for calibrated nuisance parameters (e.g. a frequency
    # scale ~0.96). Sharing UMA's tiny lr leaves them at their prior, and the
    # potential's weights then absorb the nuisance's job.
    nuisance_lr: float | None = None
    lr_shrink_on_reject: float = 0.5
    max_rejections_per_step: int = 3
    max_sampling_attempts: int = 3  # extend/refresh attempts before abstaining
    family_weights: dict[str, FamilyWeight] = field(default_factory=dict)
    lambda_baseline: float = 0.0
    lambda_quantum: float = 0.0
    # L2-SP: 0.5 * l2sp * ||w - w0||^2 over trainable potential weights. Plain
    # weight decay pulls fine-tuned weights toward ZERO, not toward the
    # pretrained UMA weights; prefer l2sp (parameter space) and/or
    # lambda_baseline (function space).
    l2sp: float = 0.0
    # "adamw", or "lm": Levenberg-Marquardt in observation space,
    #   dw = -J^T (J J^T + lam I)^{-1} r   (whitened residuals r, Jacobian J),
    # which needs only one Jacobian row per observed component (cheap when
    # observables are few and parameters many) and handles curved, noisy
    # valleys far better than Adam. Includes experimental rows and nuisance
    # priors; l2sp / baseline / quantum terms are Adam-only (use the damping
    # lam and the trust region to bound the step instead).
    optimizer: str = "adamw"
    # Relative damping: lam = lm_lambda * tr(J J^T)/n_obs (scale-free). The step
    # LENGTH is set separately by backtracking along the Gauss-Newton direction
    # until the post-update reweighting ESS is acceptable (overlap-limited line
    # search): reweighting, not the linearization, limits how far one step can go.
    lm_lambda: float = 1e-3
    lm_max_backtracks: int = 8
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    checkpoint_dir: str | None = None
    checkpoint_every: int = 10
    log_path: str | None = None
    seed: int = 0
    batch_size: int | None = None  # direct tasks per step, drawn without replacement (None: all)


class Trainer:
    def __init__(self, potential, tasks: list, config: TrainerConfig, baseline=None, cache_manager=None,
                 anchors=None, replay=None, nuisance: dict | None = None, priors: dict | None = None):
        torch.manual_seed(config.seed)
        self.pot = potential
        self.tasks = tasks
        self.cfg = config
        self.baseline = baseline
        self.cm = cache_manager or CacheManager()
        self.anchors = anchors or []
        # [(system, positions (M, N, 3))] replay configurations for baseline drift
        self.replay = replay if isinstance(replay, list) or replay is None else [replay]
        self.nuisance = nuisance or {}
        self.priors = priors or {}
        pot_params = potential.trainable_parameters()
        if not pot_params and not self.nuisance:
            raise ValueError("no trainable parameters: select UMA parameters first")
        decay = [p for p in pot_params if p.dim() >= 2]
        no_decay = [p for p in pot_params if p.dim() < 2]  # biases, norm scales
        groups = [{"params": decay, "weight_decay": config.weight_decay},
                  {"params": no_decay, "weight_decay": 0.0},
                  {"params": list(self.nuisance.values()), "weight_decay": 0.0,  # never decay nuisances
                   "lr": config.nuisance_lr or config.lr}]
        self.opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=config.lr, betas=config.betas)
        self._w0 = [p.detach().clone() for p in pot_params] if config.l2sp else None
        self._pot_params = pot_params
        self._lm_lambda = config.lm_lambda
        if config.optimizer not in ("adamw", "lm"):
            raise ValueError(config.optimizer)
        if config.optimizer == "lm" and (config.l2sp or config.lambda_baseline or config.lambda_quantum):
            log.warning("LM optimizer ignores l2sp/baseline/quantum terms; bound steps with lm_lambda and the trust region")
        self.history: list[dict] = []
        self._batch_rng = np.random.default_rng(config.seed)
        self._queue: list[int] = []

    # ------------------------------------------------------------------
    def verify_initialization(self, systems_positions: list) -> dict:
        """Starting weights reproduce the baseline exactly (same settings)."""
        if self.baseline is None:
            return {"checked": False}
        worst = 0.0
        with torch.no_grad():
            for sys, pos in systems_positions:
                worst = max(worst, abs(float(self.pot.energy(sys, pos)) - float(self.baseline.energy(sys, pos))))
        rec = {"checked": True, "max_abs_energy_diff_eV": worst,
               "base_fingerprint": getattr(self.pot, "base_fingerprint", None),
               "trainable_fingerprint": self.pot.parameter_fingerprint(trainable_only=True)}
        if worst > 1e-9:
            raise RuntimeError(f"initialization parity failed: {worst:.3e} eV")
        return rec

    # ------------------------------------------------------------------
    def _batch(self) -> set[int] | None:
        direct = [i for i, t in enumerate(self.tasks) if isinstance(t, DirectTask)]
        if not self.cfg.batch_size or self.cfg.batch_size >= len(direct):
            return None
        chosen = []
        while len(chosen) < self.cfg.batch_size:
            if not self._queue:
                self._queue = [direct[k] for k in self._batch_rng.permutation(len(direct))]
            i = self._queue.pop(0)
            if i not in chosen:
                chosen.append(i)
        return set(chosen)

    def _evaluate(self, step: int):
        items, surrogate, decisions = [], [], {}
        batch = self._batch()
        for ti, t in enumerate(self.tasks):
            if step % t.every:
                continue
            if batch is not None and isinstance(t, DirectTask) and ti not in batch:
                continue
            if isinstance(t, DirectTask):
                pred = t.predict(self.pot, step)
                items.append((pred, t.get_target(pred)))
                decisions[t.name] = "direct" if pred.valid else f"abstain: {pred.reason}"
                if pred.backward_fn is not None and pred.valid:
                    surrogate.append(pred)  # memory-bounded (e.g. Hessian-vector) backward
                continue
            if isinstance(t, EnsembleTask):
                pred, dec = self._ensemble_predict(t)
                decisions[t.name] = dec
                if pred is None:
                    continue
                items.append((pred, t.get_target(pred)))
                if pred.backward_fn is not None:
                    surrogate.append(pred)
        return items, surrogate, decisions

    def _ensemble_predict(self, t: EnsembleTask):
        sigma = float(t.target.meas_sd.min())
        reasons = []
        for attempt in range(self.cfg.max_sampling_attempts + 1):
            ens = self.cm.ensemble(t, self.pot)
            pred = t.observable.predict(self.pot, ens, t.mode)
            var = pred.variance
            pred_sd = None
            if var is not None:
                v = var.diagonal() if var.dim() == 2 else var.reshape(-1)
                pred_sd = float(v.max().clamp_min(0).sqrt())
            rec = assess(pred.meta.get("estimate"), sigma, self.cfg.controller, prediction_sd=pred_sd)
            if rec.decision == Decision.ACCEPT:
                return pred, f"accept (ESS={rec.ess_fraction:.2f}, n_eff={rec.n_eff:.0f}) after {attempt} action(s)"
            reasons.append(f"{rec.decision.value}: {rec.reason}")
            if attempt == self.cfg.max_sampling_attempts:
                break
            if rec.decision == Decision.REFRESH:
                self.cm.refresh(t, self.pot, rec.reason)
            elif rec.decision == Decision.EXTEND:
                self.cm.extend(t, self.pot, rec.reason)
            else:
                break
        # insufficient sampling after the allowed actions: abstain, do not fit noise
        return None, "abstain after " + " | ".join(reasons)

    def _extra_terms(self):
        from ..losses.anchors import baseline_drift_loss, quantum_anchor_loss

        extra = {}
        if self.cfg.lambda_quantum and self.anchors:
            extra["quantum_anchor"] = self.cfg.lambda_quantum * quantum_anchor_loss(self.pot, self.anchors)
        if self.cfg.lambda_baseline and self.baseline is not None and self.replay:
            drift = sum(baseline_drift_loss(self.pot, self.baseline, sys, pos) for sys, pos in self.replay)
            extra["baseline_drift"] = self.cfg.lambda_baseline * drift / len(self.replay)
        for name, prior in self.priors.items():
            extra[f"prior:{name}"] = prior.nll(self.nuisance[name])
        if self._w0 is not None:
            sq = sum(((p - p0) ** 2).sum() for p, p0 in zip(self._pot_params, self._w0))
            extra["l2sp"] = 0.5 * self.cfg.l2sp * sq
        return extra

    def _overlap_after_update(self) -> tuple[float, str | None]:
        worst, worst_task = 1.0, None
        with torch.no_grad():
            for t in self.tasks:
                if isinstance(t, EnsembleTask) and t.name in self.cm.caches:
                    ens = self.cm.ensemble(t, self.pot)
                    U = ens.energies(self.pot, with_graph=False)
                    la = normalized_log_weights(ens._u_target(U), ens.u_s)
                    frac = float(1.0 / (la.exp() ** 2).sum()) / la.numel()
                    if frac < worst:
                        worst, worst_task = frac, t.name
        return worst, worst_task

    # ------------------------------------------------------------------
    def step(self, step: int) -> dict:
        t0 = time.time()
        self.opt.zero_grad(set_to_none=True)
        items, surrogate, decisions = self._evaluate(step)
        if self.cfg.optimizer == "lm":
            return self._lm_step(step, items, decisions, t0)
        out = combine_objective(items, self.cfg.family_weights, self._extra_terms())
        loss = out.total
        if loss.requires_grad:
            svals = [p.value for p in surrogate if p.value.requires_grad]
            sgrads = torch.autograd.grad(loss, svals, retain_graph=True, allow_unused=True) if svals else []
            loss.backward()
            for p, g in zip([p for p in surrogate if p.value.requires_grad], sgrads):
                if g is not None:
                    p.backward(g)
        params = [p for g in self.opt.param_groups for p in g["params"]]
        grads = [p.grad for p in params if p.grad is not None]
        gnorm = math.sqrt(sum(float((g.double() ** 2).sum()) for g in grads)) if grads else 0.0
        finite = all(bool(torch.isfinite(g).all()) for g in grads)
        rec = {"step": step, "loss": float(loss.detach()), "per_family": out.per_family,
               "skipped": out.skipped, "decisions": decisions, "grad_norm": gnorm,
               "n_params_with_grad": len(grads), "lr": self.opt.param_groups[0]["lr"]}
        if not finite or not grads or gnorm == 0.0:
            rec["update"] = "skipped: " + ("non-finite gradient" if not finite
                                           else "no gradient reached the trainable parameters")
            self._log(rec, t0)
            return rec
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
        snapshot = [p.detach().clone() for p in params]
        opt_state = copy.deepcopy(self.opt.state_dict())
        base_lrs = [g["lr"] for g in self.opt.param_groups]
        shrink = 1.0
        for attempt in range(self.cfg.max_rejections_per_step + 1):
            self.opt.step()
            self._apply_trust_region(params, snapshot)
            ess, task = self._overlap_after_update()
            rec["ess_after_update"] = ess
            rec.update(getattr(self, "_last_trust", {}))
            if ess >= self.cfg.controller.ess_reject:
                rec["update"] = "accepted" if attempt == 0 else f"accepted after {attempt} shrink(s)"
                for g in self.opt.param_groups:
                    g["lr"] *= self.cfg.lr_decay
                break
            with torch.no_grad():
                for p, s in zip(params, snapshot):
                    p.copy_(s)
            self.opt.load_state_dict(opt_state)
            shrink *= self.cfg.lr_shrink_on_reject
            for g, lr0 in zip(self.opt.param_groups, base_lrs):
                g["lr"] = lr0 * shrink  # stays reduced for later steps (conservative)
            rec["update"] = f"rejected: ESS {ess:.3f} on {task}"
        if self.cfg.checkpoint_dir and (step + 1) % self.cfg.checkpoint_every == 0:
            save_training_state(os.path.join(self.cfg.checkpoint_dir, f"step{step + 1:05d}.pt"), self.pot,
                                self.opt, step + 1, {"loss": rec["loss"]})
        self._log(rec, t0)
        return rec

    # ------------------------------------------------------------------
    # Levenberg-Marquardt in observation space
    # ------------------------------------------------------------------
    def _all_params(self):
        return [p for g in self.opt.param_groups for p in g["params"]]

    def _jacobian_rows(self, pred, params) -> torch.Tensor:
        value = pred.value.reshape(-1)
        rows = []
        for k in range(value.numel()):
            for p in params:
                p.grad = None
            e = torch.zeros_like(value)
            e[k] = 1.0
            if pred.backward_fn is not None:
                # potential weights via the deferred (surrogate / HVP) backward ...
                pred.backward_fn(e.reshape(pred.value.shape))
                grads = [p.grad for p in params]
                # ... plus parameters the reported value depends on directly
                # (e.g. a frequency-scale nuisance in the forward model)
                if pred.value.requires_grad:
                    direct = torch.autograd.grad(value[k], params, retain_graph=True, allow_unused=True)
                    grads = [a if b is None else (b if a is None else a + b) for a, b in zip(grads, direct)]
            else:
                grads = torch.autograd.grad(value[k], params, retain_graph=True, allow_unused=True)
            rows.append(torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).to(torch.float64)
                                   for g, p in zip(grads, params)]))
        for p in params:
            p.grad = None
        return torch.stack(rows)

    def _lm_step(self, step, items, decisions, t0) -> dict:
        from ..losses.likelihood import build_covariance

        params = self._all_params()
        res, jac, per_family, skipped = [], [], {}, {}
        fam_n: dict[str, int] = {}
        for pred, tgt in items:
            if pred.valid:
                fam_n[tgt.family] = fam_n.get(tgt.family, 0) + len(tgt.observation_ids)
        for pred, tgt in items:
            key = ",".join(tgt.observation_ids)
            if not pred.valid:
                skipped[key] = pred.reason or "invalid prediction"
                continue
            extra = pred.variance
            if extra is not None and extra.dim() < 2:
                extra = extra.reshape(-1)
            cov = build_covariance(tgt.meas_sd, None if extra is None else extra.detach(),
                                   tgt.correlation_groups, tgt.correlation_rho)
            L = torch.linalg.cholesky(cov)
            w = self.cfg.family_weights.get(tgt.family, FamilyWeight())
            scale = math.sqrt(w.priority / max(fam_n[tgt.family], 1)) if w.normalization != "none" else 1.0
            r = torch.linalg.solve_triangular(L, (pred.value.detach().reshape(-1).to(torch.float64)
                                                  - tgt.value.to(torch.float64)).reshape(-1, 1), upper=False)
            J = torch.linalg.solve_triangular(L, self._jacobian_rows(pred, params), upper=False)
            res.append(scale * r.reshape(-1))
            jac.append(scale * J)
            per_family[tgt.family] = per_family.get(tgt.family, 0.0) + float(0.5 * (scale * r).pow(2).sum())
        offsets = [0]
        for p in params:
            offsets.append(offsets[-1] + p.numel())
        for name, prior in self.priors.items():  # nuisance priors as residual rows
            phi = self.nuisance[name]
            idx = next(i for i, p in enumerate(params) if p is phi)
            row = torch.zeros(offsets[-1], dtype=torch.float64)
            row[offsets[idx]:offsets[idx + 1]] = 1.0 / prior.sd
            res.append(((phi.detach().reshape(-1).to(torch.float64) - prior.mean) / prior.sd))
            jac.append(row.reshape(1, -1))
            per_family[f"prior:{name}"] = float(0.5 * res[-1].pow(2).sum())
        rec = {"step": step, "optimizer": "lm", "per_family": per_family, "skipped": skipped,
               "decisions": decisions, "lr": self._lm_lambda}
        if not res:
            rec.update(loss=0.0, grad_norm=0.0, n_params_with_grad=0, update="skipped: no valid predictions")
            self._log(rec, t0)
            return rec
        r = torch.cat(res)
        J = torch.cat(jac)
        rec["loss"] = float(0.5 * (r**2).sum())
        rec["grad_norm"] = float((J.T @ r).norm())
        rec["n_params_with_grad"] = int((J.abs().sum(0) > 0).sum())
        if not bool(torch.isfinite(J).all()) or rec["grad_norm"] == 0.0:
            rec["update"] = "skipped: non-finite or zero Jacobian"
            self._log(rec, t0)
            return rec
        snapshot = [p.detach().clone() for p in params]
        JJt = J @ J.T
        n = JJt.shape[0]
        lam = self._lm_lambda * float(torch.trace(JJt)) / n + 1e-12
        z = torch.linalg.solve(JJt + lam * torch.eye(n, dtype=torch.float64), r)
        dw = -(J.T @ z)
        rec["lm_lambda"] = lam
        rec["gn_step_norm"] = float(dw.norm())
        alpha = 1.0
        for attempt in range(self.cfg.lm_max_backtracks + 1):
            with torch.no_grad():
                for i, p in enumerate(params):
                    p.copy_(snapshot[i] + alpha * dw[offsets[i]:offsets[i + 1]].reshape(p.shape).to(p.dtype))
            self._apply_trust_region(params, snapshot)
            ess, task = self._overlap_after_update()
            rec["ess_after_update"] = ess
            rec["step_fraction"] = alpha
            if ess >= self.cfg.controller.ess_reject:
                rec["update"] = "accepted" if attempt == 0 else f"accepted at {alpha:.3g} x Gauss-Newton step"
                break
            with torch.no_grad():
                for p, s0 in zip(params, snapshot):
                    p.copy_(s0)
            alpha *= 0.5
            rec["update"] = f"rejected: ESS {ess:.3f} on {task} even at {alpha * 2:.3g} x step"
        if self.cfg.checkpoint_dir and (step + 1) % self.cfg.checkpoint_every == 0:
            save_training_state(os.path.join(self.cfg.checkpoint_dir, f"step{step + 1:05d}.pt"), self.pot,
                                None, step + 1, {"loss": rec["loss"], "lm_lambda": rec.get("lm_lambda")})
        self._log(rec, t0)
        return rec

    def _apply_trust_region(self, params, snapshot):
        """Cap ||dw|| / max(||w||, floor); records the proposed and applied relative step."""
        cap = self.cfg.max_relative_update
        with torch.no_grad():
            dn = math.sqrt(sum(float(((p - s).double() ** 2).sum()) for p, s in zip(params, snapshot)))
            wn = max(math.sqrt(sum(float((s.double() ** 2).sum()) for s in snapshot)),
                     self.cfg.trust_region_norm_floor)
            scale = 1.0
            if cap and dn > cap * wn:
                scale = cap * wn / dn
                for p, s in zip(params, snapshot):
                    p.copy_(s + scale * (p - s))
        self._last_trust = {"proposed_rel_step": dn / wn, "applied_rel_step": scale * dn / wn,
                            "trust_scale": scale}

    def _log(self, rec, t0):
        rec["wall_s"] = time.time() - t0
        rec["cache_events"] = [asdict(e) for e in self.cm.events[-5:]]
        self.history.append(rec)
        if self.cfg.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.cfg.log_path)), exist_ok=True)
            with open(self.cfg.log_path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        log.info("step %d loss %.4g |g| %.3g %s", rec["step"], rec["loss"], rec["grad_norm"], rec.get("update"))

    def fit(self) -> list[dict]:
        for s in range(self.cfg.steps):
            self.step(s)
        return self.history
