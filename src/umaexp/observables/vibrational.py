r"""Harmonic vibrational analysis with gradients into UMA's weights.

Forward model
-------------
    H_w  = \nabla_R^2 U_w(R*)                    Cartesian Hessian (eV/A^2)
    D_w  = M^{-1/2} H_w M^{-1/2}                 mass-weighted (eV/(A^2 u))
    P    = I - X (X^T X)^{-1} X^T                removes rigid translations and
                                                 rotations (X: mass-weighted basis)
    D'   = P D_w P + c (I - P)                   external modes shifted to c >> max eig
    nu_i = sign(lambda_i) sqrt(|lambda_i|) / (2 pi c_light)    lowest 3N-6 (5) eigenvalues

Why the shift instead of eigen-decomposing P: P has a (3N-6)-fold degenerate
eigenvalue, and eigenvector derivatives through degenerate spectra are
undefined. D' has the exact vibrational spectrum of P D P plus the constant c
on the external subspace, and only *eigenvalue* derivatives are used, for
which PyTorch's eigh backward (V diag(g) V^T) needs no eigen-gaps. Losses must
be symmetric within degenerate sets (we average degenerate groups).

Geometry response
-----------------
Frequencies are evaluated at the optimized geometry R*(w), grad U_w(R*) = 0.
By the implicit function theorem dR*/dw = -H^+ d(grad U)/dw. We attach it with
a detached Newton step, R(w) = R* - H^+ g(R*; w): its value is R* (g = 0) and
its derivative is exactly the implicit one. The fixed-geometry approximation
(`geometry_response="fixed"`) drops this term and is labeled as such.

Comparison with experiment
--------------------------
Experimental fundamentals include anharmonicity; harmonic frequencies do not.
Nuclear quantum effects enter as zero-point averaging and anharmonic shifts.
Supported target kinds:
  harmonic               experimentally derived omega_e (evaluated quantity)
  fundamental_scaled     nu = s * omega, s a nuisance with an informative prior
  fundamental_corrected  nu = omega + Delta_anh (Delta from a reference
                         anharmonic calculation, carried with its uncertainty)
plus a relative model-discrepancy variance. Imaginary modes are reported and
make the prediction invalid (abstain); they are never clamped away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment, minimize

from .. import units
from ..potential.base import Potential
from ..potential.derivatives import hessian, hessian_quadratic_form_backward
from ..systems.system import System
from .base import Prediction

# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def principal_moments(pos: torch.Tensor, masses: torch.Tensor):
    m = masses[:, None]
    x = pos - (m * pos).sum(0) / masses.sum()
    inertia = torch.eye(3, dtype=pos.dtype) * (m * x * x).sum() - (m * x).T @ x
    return torch.linalg.eigh(inertia.detach())


def is_linear(pos: torch.Tensor, masses: torch.Tensor, rel_tol: float = 1e-5) -> bool:
    if pos.shape[0] == 2:
        return True
    evals, _ = principal_moments(pos, masses)
    return bool(evals[0] < rel_tol * evals[-1])


def external_basis(pos: torch.Tensor, masses: torch.Tensor, mass_weighted: bool = True,
                   linear: bool | None = None) -> torch.Tensor:
    """(3N, k) raw basis of rigid translations and rotations (not orthonormal).

    Differentiable in positions. For linear molecules the rotation about the
    molecular axis vanishes; the axis direction is taken from the (detached)
    inertia tensor, which is fixed by symmetry at linear stationary points.
    """
    n = pos.shape[0]
    w = masses.sqrt() if mass_weighted else torch.ones_like(masses)
    com = (masses[:, None] * pos).sum(0) / masses.sum()
    x = pos - com
    cols = []
    eye = torch.eye(3, dtype=pos.dtype, device=pos.device)
    for a in range(3):
        cols.append((w[:, None] * eye[a].expand(n, 3)).reshape(-1))
    if n == 1:
        return torch.stack(cols, 1)
    if linear is None:
        linear = is_linear(pos.detach(), masses)
    if linear:
        _, vecs = principal_moments(pos, masses)
        axes = [vecs[:, 1], vecs[:, 2]]  # perpendicular to the molecular axis
    else:
        axes = [eye[0], eye[1], eye[2]]
    for ax in axes:
        rot = torch.linalg.cross(ax.expand(n, 3), x, dim=-1)
        cols.append((w[:, None] * rot).reshape(-1))
    return torch.stack(cols, 1)


def external_projector(pos, masses, mass_weighted=True, linear=None):
    X = external_basis(pos, masses, mass_weighted, linear)
    G = X.T @ X
    return X @ torch.linalg.solve(G, X.T), X.shape[1]


def optimize_geometry(
    potential: Potential,
    system: System,
    positions: torch.Tensor | None = None,
    fmax: float = 1e-6,
    max_steps: int = 5000,
    newton_steps: int = 4,
) -> tuple[torch.Tensor, dict]:
    """Detached local minimization: L-BFGS followed by projected Newton steps.

    Returns positions (float64, no graph) and diagnostics. Tight convergence
    matters: residual forces contaminate low frequencies.
    """
    x0 = (system.positions_tensor() if positions is None else positions.detach()).to(torch.float64)
    shape = x0.shape

    def fun(flat):
        x = torch.tensor(flat.reshape(shape), dtype=torch.float64)
        e, f = potential.energy_and_forces(system, x)
        return float(e), (-f).reshape(-1).numpy().astype(np.float64)

    res = minimize(fun, x0.reshape(-1).numpy(), jac=True, method="L-BFGS-B",
                   options={"maxiter": max_steps, "gtol": fmax * 0.1, "ftol": 1e-15, "maxcor": 50})
    x = torch.tensor(res.x.reshape(shape), dtype=torch.float64)
    masses = system.mass_tensor()
    info = {"lbfgs_iterations": int(res.nit), "lbfgs_message": str(res.message)}
    for _ in range(newton_steps):
        _, f = potential.energy_and_forces(system, x)
        if float(f.abs().max()) < fmax:
            break
        H = hessian(potential, system, x, create_graph=False).detach()
        step = solve_internal(H, -f.reshape(-1), x, masses)
        x = x - step.reshape(shape)
    _, f = potential.energy_and_forces(system, x)
    info["max_force"] = float(f.abs().max())
    info["converged"] = info["max_force"] < fmax
    return x, info


def solve_internal(H: torch.Tensor, rhs: torch.Tensor, pos: torch.Tensor, masses: torch.Tensor):
    """Solve H y = rhs on the internal (non-rigid) subspace, Cartesian coordinates.

    H is singular along rigid-body motions; rhs (a gradient) has no rigid
    components by translational/rotational invariance, so the solution is
    unique in the internal subspace.
    """
    P_ext, _ = external_projector(pos.detach(), masses, mass_weighted=False)
    P = torch.eye(H.shape[0], dtype=H.dtype) - P_ext
    shift = float(H.detach().abs().max()) + 1.0
    A = P @ H @ P + shift * P_ext
    return P @ torch.linalg.solve(A, P @ rhs)


def attach_implicit_geometry(potential, system, x_star, H_star=None):
    """R(w) = R* - H^+ grad U_w(R*): value R*, derivative dR*/dw (implicit)."""
    masses = system.mass_tensor()
    x = x_star.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        e = potential.energy(system, x)
        (g,) = torch.autograd.grad(e, x, create_graph=True)
    if H_star is None:
        H_star = hessian(potential, system, x_star.detach(), create_graph=False).detach()
    P_ext, _ = external_projector(x_star.detach(), masses, mass_weighted=False)
    P = torch.eye(H_star.shape[0], dtype=H_star.dtype) - P_ext
    shift = float(H_star.abs().max()) + 1.0
    A_inv = torch.linalg.inv(P @ H_star @ P + shift * P_ext)
    dx = P @ A_inv @ P @ g.reshape(-1)
    return x_star.detach() - dx.reshape(x_star.shape)


# ---------------------------------------------------------------------------
# normal modes
# ---------------------------------------------------------------------------


@dataclass
class NormalModes:
    wavenumbers: torch.Tensor  # (n_vib,), signed: negative = imaginary; graph kept
    eigenvalues: torch.Tensor  # (n_vib,) eV/(A^2 u)
    modes: torch.Tensor  # (3N, n_vib) mass-weighted eigenvectors, DETACHED
    n_external: int
    linear: bool
    hessian: torch.Tensor  # Cartesian Hessian used (graph kept if requested)


def harmonic_modes(
    potential: Potential,
    system: System,
    positions: torch.Tensor,
    create_graph: bool = True,
    hessian_method: str = "rows",
) -> NormalModes:
    masses = system.mass_tensor(dtype=torch.float64)
    H = hessian(potential, system, positions, create_graph=create_graph, method=hessian_method)
    return modes_from_hessian(H, positions, masses)


def modes_from_hessian(H: torch.Tensor, positions: torch.Tensor, masses: torch.Tensor) -> NormalModes:
    n = positions.shape[0]
    linear = is_linear(positions.detach(), masses)
    inv_sqrt_m = masses.rsqrt().repeat_interleave(3)
    D = inv_sqrt_m[:, None] * H * inv_sqrt_m[None, :]
    P_ext, k = external_projector(positions, masses, mass_weighted=True, linear=linear)
    P = torch.eye(3 * n, dtype=H.dtype) - P_ext
    c = float(torch.linalg.matrix_norm(D.detach())) * 4.0 + 10.0
    Dp = P @ D @ P + c * P_ext
    Dp = 0.5 * (Dp + Dp.T)
    evals, evecs = torch.linalg.eigh(Dp)
    n_vib = 3 * n - k
    lam = evals[:n_vib]
    nu = torch.sign(lam) * torch.sqrt(lam.abs() + 1e-30) * units.EIGVAL_TO_WAVENUMBER
    return NormalModes(nu, lam, evecs[:, :n_vib].detach(), k, linear, H)


# ---------------------------------------------------------------------------
# mode assignment and tracking (discrete; never differentiated)
# ---------------------------------------------------------------------------


@dataclass
class ExperimentalMode:
    label: str
    value: float  # cm^-1
    sigma: float  # cm^-1 (1 standard uncertainty)
    degeneracy: int = 1
    species: str | None = None
    kind: str = "fundamental"  # or "harmonic"
    anharmonic_correction: float | None = None  # cm^-1, for fundamental_corrected
    anharmonic_sigma: float | None = None
    number: int | None = None  # source mode number (spectroscopic numbering within species)
    usable: bool = True  # False: occupies its slot in symmetry assignment but is never scored


@dataclass
class ModeMatch:
    exp_index: int
    comp_indices: list[int]
    ambiguous: bool = False
    note: str = ""


def degenerate_groups(nu: np.ndarray, abs_tol: float = 0.5, rel_tol: float = 2e-4) -> list[list[int]]:
    order = np.argsort(nu)
    groups: list[list[int]] = []
    for i in order:
        if groups and abs(nu[i] - nu[groups[-1][-1]]) <= max(abs_tol, rel_tol * abs(nu[i])):
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return groups


def species_degeneracy(species: str | None) -> int:
    """Degeneracy from a Mulliken symbol: e/pi -> 2, t/f -> 3, else 1."""
    if not species:
        return 1
    s = species.strip().lower()
    if s.startswith(("e", "pi", "π")):
        return 2
    if s.startswith(("t", "f")):
        return 3
    return 1


def assign_modes(
    nu: np.ndarray,
    exp_modes: list[ExperimentalMode],
    strategy: str = "degeneracy_sorted",
    explicit: dict[int, list[int]] | None = None,
    suspicious_rel: float = 0.15,
) -> list[ModeMatch]:
    """Map experimental fundamentals to computed modes.

    degeneracy_sorted: computed degenerate groups of size d are matched in
      frequency order to experimental modes of degeneracy d. Assumes no
      crossing between model and experiment ordering within a degeneracy
      class; the result must be reviewed once at the baseline and is then
      frozen and tracked by eigenvector overlap during training.
    explicit: curated {exp_index: [computed indices]} at the baseline.
    """
    if strategy == "explicit":
        if explicit is None:
            raise ValueError("explicit strategy needs a mapping")
        return [ModeMatch(i, list(c)) for i, c in explicit.items()]
    groups = degenerate_groups(nu)
    matches: list[ModeMatch] = []
    for d in sorted({m.degeneracy for m in exp_modes}):
        comp = sorted([g for g in groups if len(g) == d], key=lambda g: float(np.mean(nu[g])))
        exps = sorted([i for i, m in enumerate(exp_modes) if m.degeneracy == d],
                      key=lambda i: exp_modes[i].value)
        ambiguous = len(comp) != len(exps)
        if strategy != "degeneracy_sorted":
            raise ValueError(strategy)
        if ambiguous and len(comp) > len(exps):
            # more computed than observed: match by minimal |difference| instead
            cost = np.array([[abs(float(np.mean(nu[g])) - exp_modes[i].value) for g in comp] for i in exps])
            r, c = linear_sum_assignment(cost)
            pairs = [(exps[a], comp[b]) for a, b in zip(r, c)]
        else:
            pairs = list(zip(exps, comp))
        for ei, g in pairs:
            val = float(np.mean(nu[g]))
            rel = abs(val - exp_modes[ei].value) / exp_modes[ei].value
            note = "suspicious (>{:.0f}% off)".format(100 * suspicious_rel) if rel > suspicious_rel else ""
            matches.append(ModeMatch(ei, list(g), ambiguous=ambiguous, note=note))
    return sorted(matches, key=lambda m: m.exp_index)


def track_modes(reference_modes: np.ndarray, matches: list[ModeMatch], new_modes: np.ndarray,
                new_nu: np.ndarray, new_groups: list[list[int]] | None = None) -> list[ModeMatch]:
    """Re-identify matched modes after an update by subspace projection (Hungarian over modes).

    reference_modes/new_modes: (3N, n_vib) mass-weighted eigenvectors.
    A match of dimension d takes the d new modes with the largest weight in its reference subspace,
    one-to-one across matches; overlap = (sum of those squared projections) / d.
    """
    # DEVIATIONS.md DV-03: assignment is over individual modes, not over a grouping of the new modes.
    # A grouping (frequency window or irrep detection) that splits a degenerate pair, or differs from
    # the base one, used to turn a full-projection match into overlap 0 through a size penalty.
    # new_nu and new_groups are kept for the call signature and are not used.
    if not matches:
        return []
    rows = [a for a, m in enumerate(matches) for _ in m.comp_indices]
    weight = np.stack([((reference_modes[:, matches[a].comp_indices].T @ new_modes) ** 2).sum(axis=0) for a in rows])
    r, c = linear_sum_assignment(-weight)
    out = []
    for a, m in enumerate(matches):
        picked = sorted(int(j) for i, j in zip(r, c) if rows[i] == a)
        overlap = float(sum(weight[i, j] for i, j in zip(r, c) if rows[i] == a)) / len(m.comp_indices)
        out.append(ModeMatch(m.exp_index, picked, ambiguous=overlap < 0.8, note=f"overlap={overlap:.3f}"))
    return sorted(out, key=lambda m: m.exp_index)


# ---------------------------------------------------------------------------
# forward model to measured quantities
# ---------------------------------------------------------------------------


@dataclass
class VibrationalForwardModel:
    """Map harmonic wavenumbers to what was measured.

    kind: "harmonic" | "fundamental_scaled" | "fundamental_corrected"
    scale: nuisance scale factor s (a tensor so it can be a calibrated
      parameter phi with a prior in the loss); None -> 1.
    rel_model_sd: relative model-discrepancy SD added in quadrature.
    abs_model_sd: absolute model-discrepancy SD (cm^-1) added in quadrature;
      with rel_model_sd = 0 the per-mode weights are near-uniform, so the
      objective tracks an unweighted RMSE (Vib-v1, critic M2).
    """

    kind: str = "fundamental_scaled"
    scale: torch.Tensor | None = None
    rel_model_sd: float = 0.015
    abs_model_sd: float = 0.0
    min_abs_wavenumber: float = 50.0  # soft modes below this are excluded

    def apply(self, omega: torch.Tensor, modes: list[ExperimentalMode]):
        if self.kind == "harmonic":
            pred = omega
            extra = torch.zeros_like(omega)
        elif self.kind == "fundamental_scaled":
            s = self.scale if self.scale is not None else torch.tensor(1.0, dtype=omega.dtype)
            pred = s * omega
            extra = torch.zeros_like(omega)
        elif self.kind == "fundamental_corrected":
            missing = [m.label for m in modes if m.anharmonic_correction is None]
            if missing:  # never default a missing correction to 0 (critic C3)
                raise ValueError(f"fundamental_corrected: no anharmonic correction for {missing}")
            corr = torch.tensor([m.anharmonic_correction for m in modes], dtype=omega.dtype)
            csd = torch.tensor([m.anharmonic_sigma or 0.0 for m in modes], dtype=omega.dtype)
            pred = omega + corr
            extra = csd**2
        else:
            raise ValueError(self.kind)
        # stop-gradient: the discrepancy variance is a constant within a step, so
        # the fit cannot reduce its loss by shifting frequencies to inflate
        # its own error bars
        var = (self.rel_model_sd * omega.detach()) ** 2 + self.abs_model_sd**2 + extra
        return pred, var


@dataclass
class VibrationalConfig:
    geometry_response: str = "implicit"  # or "fixed"
    fmax: float = 1e-6
    max_opt_steps: int = 5000
    # "graph": keep the full create_graph Hessian (small molecules);
    # "hvp": detached Hessian + memory-bounded quadratic-form backward
    #        (same gradient, one double-backward graph at a time)
    memory_mode: str = "graph"
    assignment: str = "degeneracy_sorted"  # or "symmetry" (Vib-v1: value-free, see observables/symmetry.py)
    point_group: str | None = None  # source point-group label, checked against the detected group
    hessian_method: str = "rows"
    min_track_overlap: float = 0.0  # Vib-v1: 0.5; below it the prediction is invalid (abstains for the step)
    forward: VibrationalForwardModel = field(default_factory=VibrationalForwardModel)


class VibrationalObservable:
    """Stateful predictor for one system: keeps the frozen baseline assignment."""

    def __init__(self, system: System, exp_modes: list[ExperimentalMode], config: VibrationalConfig | None = None,
                 explicit_assignment: dict[int, list[int]] | None = None):
        self.system = system
        self.exp_modes = exp_modes
        self.config = config or VibrationalConfig()
        self.explicit = explicit_assignment
        self.reference_modes: np.ndarray | None = None
        self.matches: list[ModeMatch] | None = None
        self.base_matches: list[ModeMatch] | None = None  # fixed anchor: assignment on the base model
        self.last_geometry: torch.Tensor | None = None
        self.assignment_info: dict = {}

    def _irrep_groups(self, x_star: torch.Tensor, nm: NormalModes) -> list[list[int]]:
        from .symmetry import find_point_group, symmetry_mode_groups

        pg = find_point_group(x_star.detach().numpy(), np.asarray(self.system.masses, float),
                              np.asarray(self.system.atomic_numbers))
        groups = symmetry_mode_groups(pg, nm.eigenvalues.detach().numpy(), nm.modes.numpy(),
                                      nm.wavenumbers.detach().numpy())
        return [g.indices for g in groups]

    def _symmetry_assignment(self, x_star: torch.Tensor, nm: NormalModes) -> list[ModeMatch]:
        """Value-free assignment on the full table; returns matches for USABLE modes only.

        Every usable mode must be matched or belong to a class whose counts
        disagree (recorded, then dropped for all models); anything else raises.
        """
        from .symmetry import assign_by_symmetry, find_point_group, symmetry_mode_groups

        em = self.exp_modes
        if any(m.species is None or m.number is None for m in em):
            raise ValueError("symmetry assignment needs species and mode number for every tabulated mode")
        pg = find_point_group(x_star.detach().numpy(), np.asarray(self.system.masses, float),
                              np.asarray(self.system.atomic_numbers))
        groups = symmetry_mode_groups(pg, nm.eigenvalues.detach().numpy(), nm.modes.numpy(),
                                      nm.wavenumbers.detach().numpy())
        res = assign_by_symmetry(pg, groups, [m.species for m in em], [m.number for m in em], [m.value for m in em],
                                 expected_point_group=self.config.point_group)
        matched = {e for e, _ in res.matches}
        lost = [i for i, m in enumerate(em) if m.usable and i not in matched and i not in res.excluded_exp]
        if lost and res.point_group_ok:
            raise RuntimeError(f"{self.system.system_id}: usable modes {lost} left unmatched by symmetry assignment")
        self.assignment_info = {"point_group_order": res.point_group_order, "point_group_ok": res.point_group_ok,
                                "merged_classes": res.merged_classes, "notes": res.notes,
                                "excluded_usable": [em[i].label for i in res.excluded_exp if em[i].usable]}
        return [ModeMatch(e, list(idx), ambiguous=any(em[e].species in c for c in res.merged_classes),
                          note="merged class" if any(em[e].species in c for c in res.merged_classes) else "")
                for e, idx in res.matches if em[e].usable]

    def predict(self, potential: Potential) -> Prediction:
        cfg = self.config
        start = self.last_geometry if self.last_geometry is not None else None
        x_star, opt = optimize_geometry(potential, self.system, start, fmax=cfg.fmax, max_steps=cfg.max_opt_steps)
        self.last_geometry = x_star.detach()
        if cfg.geometry_response == "implicit":
            H_star = hessian(potential, self.system, x_star, create_graph=False).detach()
            x = attach_implicit_geometry(potential, self.system, x_star, H_star)
        elif cfg.geometry_response == "fixed":
            x = x_star.detach()
        else:
            raise ValueError(cfg.geometry_response)
        hvp = cfg.memory_mode == "hvp"
        nm = harmonic_modes(potential, self.system, x.detach() if hvp else x, create_graph=not hvp,
                            hessian_method=cfg.hessian_method)
        nu_np = nm.wavenumbers.detach().numpy()
        modes_np = nm.modes.numpy()
        if self.base_matches is None:
            if cfg.assignment == "symmetry":
                self.base_matches = self._symmetry_assignment(x_star, nm)
            else:
                self.base_matches = assign_modes(nu_np, self.exp_modes, cfg.assignment, self.explicit)
            self.reference_modes = modes_np
            self.matches = self.base_matches
        else:  # always from the fixed base assignment, never from the previous step's indices
            groups = self._irrep_groups(x_star, nm) if cfg.assignment == "symmetry" else None
            self.matches = track_modes(self.reference_modes, self.base_matches, modes_np, nu_np, groups)
        overlaps = [float(m.note.split("=")[1]) for m in self.matches if m.note.startswith("overlap=")]
        if overlaps and min(overlaps) < cfg.min_track_overlap:
            return Prediction(value=torch.zeros(0, dtype=torch.float64), variance=torch.zeros(0, dtype=torch.float64),
                              valid=False, reason=f"mode tracking overlap {min(overlaps):.2f} < {cfg.min_track_overlap}",
                              diagnostics={**opt, "assignment_info": self.assignment_info},
                              meta={"exp_indices": [], "labels": []})
        if not self.matches:
            return Prediction(value=torch.zeros(0, dtype=torch.float64), variance=torch.zeros(0, dtype=torch.float64),
                              valid=False, reason="; ".join(self.assignment_info.get("notes", [])) or "no scored modes",
                              diagnostics={**opt, "assignment_info": self.assignment_info},
                              meta={"exp_indices": [], "labels": []})
        n_imag = int((nu_np < 0).sum())
        used = [self.exp_modes[m.exp_index] for m in self.matches]
        backward_fn = None
        if hvp:
            omega0 = torch.stack([nm.wavenumbers[m.comp_indices].mean() for m in self.matches]).detach()
            leaf = omega0.clone().requires_grad_(True)
            with torch.enable_grad():
                pred, var = cfg.forward.apply(leaf, used)
            backward_fn = self._hvp_backward(potential, x, nm, omega0, used)
        else:
            omega = torch.stack([nm.wavenumbers[m.comp_indices].mean() for m in self.matches])
            pred, var = cfg.forward.apply(omega, used)
        soft = [m.exp_index for m in self.matches
                if abs(float(nu_np[m.comp_indices].mean())) < cfg.forward.min_abs_wavenumber]
        valid = opt["converged"] and n_imag == 0 and not soft
        reason = []
        if not opt["converged"]:
            reason.append(f"geometry not converged (max|F|={opt['max_force']:.2e})")
        if n_imag:
            reason.append(f"{n_imag} imaginary mode(s) at the optimized geometry")
        if soft:
            reason.append(f"soft modes {soft}")
        return Prediction(
            value=pred,
            variance=var,
            valid=valid,
            reason="; ".join(reason),
            diagnostics={
                **opt,
                "n_imaginary": n_imag,
                "linear": nm.linear,
                "n_external": nm.n_external,
                "all_wavenumbers": nu_np.tolist(),
                "assignment": [(m.exp_index, m.comp_indices, m.ambiguous, m.note) for m in self.matches],
                "assignment_info": self.assignment_info,
                "geometry_response": cfg.geometry_response,
            },
            meta={"exp_indices": [m.exp_index for m in self.matches],
                  "labels": [self.exp_modes[m.exp_index].label for m in self.matches]},
            backward_fn=backward_fn,
        )

    def _hvp_backward(self, potential, x, nm: NormalModes, omega0, used):
        """dL/dw via G = dL/dH = sum_k g_lambda_k u_k u_k^T (u_k = M^-1/2 v_k).

        At a stationary point the rigid-body projector does not change the
        vibrational eigenvalues to first order (H annihilates rigid motions),
        so d lambda_k = u_k^T dH u_k.
        """
        matches = list(self.matches)
        fwd = self.config.forward
        inv_sqrt_m = self.system.mass_tensor().rsqrt().repeat_interleave(3)
        lam = nm.eigenvalues.detach()
        vecs = nm.modes  # (3N, n_vib) mass-weighted, detached

        def backward_fn(g):
            w = omega0.clone().requires_grad_(True)
            with torch.enable_grad():
                p, _ = fwd.apply(w, used)
                # gradient w.r.t. omega only: nuisance parameters (e.g. the scale
                # factor) already received theirs from the caller's loss.backward()
                (g_omega,) = torch.autograd.grad(p, w, grad_outputs=g.to(p.dtype).reshape(p.shape))
            dirs, coefs = [], []
            for j, m in enumerate(matches):
                for k in m.comp_indices:
                    lk = float(lam[k])
                    dnu_dlam = units.EIGVAL_TO_WAVENUMBER / (2.0 * math.sqrt(abs(lk)))
                    coefs.append(float(g_omega[j]) / len(m.comp_indices) * dnu_dlam)
                    dirs.append((vecs[:, k] * inv_sqrt_m).reshape(-1, 3))
            if dirs:
                hessian_quadratic_form_backward(potential, self.system, x, torch.stack(dirs),
                                                torch.tensor(coefs, dtype=torch.float64))

        return backward_fn

    def targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.matches is None:
            raise RuntimeError("predict() first")
        y = torch.tensor([self.exp_modes[m.exp_index].value for m in self.matches], dtype=torch.float64)
        s = torch.tensor([self.exp_modes[m.exp_index].sigma for m in self.matches], dtype=torch.float64)
        return y, s**2
