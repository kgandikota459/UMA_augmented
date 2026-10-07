"""Jacobian of (density, U/N) w.r.t. (ln eps, ln sigma) at the LJ pilot start; exact Gauss-Newton step vs truth."""
import math, os, time, torch, numpy as np, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_equilibrium_pilot import EnergyPerAtom
from umaexp.observables.density import DensityObservable
from umaexp.potential.toy import LennardJones, lj_argon_system
from umaexp.reweighting import ReweightedEnsemble
from umaexp.sampling import SamplingProtocol, ThermoState, run_sampling
system = lj_argon_system(n_side=5, density_g_cm3=1.35)
state = ThermoState(85.0, 50.0, "NPT")
pot = LennardJones(epsilon=0.0104 * 0.85, sigma=3.40 * 1.03, cutoff=8.0)
t = time.time()
cache = run_sampling(pot, system, state, SamplingProtocol(dt_fs=10.0, friction_per_ps=2.0, n_equilibration=4000,
                     n_production=60000, stride=20, barostat_every=10, barostat_max_dlnv=0.01, seed=5))
ens = ReweightedEnsemble(cache, system)
preds = [DensityObservable(system).predict(pot, ens, "graph"), EnergyPerAtom(system.n_atoms).predict(pot, ens, "graph")]
J = []
for p in preds:
    g = torch.autograd.grad(p.value.sum(), [pot.log_eps, pot.log_sigma], retain_graph=True)
    J.append([float(x) for x in g])
J = np.array(J)
vals = np.array([float(p.value.detach()) for p in preds])
sds = np.array([math.sqrt(float(p.variance.reshape(-1)[0])) for p in preds])
target = np.array([1.200697, -0.037912])
r = vals - target
dw = np.linalg.solve(J, -r)
truth = np.array([math.log(1/0.85), math.log(1/1.03)])
print(f"start: rho={vals[0]:.4f}±{sds[0]:.4f}  U/N={vals[1]:.5f}±{sds[1]:.5f}  (n_eff={preds[0].diagnostics['n_eff']:.0f}, {cache.n_frames} frames, {time.time()-t:.0f}s)")
print("J = d(rho, U/N)/d(log eps, log sigma):\n", np.round(J, 4))
print("Gauss-Newton step (log eps, log sigma):", np.round(dw, 4), " vs truth offset:", np.round(truth, 4))
print("cond(J) = %.1f" % np.linalg.cond(J))
