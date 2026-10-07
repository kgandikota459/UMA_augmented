"""umaexp: fine-tuning UMA's weights through experimental observables.

Layers (see docs/DESIGN.md):
  potential   U_w(system, R, h) with UMA as the trainable model
  systems     explicit atomic systems with provenance
  data        typed experimental/reference records, manifests, splits
  observables forward models g_k[U_w, S, c_k; phi_k]
  sampling    MD/MC samplers and sample caches
  reweighting DiffTRe-style estimators, differentiable FEP/BAR/MBAR
  losses      uncertainty-aware likelihoods and the combined objective
  training    the controlled update loop
  validation  derivative, physical and statistical checks
"""

__version__ = "0.1.0"
