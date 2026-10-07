import logging

import numpy as np
import pytest
import torch

logging.getLogger("fairchem").setLevel(logging.ERROR)
torch.set_num_threads(max(1, torch.get_num_threads()))


@pytest.fixture
def water_toy():
    """Bent triatomic with harmonic bonds/angle (toy water, not a water model)."""
    from umaexp.potential.toy import ValenceToy
    from umaexp.systems import System, isotopologue_masses

    theta0 = np.deg2rad(104.5)
    r0 = 0.9572
    pos = np.array([[0.0, 0.0, 0.0],
                    [r0, 0.0, 0.0],
                    [r0 * np.cos(theta0), r0 * np.sin(theta0), 0.0]])
    sys = System("toy_H2O", (8, 1, 1), isotopologue_masses((8, 1, 1)), task="toy", spin=0,
                 positions=pos + 0.02)
    pot = ValenceToy(bonds=[(0, 1), (0, 2)], angles=[(1, 0, 2)], kb=[50.0], r0=[r0],
                     ka=[4.0], theta0=[theta0])
    return sys, pot


@pytest.fixture(scope="session")
def uma_random():
    from umaexp.potential import UMAPotential

    refs = torch.zeros(101, dtype=torch.float64)
    refs[1], refs[6], refs[7], refs[8] = -13.6, -1027.0, -1484.0, -2045.0
    return UMAPotential.random_architecture(seed=0, element_references=refs)
