"""The DOLFINx backend on several processes: the same statistics as on one, at every
step, for a periodic case (PeriodicBubble2D), one with unequal properties, walls and
periodic sides (Bubble2D) and one with walls all round (LaplaceBubble2D). Periodic dofs
are folded across process boundaries here, the part of the backend most likely to go
wrong in parallel."""
import importlib.util

import numpy as np
import pytest

from conftest import BUBBLE, QUIET, load_tdata, require_mpi, run_twoasis

pytestmark = pytest.mark.skipif(importlib.util.find_spec("dolfinx") is None,
                                reason="needs DOLFINx")

CASES = {
    "PeriodicBubble2D": dict(BUBBLE, T=0.01, checkpoint=1000000),
    "Bubble2D": dict(QUIET, problem="Bubble2D", res=1/32, epsilon=0.04, dt=0.002, T=0.01),
    "LaplaceBubble2D": dict(QUIET, problem="LaplaceBubble2D", res=1/32, epsilon=0.04,
                            dt=0.002, T=0.01),
}
NPROCS = 3


@pytest.mark.parametrize("case", CASES)
def test_same_statistics_on_several_processes(tmp_path, monkeypatch, case):
    require_mpi(NPROCS)
    monkeypatch.delenv("TWOASIS_BACKEND", raising=False)
    tds = []
    for n in (1, NPROCS):
        wd = tmp_path / "np{}".format(n)
        wd.mkdir()
        run_twoasis(wd, nprocs=n, solver="BDF", backend="dolfinx", telemetry=True,
                    **CASES[case])
        tds.append(load_tdata(wd))
        assert len(list(wd.glob("results/data/1/Stats/telemetry.jsonl"))) == 1
    serial, parallel = tds
    assert serial.shape == parallel.shape and len(serial) >= 5
    # The statistics are printed to 8 decimals (LaplaceBubble2D's to 10 digits).
    tol = 1e-6 * np.abs(serial) + 2e-8
    worst = np.max(np.abs(parallel - serial) - tol)
    assert worst <= 0, "{} ranks differ from 1 by up to {:.3g} beyond tolerance".format(
        NPROCS, worst)
