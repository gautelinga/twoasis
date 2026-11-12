"""Adelsberger et al.'s 3D rising bubble (test case 1, Bubble3D) on both backends, on a
coarse mesh over the first part of the rise: the largest rise velocity and the smallest
sphericity inside 15% bands around the references (0.355 near t = 0.9; 0.955), the
bubble risen part of the way, and its volume kept. The bands and the window are
twoasisx's coarse gate (res = 1/16, where a P1 phase field resolves the interface;
twoasisx measured 0.304 and 0.903 there).
"""
import importlib.util

import numpy as np
import pytest

from conftest import QUIET, require_mpi, run_twoasis

BACKENDS = ["dolfin"] + (["dolfinx"] if importlib.util.find_spec("dolfinx") else [])


@pytest.mark.slow
@pytest.mark.parametrize("backend", BACKENDS)
def test_adelsberger_case_1_coarse(tmp_path, monkeypatch, backend):
    require_mpi(4)
    monkeypatch.delenv("TWOASIS_BACKEND", raising=False)
    run_twoasis(tmp_path, nprocs=4, timeout=7200, problem="Bubble3D", res=1 / 16, dt=0.002,
                T=0.96, backend=backend, **dict(QUIET, stat_interval=4))
    b = np.loadtxt(tmp_path / "results" / "data" / "1" / "Stats" / "benchmark.dat")
    volume, z_c, v_rise, sphericity = b[:, 2], b[:, 3], b[:, 4], b[:, 5]
    assert 0.30 <= v_rise.max() <= 0.41, "max rise velocity {:.4f}".format(v_rise.max())
    assert 0.81 <= sphericity.min() <= 1.0, "min sphericity {:.4f}".format(sphericity.min())
    assert 0.60 <= z_c[-1] <= 0.85, "z_c at t = 0.96: {:.4f}".format(z_c[-1])
    assert np.abs(volume / volume[0] - 1).max() < 1e-3
