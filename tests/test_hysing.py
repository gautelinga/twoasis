"""The rising-bubble benchmark of Hysing et al. (2009), test case 1, on both backends: the
bubble's centroid height at t = 3, its largest rise velocity and its smallest
circularity inside bands around the reference values (as twoasisx tested them).

S. Hysing et al., Int. J. Numer. Meth. Fluids 60, 1259-1288 (2009), table V:
y_c(3) = 1.0813, max V_c = 0.2417, min circularity = 0.9013. The bands allow for the
diffuse interface (epsilon = 0.02) at res 1/64: 3% on y_c, 5% on the others.

Bubble2D with free-slip side walls (periodic=False), the benchmark's setup; the
quantities are Stats/benchmark.dat's (shared.hysing_quantities). 1500 steps on four
processes: a weekly test. Also on the dual mesh (twoasis/common/dual.py): the flow at
res 1/32, the phase field at 1/64 (phase_refine=1), in the same bands.
"""
import importlib.util

import numpy as np
import pytest

from conftest import QUIET, require_mpi, run_twoasis

REFERENCE = dict(y_c=(1.0813, 0.03), V_c_max=(0.2417, 0.05), circularity_min=(0.9013, 0.05))
BACKENDS = ["dolfin"] + (["dolfinx"] if importlib.util.find_spec("dolfinx") else [])


MESHES = dict(single={}, dual=dict(res=1 / 32, phase_refine=1))


@pytest.mark.slow
@pytest.mark.parametrize("meshes", sorted(MESHES))
@pytest.mark.parametrize("backend", BACKENDS)
def test_hysing_case_1(tmp_path, monkeypatch, backend, meshes):
    require_mpi(4)
    monkeypatch.delenv("TWOASIS_BACKEND", raising=False)
    run_twoasis(tmp_path, nprocs=4, timeout=7200, problem="Bubble2D", periodic=False, hysing_stats=True,
                T=3.0, backend=backend, **dict(QUIET, stat_interval=10), **MESHES[meshes])
    b = np.loadtxt(tmp_path / "results" / "data" / "1" / "Stats" / "benchmark.dat")
    assert b[-1, 1] == pytest.approx(3.0)
    measured = dict(y_c=b[-1, 3], V_c_max=b[:, 4].max(), circularity_min=b[:, 5].min())
    for name, (ref, band) in REFERENCE.items():
        assert abs(measured[name] - ref) <= band * ref, \
            "{} = {:.4f}, outside {:.4f} +- {:.0%}".format(name, measured[name], ref, band)
