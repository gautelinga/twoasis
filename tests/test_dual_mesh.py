"""The dual mesh (phase_refine, twoasis/common/dual.py): the phase field on the flow mesh
refined uniformly, the flow on the flow mesh.

PeriodicBubble2D with the flow mesh at res 1/16 and the phase field at 1/32: the two
backends agree, as do one and three ranks; a checkpoint written on dolfin on one rank
restarts on DOLFINx on two (the fine phase field stored by flow-mesh cell, the fine mesh
made anew) and continues as the uninterrupted run; and the bubble moves as on one mesh
at the fine resolution (the statistics weigh the velocity, moved exactly to the fine
mesh, by the phase field there). Every problem runs on the dual mesh in
test_problems.py; the Hysing benchmark on it in test_hysing.py.
"""
import importlib.util

import numpy as np
import pytest

from conftest import load_tdata, QUIET, require_mpi, run_twoasis

PARAMS = dict(QUIET, problem="PeriodicBubble2D", res=1 / 16, dt=0.002, epsilon=0.05, R=0.25,
              M=0.000125, T=0.02, phase_refine=1)
HAS_DOLFINX = importlib.util.find_spec("dolfinx") is not None
# atol: one unit of the 8th decimal the statistics are printed with
CLOSE = dict(rtol=1e-6, atol=1.1e-8)


def _run(path, monkeypatch, backend, nprocs=1, **params):
    monkeypatch.delenv("TWOASIS_BACKEND", raising=False)
    path.mkdir(parents=True, exist_ok=True)
    run_twoasis(path, nprocs=nprocs, backend=backend, **dict(PARAMS, **params))
    return load_tdata(path)


@pytest.mark.skipif(not HAS_DOLFINX, reason="needs DOLFINx")
def test_backends_agree(tmp_path, monkeypatch):
    a = _run(tmp_path / "dolfin", monkeypatch, "dolfin")
    b = _run(tmp_path / "dolfinx", monkeypatch, "dolfinx")
    assert a.shape == b.shape == (10, a.shape[1])
    np.testing.assert_allclose(a, b, **CLOSE)


@pytest.mark.parametrize("backend", ["dolfin"] + (["dolfinx"] if HAS_DOLFINX else []))
def test_ranks_agree(tmp_path, monkeypatch, backend):
    require_mpi(3)
    a = _run(tmp_path / "1", monkeypatch, backend)
    b = _run(tmp_path / "3", monkeypatch, backend, nprocs=3)
    np.testing.assert_allclose(a, b, **CLOSE)


@pytest.mark.skipif(not HAS_DOLFINX, reason="needs DOLFINx")
def test_restart_on_other_backend_and_ranks(tmp_path, monkeypatch):
    """A dual-mesh checkpoint from dolfin on one rank, at step 6, restarted on DOLFINx on
    two ranks, ends as the uninterrupted run."""
    require_mpi(2)
    whole = _run(tmp_path / "whole", monkeypatch, "dolfin")
    _run(tmp_path / "r", monkeypatch, "dolfin", T=0.012, checkpoint=6)
    restarted = _run(tmp_path / "r", monkeypatch, "dolfinx", nprocs=2,
                     restart_folder="results/data/1/Checkpoint")
    np.testing.assert_allclose(restarted, whole, **CLOSE)


def test_close_to_one_fine_mesh(tmp_path, monkeypatch):
    """The bubble's area, height and velocity (the last three columns) on the dual mesh
    and on one mesh at the phase field's resolution."""
    dual = _run(tmp_path / "dual", monkeypatch, "dolfin")
    fine = _run(tmp_path / "fine", monkeypatch, "dolfin", res=1 / 32, phase_refine=0)
    np.testing.assert_allclose(dual[:, -3:], fine[:, -3:], rtol=1e-3, atol=1e-6)


def test_restart_needs_its_phase_refine(tmp_path, monkeypatch):
    """A dual-mesh checkpoint restarted on one mesh is refused with a message that says
    so (not taken for an old checkpoint of bare vectors)."""
    _run(tmp_path, monkeypatch, "dolfin", T=0.004, checkpoint=2)
    with pytest.raises(AssertionError, match="phase field on the dual mesh"):
        _run(tmp_path, monkeypatch, "dolfin", phase_refine=0,
             restart_folder="results/data/1/Checkpoint")


def test_only_lagrange_functions_move(tmp_path):
    """A flow-mesh Function in an integral with the phase field moves to its mesh when
    it is continuous Lagrange (the nested spaces' prolongation); a DG one is refused."""
    from twoasis.backend import dx, Function, FunctionSpace, UnitSquareMesh
    from twoasis.common import dual
    mesh = UnitSquareMesh(4, 4)
    d = dual.start(mesh, 1)
    try:
        phi = Function(FunctionSpace(d.fine, "CG", 1))
        phi.vector()[:] = 1.0
        f = Function(FunctionSpace(mesh, "CG", 1))
        f.vector()[:] = 1.0
        from twoasis.backend import assemble
        assert assemble(f * phi * dx) == pytest.approx(1.0)
        g = Function(FunctionSpace(mesh, "DG", 1))
        with pytest.raises(ValueError, match="continuous Lagrange"):
            g * phi * dx
    finally:
        dual.start(mesh, 0)
