"""The dual mesh (phase_refine, twoasis/common/dual.py): the phase field on the flow mesh
refined uniformly, the flow on the flow mesh.

PeriodicBubble2D with the flow mesh at res 1/16 and the phase field at 1/32: the two
backends agree, as do one and three ranks; a checkpoint written on dolfin on one rank
restarts on DOLFINx on two (the fine phase field stored by flow-mesh cell, the fine mesh
made anew) and continues as the uninterrupted run; and the bubble moves as on one mesh
at the fine resolution (the statistics weigh the velocity, moved exactly to the fine
mesh, by the phase field there); each with the split CH-NS step and with the coupled one.
The coupled step's blocks that couple the meshes are those of one mesh for a phase field
the flow mesh holds, and exchange energy exactly. Every problem runs on the dual mesh in
test_problems.py; the Hysing benchmark on it in test_hysing.py.
"""
import importlib.util

import numpy as np
import pytest

from conftest import load_tdata, once, QUIET, require_mpi, run_twoasis

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


def _baseline(tmp_path_factory, backend, chns):
    """PARAMS on one rank on `backend` with chns_solve=`chns`, run once per session
    (several tests compare with it)."""
    wd = once(tmp_path_factory, "dual_{}_{}".format(backend, chns),
              lambda wd: run_twoasis(wd, backend=backend, chns_solve=chns, **PARAMS))
    return load_tdata(wd)


@pytest.fixture(scope="session", params=["split", "coupled"])
def chns(request):
    """The CH-NS step the dual-mesh runs take: the split one, or the coupled one (its
    coupling blocks assembled on the fine mesh, monolithic.CouplingBlocks)."""
    return request.param


@pytest.fixture(scope="session")
def dual_dolfin(tmp_path_factory, chns):
    return _baseline(tmp_path_factory, "dolfin", chns)


@pytest.fixture(scope="session")
def dual_dolfinx(tmp_path_factory, chns):
    return _baseline(tmp_path_factory, "dolfinx", chns)


@pytest.mark.skipif(not HAS_DOLFINX, reason="needs DOLFINx")
def test_backends_agree(dual_dolfin, dual_dolfinx):
    a, b = dual_dolfin, dual_dolfinx
    assert a.shape == b.shape == (10, a.shape[1])
    np.testing.assert_allclose(a, b, **CLOSE)


@pytest.mark.parametrize("backend", ["dolfin"] + (["dolfinx"] if HAS_DOLFINX else []))
def test_ranks_agree(tmp_path, monkeypatch, request, backend, chns):
    require_mpi(3)
    a = request.getfixturevalue("dual_" + backend)
    b = _run(tmp_path / "3", monkeypatch, backend, nprocs=3, chns_solve=chns)
    np.testing.assert_allclose(a, b, **CLOSE)


@pytest.mark.skipif(not HAS_DOLFINX, reason="needs DOLFINx")
def test_restart_on_other_backend_and_ranks(tmp_path, monkeypatch, dual_dolfin, chns):
    """A dual-mesh checkpoint from dolfin on one rank, at step 6, restarted on DOLFINx on
    two ranks, ends as the uninterrupted run."""
    require_mpi(2)
    _run(tmp_path / "r", monkeypatch, "dolfin", T=0.012, checkpoint=6, chns_solve=chns)
    restarted = _run(tmp_path / "r", monkeypatch, "dolfinx", nprocs=2,
                     restart_folder="results/data/1/Checkpoint")
    np.testing.assert_allclose(restarted, dual_dolfin, **CLOSE)


def test_close_to_one_fine_mesh(tmp_path, monkeypatch, dual_dolfin, chns):
    """The bubble's area, height and velocity (the last three columns) on the dual mesh
    and on one mesh at the phase field's resolution."""
    fine = _run(tmp_path / "fine", monkeypatch, "dolfin", res=1 / 32, phase_refine=0,
                chns_solve=chns)
    np.testing.assert_allclose(dual_dolfin[:, -3:], fine[:, -3:], rtol=1e-3, atol=1e-6)


def test_coupled_direct_is_iterative(tmp_path, monkeypatch, tmp_path_factory):
    """The coupled step on the dual mesh solved directly (its nest of blocks, the
    prolonged ones among them, made one AIJ matrix and factorised) ends where the
    iterative solve does."""
    iterative = _baseline(tmp_path_factory, "dolfin", "coupled")
    direct = _run(tmp_path, monkeypatch, "dolfin", chns_solve="coupled",
                  chns_solver='{"method": "direct"}')
    np.testing.assert_allclose(direct, iterative, **CLOSE)


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


def _dense(M):
    return M.convert("dense").getDenseArray().copy()


def _blocks(mesh, V, Q, W, phig, physical):
    """The coupled step's coupling blocks (monolithic.CouplingBlocks) for phig's phi,
    assembled with every row of C unscaled (1/rho one) and no phase-field Dirichlet rows,
    as dense arrays: C_0, C_1, T_0, T_1 and B (with `physical`, Q's boundary dofs) or
    None."""
    from petsc4py import PETSc
    from twoasis.backend import FacetNormal, Measure
    from twoasis.solvers.TPfracStep.monolithic import CouplingBlocks
    blocks = CouplingBlocks(V, W, Q, phig, FacetNormal(mesh), Measure("ds", domain=mesh), 2,
                            physical, np.zeros(0, PETSc.IntType))
    ones = []
    for C in blocks.C:
        w = C.createVecLeft()
        w.set(1.0)
        ones.append(w)
    blocks.assemble(ones)
    return ([_dense(M) for M in blocks.C + blocks.T]
            + [None if blocks.B is None else _dense(blocks.B)])


def _spaces(mesh, box=None):
    from twoasis.backend import FiniteElement, FunctionSpace, MixedElement
    el = FiniteElement("CG", mesh.ufl_cell(), 1)
    return FunctionSpace(mesh, MixedElement([el, el]), constrained_domain=box)


def _dim(V):
    from twoasis.backend import global_dim
    return global_dim(V)


def _array(vector):
    """A vector's values (serial), on either backend."""
    from twoasis.backend import as_backend_type
    return as_backend_type(vector).vec().getArray().copy()


def _set(f, values):
    from twoasis.backend import as_backend_type
    as_backend_type(f.vector()).vec().setArray(values)
    as_backend_type(f.vector()).update_ghost_values()


def _mixed_prolongation(d, Wc, Wf):
    """The (phi, g) prolongation from Wc to Wf as a dense array (serial): the scalar one
    for each component, through the cells' dof tables (feluga.tables)."""
    from feluga.tables import space_tables
    from twoasis.backend import FunctionSpace
    Pc = FunctionSpace(d.coarse, "CG", 1)
    Pf = FunctionSpace(d.fine, "CG", 1)
    P1 = _dense(d.prolongation(Pc, Pf))
    tc, tf = space_tables(Wc), space_tables(Wf)
    sc, sf = space_tables(Pc).component()[0], space_tables(Pf).component()[0]
    P = np.zeros((_dim(Wf), _dim(Wc)))
    for i in (0, 1):
        mc, mf = np.zeros(_dim(Pc), int), np.zeros(_dim(Pf), int)
        mc[sc.ravel()] = tc.component(i)[0].ravel()
        mf[sf.ravel()] = tf.component(i)[0].ravel()
        P[np.ix_(mf, mc)] = P1
    return P


def test_coupling_blocks_exact():
    """The coupled step's blocks on the dual mesh (C_i = P^T C_f,i, T_i = T_f,i P, and
    B = diag(phi*) R_g on the boundary rows) are those of one mesh for a phase field the
    flow mesh holds: C_dual P_W = C, P_W^T T_dual = T, B_dual P_W = B, with P_W the
    prolongation of (phi, g) and phi* prolonged."""
    from twoasis.backend import assemble, Function, FunctionSpace, Measure, TestFunction
    from twoasis.common import dual
    from twoasis.backend import UnitSquareMesh
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    q = TestFunction(Q)
    w = _array(assemble(q * Measure("ds", domain=mesh)))
    physical = np.flatnonzero(np.abs(w) > 1e-12)
    rng = np.random.default_rng(1)
    Wc = _spaces(mesh)
    phig_c = Function(Wc)
    _set(phig_c, rng.uniform(-1, 1, _dim(Wc)))
    d = dual.start(mesh, 1)
    try:
        Wf = _spaces(d.fine)
        P = _mixed_prolongation(d, Wc, Wf)
        phig_f = Function(Wf)
        _set(phig_f, P @ _array(phig_c.vector()))
        fine = _blocks(mesh, V, Q, Wf, phig_f, physical)
    finally:
        dual.start(mesh, 0)
    one = _blocks(mesh, V, Q, Wc, phig_c, physical)
    for k in (0, 1):
        np.testing.assert_allclose(fine[k] @ P, one[k], atol=1e-12)
        np.testing.assert_allclose(P.T @ fine[2 + k], one[2 + k], atol=1e-12)
    np.testing.assert_allclose(fine[4] @ P, one[4], atol=1e-12)
    assert np.abs(one[4]).max() > 0.1 and np.abs(one[0]).max() > 0.01


def test_coupling_blocks_exchange_energy():
    """On the dual mesh, for any phase field on the fine mesh, the capillary force and the
    advection exchange energy exactly on a periodic box, where the boundary term cancels,
    as on one mesh: C_i^T = -S T_i, S swapping phi's and g's dofs at each node (C's
    columns are g's; T's rows are the phase-field equation's, tested by g)."""
    from twoasis.backend import Function, FunctionSpace, PeriodicBox, UnitSquareMesh
    from twoasis.common import dual
    mesh = UnitSquareMesh(4, 4)
    box = PeriodicBox({0: (0, 1), 1: (0, 1)})
    V = FunctionSpace(mesh, "CG", 2, constrained_domain=box)
    Q = FunctionSpace(mesh, "CG", 1, constrained_domain=box)
    d = dual.start(mesh, 1, box)
    try:
        Wf = _spaces(d.fine, box)
        phig = Function(Wf)
        _set(phig, np.random.default_rng(2).uniform(-1, 1, _dim(Wf)))
        C0, C1, T0, T1, B = _blocks(mesh, V, Q, Wf, phig, None)
        S = _swap(Wf)
    finally:
        dual.start(mesh, 0)
    assert B is None and np.abs(C0).max() > 0.01
    np.testing.assert_allclose(C0.T, -S @ T0, atol=1e-12)
    np.testing.assert_allclose(C1.T, -S @ T1, atol=1e-12)


def _swap(W):
    """The permutation of W's dofs that swaps phi's and g's at each node (one element)."""
    from feluga.tables import space_tables
    t = space_tables(W)
    phi, g = t.component(0)[0].ravel(), t.component(1)[0].ravel()
    S = np.zeros((_dim(W), _dim(W)))
    S[g, phi] = S[phi, g] = 1.0
    return S


def test_coupling_blocks_reassembled():
    """On the dual mesh the blocks follow phi* from step to step: blocks built for one phi*
    and assembled again for another, with C's rows scaled by a 1/rho and some of the phase
    field's rows held (Dirichlet), equal blocks built for the second phi* directly (the
    reused products, C's scaling and T's zeroed rows); T's held rows are zero."""
    from petsc4py import PETSc
    from twoasis.backend import (assemble, FacetNormal, Function, FunctionSpace, Measure,
                                 TestFunction, UnitSquareMesh)
    from twoasis.common import dual
    from twoasis.solvers.TPfracStep.monolithic import CouplingBlocks
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    ds = Measure("ds", domain=mesh)
    physical = np.flatnonzero(np.abs(_array(assemble(TestFunction(Q) * ds))) > 1e-12)
    rng = np.random.default_rng(3)
    d = dual.start(mesh, 1)
    try:
        Wf = _spaces(d.fine)
        held = np.arange(0, _dim(Wf), 7, dtype=PETSc.IntType)
        first, second = Function(Wf), Function(Wf)
        _set(first, rng.uniform(-1, 1, _dim(Wf)))
        _set(second, rng.uniform(-1, 1, _dim(Wf)))
        phig = Function(Wf)
        _set(phig, _array(first.vector()))
        reused = CouplingBlocks(V, Wf, Q, phig, FacetNormal(mesh), ds, 2, physical, held)
        w = []
        for C in reused.C:
            wi = C.createVecLeft()
            wi.setArray(rng.uniform(0.5, 2.0, wi.getLocalSize()))
            w.append(wi)
        reused.assemble(w)
        _set(phig, _array(second.vector()))
        reused.assemble(w)
        fresh = CouplingBlocks(V, Wf, Q, second, FacetNormal(mesh), ds, 2, physical, held)
        fresh.assemble(w)
        blocks = [(reused.C + reused.T + [reused.B]), (fresh.C + fresh.T + [fresh.B])]
        a, b = ([_dense(M) for M in bl] for bl in blocks)
    finally:
        dual.start(mesh, 0)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, atol=1e-12)
    for T in a[2:4]:
        assert np.abs(T[held]).max() == 0 and np.abs(T).max() > 0.01
    assert np.abs(a[4]).max() > 0.1
