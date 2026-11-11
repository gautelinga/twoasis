"""Unit tests for small pieces of logic, without running a simulation."""
import sys

import numpy as np
import pytest

dolfin = pytest.importorskip("dolfin")


# --- Krylov tolerances ------------------------------------------------------

@pytest.fixture
def krylov():
    from twoasis.problems.TPfracStep import KRYLOV_TOLERANCES, NS_parameters
    from twoasis.solvers.TPfracStep import krylov_solver_parameters
    defaults = {name: dict(NS_parameters[name]) for name in
                ("krylov_solvers", "pressure_krylov_solver", "velocity_update_solver")}
    return krylov_solver_parameters, defaults, KRYLOV_TOLERANCES


def test_krylov_defaults(krylov):
    params, d, table = krylov
    p = params("pressure_krylov_solver", d["pressure_krylov_solver"], d["krylov_solvers"])
    assert p["relative_tolerance"] == table["pressure_krylov_solver"]["relative_tolerance"]
    assert p["maximum_iterations"] == table["pressure_krylov_solver"]["maximum_iterations"]
    assert None not in p.values()


def test_krylov_global_overrides_default(krylov):
    params, d, _ = krylov
    glob = dict(d["krylov_solvers"], relative_tolerance=1e-10)
    for name in ("pressure_krylov_solver", "velocity_update_solver"):
        assert params(name, d[name], glob)["relative_tolerance"] == 1e-10


def test_krylov_per_solver_overrides_global(krylov):
    params, d, table = krylov
    glob = dict(d["krylov_solvers"], relative_tolerance=1e-10)
    # Even a per-solver value equal to the default must win over the global one.
    default = table["pressure_krylov_solver"]["relative_tolerance"]
    cfg = dict(d["pressure_krylov_solver"], relative_tolerance=default)
    assert params("pressure_krylov_solver", cfg, glob)["relative_tolerance"] == default


def test_krylov_other_settings_pass_through(krylov):
    params, d, _ = krylov
    glob = dict(d["krylov_solvers"], monitor_convergence=True)
    assert params("pressure_krylov_solver", d["pressure_krylov_solver"], glob)["monitor_convergence"]


def ksp_of(solver_type, preconditioner_type, **cfg):
    from twoasis.solvers.TPfracStep import krylov_solver
    cfg.update(solver_type=solver_type, preconditioner_type=preconditioner_type)
    return krylov_solver("velocity_krylov_solver", cfg, {}).ksp()


@pytest.mark.parametrize("solver_type, pc, expected", [
    ("bicgstab", "jacobi", ("bcgs", "jacobi", None)),
    ("cg", "bjacobi", ("cg", "bjacobi", None)),
    ("gmres", "sor", ("gmres", "sor", None)),
    ("gmres", "none", ("gmres", "none", None)),
    ("default", "default", ("gmres", "bjacobi", None)),
    ("gmres", "hypre_amg", ("gmres", "hypre", "boomeramg")),
    ("gmres", "hypre_euclid", ("gmres", "hypre", "euclid")),
    ("gmres", "petsc_amg", ("gmres", "gamg", None)),
])
def test_krylov_solver_sets_what_is_asked(solver_type, pc, expected):
    # dolfin's PETScPreconditioner objects drop jacobi, sor and none, and its
    # string names drop the hypre type.
    ksp = ksp_of(solver_type, pc)
    hypre = ksp.getPC().getHYPREType() if expected[1] == "hypre" else None
    assert (ksp.getType(), ksp.getPC().getType(), hypre) == expected


def test_krylov_solver_defaults_build():
    from twoasis.problems.TPfracStep import NS_parameters
    for name, cfg in NS_parameters.items():
        # 'schur' is built by phase_field_schur_solver
        if name.endswith("_solver") and cfg.get("preconditioner_type", "schur") != "schur":
            ksp_of(**{k: cfg[k] for k in ("solver_type", "preconditioner_type")})


def schur_solve(n, fixed=False, petsc_options=None, alpha=1., mobility=1e-2):
    """Solve a (phi, g) system on an n x n mesh with the PhaseFieldSchur
    preconditioner (fixed: phi prescribed on the boundary); returns the solver."""
    from twoasis.solvers.TPfracStep import phase_field_krylov_solve, phase_field_schur_solver
    from femtools.fast_bc import apply
    mesh = dolfin.UnitSquareMesh(n, n)
    P1 = dolfin.FiniteElement("CG", mesh.ufl_cell(), 1)
    W = dolfin.FunctionSpace(mesh, dolfin.MixedElement([P1, P1]))
    phi, g = dolfin.TrialFunctions(W)
    xi, eta = dolfin.TestFunctions(W)
    sigma_bar, epsilon = 1., 0.04
    grad, inner, dx = dolfin.grad, dolfin.inner, dolfin.dx
    a = (alpha * phi * xi + mobility * inner(grad(g), grad(xi)) + g * eta
         - sigma_bar * epsilon * inner(grad(phi), grad(eta))) * dx
    f = dolfin.Expression("sin(3*x[0])*cos(2*x[1])", degree=2)
    A, b = dolfin.assemble(a), dolfin.assemble(f * xi * dx)
    bcs = [dolfin.DirichletBC(W.sub(0), 1., "on_boundary")] if fixed else []
    for bc in bcs:
        apply(bc, A, b)
    x = dolfin.Function(W).vector()
    cfg = dict(solver_type="fgmres", preconditioner_type="schur",
               petsc_options=petsc_options or {})
    sol = phase_field_schur_solver(W, cfg, {}, bcs)
    phase_field_krylov_solve(sol, A, x, b, alpha, mobility, sigma_bar, epsilon)
    r = b.copy()
    A.mult(x, r)
    r -= b
    assert r.norm("l2") < 1e-7 * b.norm("l2")
    return sol


def test_phase_field_schur_solver_converges():
    """Iterations do not grow with the mesh, where block Jacobi's grow sixfold
    from 16 x 16 to 64 x 64; with phi prescribed on the boundary they grow, but
    slowly."""
    its = {n: schur_solve(n).ksp().getIterationNumber() for n in (16, 64)}
    assert its[64] <= min(its[16] + 4, 20)
    assert schur_solve(32, fixed=True, alpha=100., mobility=1e-3).ksp().getIterationNumber() <= 16


def test_phase_field_schur_inner_solvers():
    """H is solved with BoomerAMG in 2D and CG-Jacobi in 3D, and petsc_options
    with the prefix H_ choose another."""
    from twoasis.solvers.TPfracStep import phase_field_schur_solver

    def inner(sol):
        kH = sol.ksp().getPC().getPythonContext().kH
        return kH.getType(), kH.getPC().getType()

    assert inner(schur_solve(8)) == ("preonly", "hypre")
    assert inner(schur_solve(8, petsc_options={"H_ksp_type": "cg", "H_pc_type": "jacobi"})) \
        == ("cg", "jacobi")
    mesh = dolfin.UnitCubeMesh(2, 2, 2)
    P1 = dolfin.FiniteElement("CG", mesh.ufl_cell(), 1)
    W = dolfin.FunctionSpace(mesh, dolfin.MixedElement([P1, P1]))
    assert inner(phase_field_schur_solver(
        W, dict(solver_type="fgmres", preconditioner_type="schur"), {}, [])) == ("cg", "jacobi")


def test_phase_field_schur_solver_needs_a_flexible_method():
    from twoasis.solvers.TPfracStep import phase_field_schur_solver
    mesh = dolfin.UnitCubeMesh(2, 2, 2)
    P1 = dolfin.FiniteElement("CG", mesh.ufl_cell(), 1)
    W = dolfin.FunctionSpace(mesh, dolfin.MixedElement([P1, P1]))
    with pytest.raises(ValueError, match="flexible"):
        phase_field_schur_solver(W, dict(solver_type="gmres", preconditioner_type="schur"),
                                 {}, [])


def test_krylov_solver_petsc_options_reach_the_blocks():
    mesh = dolfin.UnitSquareMesh(4, 4)
    V = dolfin.FunctionSpace(mesh, "CG", 1)
    u, v = dolfin.TrialFunction(V), dolfin.TestFunction(V)
    A = dolfin.as_backend_type(dolfin.assemble(u * v * dolfin.dx))
    # sub_ options are read when the blocks are set up (petsc4py can only
    # reach the blocks of asm, which reads them like bjacobi).
    ksp = ksp_of("cg", "asm", petsc_options={"sub_pc_type": "icc"})
    ksp.setOperators(A.mat())
    ksp.setUp()
    assert ksp.getPC().getASMSubKSP()[0].getPC().getType() == "icc"
    # The next solver with this prefix does not inherit the option.
    ksp = ksp_of("cg", "asm")
    ksp.setOperators(A.mat())
    ksp.setUp()
    assert ksp.getPC().getASMSubKSP()[0].getPC().getType() == "ilu"


def test_krylov_solver_rejects_unknown_preconditioner():
    from petsc4py import PETSc
    with pytest.raises(PETSc.Error):
        ksp_of("gmres", "jacobii")


def solvers(mesh, **cfg):
    """krylov_get_solvers with the default settings, updated by cfg."""
    from twoasis.problems.TPfracStep import NS_parameters
    from twoasis.solvers.TPfracStep import krylov_get_solvers
    W = dolfin.FunctionSpace(mesh, dolfin.MixedElement([dolfin.FiniteElement(
        "CG", mesh.ufl_cell(), 1)] * 2))
    params = {k: NS_parameters[k] for k in (
        "krylov_solvers", "velocity_krylov_solver", "velocity_update_solver",
        "pressure_krylov_solver", "phase_field_krylov_solver", "velocity_tolerance_factor")}
    params.update(cfg)
    return krylov_get_solvers(q_=dict(phig=dolfin.Function(W)), mesh=mesh, bcs=dict(phig=[]),
                              **params)


@pytest.mark.parametrize("factor", [0, 0.1])
def test_velocity_tolerance_factor(factor):
    """With the factor a velocity solve stops once its residual has fallen by it
    from the one it starts from; without, at the relative tolerance 1e-6."""
    mesh = dolfin.UnitSquareMesh(32, 32)
    V = dolfin.FunctionSpace(mesh, "CG", 2)
    u, v = dolfin.TrialFunction(V), dolfin.TestFunction(V)
    A = dolfin.assemble((100 * u * v + dolfin.inner(dolfin.grad(u), dolfin.grad(v))) * dolfin.dx)
    b = dolfin.assemble(dolfin.Expression("sin(3*x[0])", degree=2) * v * dolfin.dx)
    x = dolfin.Function(V).vector()
    dolfin.solve(A, x, b, "lu")
    x.set_local(x.get_local() * (1 + 1e-3 * np.random.default_rng(0).standard_normal(
        x.local_size())))
    x.apply("insert")

    def residual():
        r = b.copy()
        A.mult(x, r)
        r -= b
        return r.norm("l2")

    r0 = residual()
    solvers(mesh, velocity_tolerance_factor=factor)[0].solve(A, x, b)
    if factor:
        assert 1e-3 < residual() / r0 <= factor
    else:
        assert residual() / b.norm("l2") < 1e-5


def test_boomeramg_3d_settings():
    """The 3D pressure's BoomerAMG takes femtools.ksp.BOOMERAMG_3D's settings, which the
    pressure's petsc_options override; 2D and other preconditioners take none."""
    from petsc4py import PETSc
    key = "pressure_krylov_solver_pc_hypre_boomeramg_coarsen_type"
    square, cube = dolfin.UnitSquareMesh(2, 2), dolfin.UnitCubeMesh(2, 2, 2)
    amg = dict(solver_type="gmres", preconditioner_type="hypre_amg")
    for mesh, cfg, expected in [
            (square, amg, None), (cube, amg, "HMIS"),
            (cube, dict(amg, preconditioner_type="petsc_amg"), None),
            (cube, dict(amg, petsc_options={"pc_hypre_boomeramg_coarsen_type": "PMIS"}), "PMIS")]:
        solvers(mesh, pressure_krylov_solver=cfg)
        assert PETSc.Options().getAll().get(key) == expected


def test_anderson_reaches_the_fixed_point():
    """Passes of an affine map p -> A p + c with A's eigenvalues in [0.5, 0.99]:
    with as many earlier passes as unknowns, Anderson acceleration reaches the
    fixed point within a few more passes than that; plain passes are far off."""
    from twoasis.solvers.TPfracStep import AndersonPressure
    p = dolfin.Function(dolfin.FunctionSpace(dolfin.UnitIntervalMesh(7), "CG", 1)).vector()
    n = p.local_size()
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = Q @ np.diag(np.linspace(0.5, 0.99, n)) @ Q.T
    c = rng.standard_normal(n)
    fixed = np.linalg.solve(np.eye(n) - A, c)

    def error(depth, passes):
        accelerate = AndersonPressure(depth) if depth else None
        p.zero()
        for k in range(1, passes + 1):
            p.set_local(A @ p.get_local() + c)
            p.apply("insert")
            if accelerate:
                accelerate(p, k)
        return np.abs(p.get_local() - fixed).max()

    assert error(n, n + 3) < 1e-10
    assert error(0, n + 3) > 1.


def test_rotational_pressure_update():
    """p += -mu*div(u*) at the pressure dofs, and prescribed pressures stay."""
    from types import SimpleNamespace
    from twoasis.solvers.TPfracStep import rotational_fields, rotational_pressure_update
    mesh = dolfin.UnitSquareMesh(8, 8)
    Q = dolfin.FunctionSpace(mesh, "CG", 1)
    q = dolfin.TestFunction(Q)
    u = dolfin.interpolate(dolfin.Expression(("x[0]*x[0]", "0"), degree=2),
                           dolfin.VectorFunctionSpace(mesh, "CG", 1))
    divu = SimpleNamespace(rhs=dolfin.assemble(dolfin.div(u) * q * dolfin.dx))
    mu_ = dolfin.interpolate(dolfin.Constant(2.), Q)
    p = dolfin.Function(Q).vector()
    bc = dolfin.DirichletBC(Q, 1.5, dolfin.DomainBoundary())
    bc.apply(p)
    rotational_pressure_update(True, {"p": p}, divu, mu_, bcs={"p": [bc]}, inner_iter=1,
                               **rotational_fields(True, q, Q, Q, mu_))
    fixed = list(bc.get_boundary_values())
    inner = np.setdiff1d(np.arange(p.local_size()), fixed)
    x = Q.tabulate_dof_coordinates()
    assert np.all(p.get_local()[fixed] == 1.5)
    assert np.allclose(p.get_local()[inner], -2 * 2 * x[inner, 0], atol=1e-12)


# --- BDF first step -----------------------------------------------------------

def levels(x1, x2):
    """x_1, x_2 dicts with one field 'f' holding the given values."""
    def vec(values):
        v = dolfin.Vector(dolfin.MPI.comm_self, len(values))
        v.set_local(list(values))
        v.apply("insert")
        return v
    return {"f": vec(x1)}, {"f": vec(x2)}


@pytest.mark.parametrize("x1, x2, expected", [
    ([1., 2.], [0.9, 1.9], True),   # distinct t - 2*dt level
    ([1., 2.], [1., 2.], False),    # copy of the t - dt level
    ([1., 2.], [0., 0.], False),    # left at zero
    ([0., 0.], [0., 0.], True),     # zero at both levels is consistent (e.g. at rest)
])
def test_has_previous_level(x1, x2, expected):
    from twoasis.solvers.TPfracStep.BDF import has_previous_level
    x_1, x_2 = levels(x1, x2)
    assert has_previous_level(x_1, x_2, ["f"]) is expected


# --- Command line ---------------------------------------------------------------

def test_parse_command_line(monkeypatch):
    from twoasis.common import parse_command_line
    monkeypatch.setattr(sys, "argv", ["twoasis", "dt=0.01", "solver=BDF", "flag=True",
                                      "rho=[1, 0.1]", 'pressure_krylov_solver={"relative_tolerance": 1e-9}'])
    kw = parse_command_line()
    assert kw == dict(dt=0.01, solver="BDF", flag=True, rho=[1, 0.1],
                      pressure_krylov_solver={"relative_tolerance": 1e-9})


def test_parse_command_line_rejects_bad_argument(monkeypatch):
    from twoasis.common import parse_command_line
    monkeypatch.setattr(sys, "argv", ["twoasis", "dt"])
    with pytest.raises(TypeError):
        parse_command_line()


def test_command_line_dicts_merge_into_defaults():
    from twoasis.problems import post_import_problem
    NS_parameters = dict(dt=0.1, velocity_update_solver=dict(method="lumping", solver_type="cg"))
    d = post_import_problem(NS_parameters, dolfin.UnitSquareMesh(2, 2),
                            dict(dt=0.01, velocity_update_solver=dict(method="default")), {})
    assert d["dt"] == 0.01
    assert d["velocity_update_solver"] == dict(method="default", solver_type="cg")


def test_restart_parameters(tmp_path):
    """A restart takes the checkpoint's parameters, but the code's Krylov
    settings; of the velocity update it keeps the method, and a checkpoint that
    does not name the rotational update ran without it."""
    import pickle
    from twoasis.problems import post_import_problem
    saved = dict(dt=0.5, pressure_krylov_solver=dict(preconditioner_type="jacobi"),
                 velocity_update_solver=dict(method="lumping", preconditioner_type="ilu"))
    (tmp_path / "params.dat").write_bytes(pickle.dumps(saved))
    defaults = dict(dt=0.1, rotational=True, restart_folder=None,
                    pressure_krylov_solver=dict(preconditioner_type="hypre_amg"),
                    velocity_update_solver=dict(method="default", preconditioner_type="jacobi"))
    d = post_import_problem(defaults, dolfin.UnitSquareMesh(2, 2),
                            dict(restart_folder=str(tmp_path)), {})
    assert (d["dt"], d["rotational"]) == (0.5, False)
    assert d["pressure_krylov_solver"] == dict(preconditioner_type="hypre_amg")
    assert d["velocity_update_solver"] == dict(method="lumping", preconditioner_type="jacobi")


def test_periodic_map_follows_command_line():
    """A periodic map given as a function is built after the command line has
    been merged in, so a length given there sets both mesh and periodicity."""
    from twoasis.problems import post_import_problem
    from twoasis.problems.TPfracStep import Bubble2D
    d = post_import_problem(dict(Lx=1, Ly=2, res=0.25), Bubble2D.mesh, dict(Lx=2), {},
                            constrained_domain=Bubble2D.constrained_domain)
    box = d["constrained_domain"]
    assert (box.periodic, box.lower[0], box.upper[0]) == ([True, False, False], 0, 2)
    V = dolfin.FunctionSpace(d["mesh"], "CG", 1, constrained_domain=d["constrained_domain"])
    assert V.dim() == 8 * 9  # 9 x 9 vertices, the right column mapped onto the left


PERIODIC_3D = r"""
import sys
from dolfin import FiniteElement, FunctionSpace, MixedElement, MPI, set_log_level
from femtools.periodic_partition import PairingError
from twoasis.common.utilities import check_periodic_pairing
from twoasis.problems.TPfracStep.shared import centred, load_mesh, PeriodicBox
set_log_level(40)
mesh = load_mesh(sys.argv[1], periodic=(True, True, False))
box = PeriodicBox({0: centred(1.), 1: centred(1.)})
el = [FiniteElement("CG", mesh.ufl_cell(), k) for k in (1, 2)]
W = FunctionSpace(mesh, MixedElement(el), constrained_domain=box)
check_periodic_pairing(box, [FunctionSpace(mesh, "CG", 2, constrained_domain=box),
                             FunctionSpace(mesh, "CG", 1, constrained_domain=box),
                             W.sub(0), W.sub(1)])
try:
    check_periodic_pairing(box, [FunctionSpace(mesh, "CG", 2)])  # not periodic
    raise AssertionError("a space without the constraint passed the check")
except PairingError:
    pass
if MPI.rank(MPI.comm_world) == 0:
    print("checked on", MPI.size(MPI.comm_world))
"""

# A square centred at the origin whose partition (stored in the file) leaves rank 1
# with one triangle touching the right edge at a single vertex: the periodic dofmap
# breaks there unless post_import_problem repairs the partition.
PERIODIC_2D = r"""
import sys
import h5py
import numpy as np
import dolfin as df
from femtools.periodic_partition import PairingError
from twoasis.common.utilities import check_periodic_pairing
from twoasis.problems import post_import_problem
from twoasis.problems.TPfracStep.shared import centred, PeriodicBox
df.set_log_level(40)
comm = df.MPI.comm_world
path = sys.argv[1]
if comm.rank == 0:
    m = df.RectangleMesh(df.MPI.comm_self, df.Point(-.5, -.5), df.Point(.5, .5), 12, 12, "crossed")
    x, topo = m.coordinates(), m.cells().astype(np.int64)
    owner = (x[topo].mean(1)[:, 1] >= 0).astype(np.int64)
    edge = np.abs(x[:, 0] - .5) < 1e-8
    deep = edge & (x[:, 1] > -.4) & (x[:, 1] < -.1)
    owner[np.nonzero((edge[topo].sum(1) == 1) & deep[topo].any(1) & (owner == 0))[0][0]] = 1
    perm = np.argsort(owner, kind="stable")
    with h5py.File(path, "w") as h:
        h["mesh/coordinates"] = x
        t = h.create_dataset("mesh/topology", data=topo[perm])
        t.attrs["celltype"] = np.bytes_("triangle")
        t.attrs["partition"] = np.searchsorted(owner[perm], np.arange(2)).astype(np.uint64)
        h["mesh/cell_indices"] = perm.astype(np.int64)
comm.Barrier()


def crafted(**params):
    mesh = df.Mesh()
    with df.HDF5File(comm, path, "r") as h:
        h.read(mesh, "mesh", True)  # the partition stored in the file
    return mesh


box = lambda mesh, **params: PeriodicBox({0: centred(1.), 1: centred(1.)})
for repair in (False, True):
    if repair:
        d = post_import_problem({}, crafted, {}, {}, constrained_domain=box)
        mesh, cd = d["mesh"], d["constrained_domain"]
    else:
        mesh = crafted()
        cd = box(mesh)
    try:
        check_periodic_pairing(cd, [df.FunctionSpace(mesh, "CG", 2, constrained_domain=cd)])
        ok = True
    except PairingError:
        ok = False
    if comm.rank == 0:
        print("repaired" if repair else "crafted", ok)
"""


def run_mpi_script(tmp_path, script, *args, nprocs=2):
    import os
    import subprocess
    from conftest import REPO_ROOT
    path = tmp_path / "script.py"
    path.write_text(script)
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run(["mpirun", "-np", str(nprocs), sys.executable, str(path)] + list(args),
                          cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
    return proc.stdout


def test_periodic_3d_in_parallel(tmp_path):
    """A periodic 3D mesh read on two processes (partitioned by femtools) gives
    spaces that pass the pairing check, on a box centred at the origin; a space
    without the constraint fails it."""
    from conftest import require_mpi
    require_mpi(2)
    mesh = dolfin.BoxMesh(dolfin.MPI.comm_self, dolfin.Point(-.5, -.5, 0.), dolfin.Point(.5, .5, .6),
                          8, 8, 4)
    with dolfin.HDF5File(mesh.mpi_comm(), str(tmp_path / "box.h5"), "w") as f:
        f.write(mesh, "mesh")
    assert "checked on 2" in run_mpi_script(tmp_path, PERIODIC_3D, str(tmp_path / "box.h5"))


def test_periodic_2d_partition_is_repaired(tmp_path):
    """A 2D partition on which dolfin's periodic dofmap breaks: the check catches
    it, and post_import_problem repairs the partition so that it passes."""
    from conftest import require_mpi
    require_mpi(2)
    out = run_mpi_script(tmp_path, PERIODIC_2D, str(tmp_path / "square.h5"))
    assert "crafted False" in out and "repaired True" in out, out[-2000:]


def test_interface_energy_of_flat_interface():
    """A flat interface at equilibrium has the energy sigma per unit length,
    half in the gradient part and half in the double-well part."""
    from twoasis.problems.TPfracStep.shared import interface_energy
    eps, sigma, width = 0.05, 2.0, 0.1
    mesh = dolfin.RectangleMesh(dolfin.Point(-1, 0), dolfin.Point(1, width), 400, 2)
    phi = dolfin.interpolate(dolfin.Expression("tanh(x[0]/(sqrt(2)*eps))", eps=eps, degree=4),
                             dolfin.FunctionSpace(mesh, "CG", 2))
    E_grad, E_dw = interface_energy(phi, sigma, eps, dolfin.dx(domain=mesh))
    assert E_grad + E_dw == pytest.approx(sigma * width, rel=1e-3)
    assert E_grad == pytest.approx(E_dw, rel=1e-3)


def test_bubble2d_cases():
    """Bubble2D case=2 sets Hysing et al.'s test case 2 and what it needs; the
    default is test case 1; any other case is refused."""
    from twoasis.problems.TPfracStep import Bubble2D
    one, two = {}, {}
    Bubble2D.problem_parameters(NS_parameters=one, commandline_kwargs={})
    Bubble2D.problem_parameters(NS_parameters=two, commandline_kwargs={"case": 2})
    assert (one["case"], one["rho"], one["mu"], one["sigma"]) == (1, [1000., 100.], [10., 1.], 24.5)
    assert one.get("phase_field_degree") is None
    assert (two["case"], two["rho"], two["mu"], two["sigma"]) == (2, [1000., 1.], [10., 0.1], 1.96)
    assert (two["phase_field_degree"], two["chemical_potential_degree"]) == (1, 1)
    assert (two["res"], two["dt"]) == (1. / 128, 0.001)
    with pytest.raises(ValueError, match="case=3"):
        Bubble2D.problem_parameters(NS_parameters={}, commandline_kwargs={"case": 3})


def test_physical_pressure_bc():
    """PhysicalPressureBC prescribes the physical pressure p + phi*g: after
    update_physical the solver's pressure gets value - phi*g on the boundary."""
    import numpy as np
    from dolfin import (DirichletBC, DomainBoundary, Expression, FiniteElement, Function,
                        FunctionSpace, MixedElement, UnitSquareMesh, interpolate)
    from twoasis.common.utilities import phi_g
    from twoasis.problems.TPfracStep.shared import PhysicalPressureBC
    mesh = UnitSquareMesh(8, 8)
    Q = FunctionSpace(mesh, "CG", 1)
    el = FiniteElement("CG", mesh.ufl_cell(), 2)
    W = FunctionSpace(mesh, MixedElement([el, el]))
    q_ = dict(p=Function(Q), phig=interpolate(Expression(("tanh(x[0] - 0.5)", "x[1] + 1"),
                                                         degree=2), W))
    bc = PhysicalPressureBC(Q, 2.5, DomainBoundary())
    bc.update_physical(phi_g(q_))
    p = Function(Q)
    bc.apply(p.vector())
    dofs = list(DirichletBC(Q, 0, DomainBoundary()).get_boundary_values().keys())
    x = Q.tabulate_dof_coordinates()[dofs]
    expected = 2.5 - np.tanh(x[:, 0] - 0.5) * (x[:, 1] + 1)
    assert np.abs(p.vector().get_local()[dofs] - expected).max() < 1e-12
