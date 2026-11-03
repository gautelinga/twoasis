"""Unit tests for small pieces of logic, without running a simulation."""
import sys

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
