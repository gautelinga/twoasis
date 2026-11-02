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
    assert d["constrained_domain"].periodic == {0: (0, 2)}
    V = dolfin.FunctionSpace(d["mesh"], "CG", 1, constrained_domain=d["constrained_domain"])
    assert V.dim() == 8 * 9  # 9 x 9 vertices, the right column mapped onto the left


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
