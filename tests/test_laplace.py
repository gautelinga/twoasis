"""Laplace pressure and spurious currents of a resting bubble (LaplaceBubble2D).

The case runs until dP and max|u| are steady and writes them, with the
equilibrium bubble radius R_eq, to Stats/steady.dat. Measured (sigma = 1,
R = 0.25, epsilon = 1.28*h, M = 0.1):

    epsilon   dP vs sigma/R_eq   shrinkage 1 - R_eq/R
    0.04      0.000%             4.5%
    0.02      0.026%             2.3%

dP follows the Laplace law for the actual radius. The bubble shrinks because
the bulk phase field shifts away from +-1 at equilibrium, by an amount first
order in epsilon. For the bubble at rest there are no steady spurious
currents: max|u| at steady state is set by the solver tolerances (1.6e-7 with
the defaults, 6.8e-10 with tolerances of 1e-10 on the coarse level). Moving interfaces
are not covered. See LaplaceBubble2D.py for the dependence on
density, viscosity, Ohnesorge and Cahn numbers.
"""
from pathlib import Path

import numpy as np
import pytest

from conftest import QUIET, golden_compare, load_tdata, run_twoasis

# Columns of LaplaceBubble2D's Stats/tdata.dat and Stats/steady.dat.
LCOLS = dict(tstep=0, t=1, dP=2, exact=3, P_in=4, P_out=5, u_max=6, E_kin=7, phim=8,
             R_eq=9, phi_in=10, phi_out=11)
SCOLS = dict(t=0, dP=1, exact=2, error=3, u_max=4, R_eq=5, laplace_error=6)

# Coarse level: epsilon = 0.04 on a 32 x 32 mesh. Steady after about 0.43.
LAPLACE = dict(QUIET, problem="LaplaceBubble2D", epsilon=0.04, res=0.03125, dt=0.002,
               T=1.0, stat_interval=5)
R = 0.25  # the problem's default radius

LAPLACE_TOL = 5e-3   # dP against sigma/R_eq
# max|u| at steady state with the default solver tolerances. Not a physical
# spurious current: the velocity stops changing once each step's update falls
# below the tolerances, and freezes at about this level.
SOLVER_LEVEL = 1e-6

# Tolerances tight enough to drive max|u| far below SOLVER_LEVEL.
TIGHT = dict(krylov_solvers='{"relative_tolerance": 1e-10, "absolute_tolerance": 1e-16, '
                            '"maximum_iterations": 5000}',
             max_error=1e-10, max_iter=100)


def run_laplace(workdir, **params):
    """Run LaplaceBubble2D; return (tdata, steady values or None)."""
    run_twoasis(workdir, **params)
    steady = Path(workdir) / "results" / "data" / "1" / "Stats" / "steady.dat"
    return load_tdata(workdir), (np.loadtxt(str(steady)) if steady.exists() else None)


def shrinkage(steady):
    return 1 - steady[SCOLS["R_eq"]] / R


@pytest.fixture(scope="session")
def laplace(tmp_path_factory):
    return run_laplace(tmp_path_factory.mktemp("laplace"), **LAPLACE)


def test_reaches_steady_state(laplace):
    td, steady = laplace
    assert steady is not None, "no steady state before T = {}".format(LAPLACE["T"])
    assert np.isfinite(td).all()
    assert td[-1, LCOLS["t"]] == pytest.approx(steady[SCOLS["t"]])  # the run stopped there


def test_laplace_law(laplace):
    """dP = sigma/R_eq for the measured equilibrium radius."""
    _, steady = laplace
    assert abs(steady[SCOLS["laplace_error"]]) < LAPLACE_TOL, \
        "dP = {:.5f}, sigma/R_eq = {:.5f}".format(steady[SCOLS["dP"]],
                                                 1 / steady[SCOLS["R_eq"]])


def test_bubble_shrinkage(laplace):
    """The bubble shrinks a little, so dP exceeds sigma/R for the initial R."""
    _, steady = laplace
    assert 0 < shrinkage(steady) < 0.06
    assert steady[SCOLS["error"]] > 0


def test_velocity_settles_to_solver_level(laplace):
    """At steady state the velocity is at the level the solver tolerances
    allow; a larger value means the capillary force and the pressure do not
    balance."""
    _, steady = laplace
    assert steady[SCOLS["u_max"]] < SOLVER_LEVEL


def test_mass_conservation(laplace):
    td, _ = laplace
    phim = td[:, LCOLS["phim"]]
    assert np.max(np.abs(phim - phim[0])) < 1e-6


def test_golden(request, laplace):
    _, steady = laplace
    golden_compare(request, "laplace", {"dP": float(steady[SCOLS["dP"]])})


@pytest.mark.slow
def test_refinement(tmp_path, laplace):
    """Halving epsilon and h halves the shrinkage (first order) and keeps the
    Laplace law, with the velocity settling at solver level. The steady state
    does not depend on the mobility, so the finer level uses a larger one to
    reach it in half the time (shrinkage 2.31% with M = 0.4 and with 0.1)."""
    _, coarse = laplace
    _, fine = run_laplace(tmp_path, **dict(LAPLACE, epsilon=0.02, res=0.015625, dt=0.0007,
                                            stat_interval=14, M=0.4))
    assert fine is not None, "no steady state on the finer level"
    ratio = shrinkage(coarse) / shrinkage(fine)
    assert 1.7 < ratio < 2.3, "shrinkage ratio {:.2f}, expected ~2 (first order)".format(ratio)
    assert abs(fine[SCOLS["laplace_error"]]) < LAPLACE_TOL
    assert fine[SCOLS["u_max"]] < SOLVER_LEVEL


@pytest.mark.slow
def test_no_steady_spurious_currents(tmp_path, laplace):
    """Tightening the solver tolerances drives the steady max|u| down by orders
    of magnitude (1.6e-7 -> 6.8e-10 measured) and leaves dP unchanged: for the
    bubble at rest, g is constant at equilibrium, so the force -phi*grad(g)
    vanishes and u = 0 is an exact steady state."""
    _, default = laplace
    _, tight = run_laplace(tmp_path, **dict(LAPLACE, **TIGHT))
    assert tight is not None, "no steady state with tight tolerances"
    assert tight[SCOLS["u_max"]] < 1e-2 * default[SCOLS["u_max"]], \
        "max|u| {:.2e} with tight tolerances, {:.2e} with the defaults".format(
            tight[SCOLS["u_max"]], default[SCOLS["u_max"]])
    assert tight[SCOLS["dP"]] == pytest.approx(default[SCOLS["dP"]], rel=1e-5)
