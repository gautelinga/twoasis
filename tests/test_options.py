"""Solver options that the default runs do not use. Each is compared with the
corresponding default run, so that an option that stops working is noticed."""
import numpy as np
import pytest

from conftest import (BUBBLE, BUBBLE_P1, MASS_TOL, assert_last_close, assert_mass_conserved,
                      load_tdata, passes, run_twoasis)


def test_phase_field_bjacobi(tmp_path, bubble_bdf):
    """Block Jacobi instead of the Schur preconditioner solves the same
    phase-field system; results differ only at the level of the Krylov
    tolerances."""
    run_twoasis(tmp_path, solver="BDF", T=0.02,
                phase_field_krylov_solver='{"solver_type": "gmres", "preconditioner_type": "bjacobi"}',
                **BUBBLE)
    td = load_tdata(tmp_path)
    assert_mass_conserved(td, MASS_TOL)
    assert_last_close(td, bubble_bdf[1], [("u0m", 1e-3), ("phim", 1e-6), ("E_int", 1e-5)], "bjacobi")


def test_lumped_mass_velocity_update(tmp_path, bubble_p1_bdf):
    """method='lumping' (lumped mass for P1) instead of the default consistent
    mass. The velocity update only applies the last pressure increment of the
    inner iterations, which is small, so the results stay close."""
    run_twoasis(tmp_path, solver="BDF", T=0.02,
                velocity_update_solver='{"method": "lumping"}', **BUBBLE_P1)
    assert_last_close(load_tdata(tmp_path), bubble_p1_bdf[1],
                      [("u0m", 1e-5), ("phim", 1e-8), ("E_int", 1e-7)], "lumped mass")


def test_ab_projection_pressure(tmp_path, bubble_bdf):
    """Extrapolating the pressure as the first guess changes only where the
    inner iterations start, not what they converge to."""
    run_twoasis(tmp_path, solver="BDF", T=0.02, AB_projection_pressure=True, **BUBBLE)
    assert_last_close(load_tdata(tmp_path), bubble_bdf[1],
                      [("u0m", 1e-4), ("phim", 1e-7), ("E_int", 1e-6)], "AB projection")


@pytest.mark.parametrize("solver", ["BDF", "IPCS"])
def test_lower_degree_phase_field(tmp_path, bubble_p1_bdf, solver):
    """A P1 phase field and chemical potential with P2 velocity: mass is
    conserved, and the interface energy is close to that of the all-P1 run
    (2e-4 relative measured)."""
    run_twoasis(tmp_path, solver=solver, T=0.02, **dict(BUBBLE_P1, velocity_degree=2,
                                                        phase_field_degree=1))
    td = load_tdata(tmp_path)
    assert np.isfinite(td).all()
    assert_mass_conserved(td, MASS_TOL)
    assert_last_close(td, bubble_p1_bdf[1], [("E_int", 2e-3)], "P1 phase field")


SET_FORCING = '''
from dolfin import Constant
from twoasis.problems.TPfracStep.PeriodicBubble2D import *


def start_timestep_hook(tstep, gradp_avg, **NS_namespace):
    """Set the mean pressure gradient (-F0) on the first step."""
    if tstep == 1:
        gradp_avg.assign(Constant((-1., 0.)))
'''


@pytest.mark.parametrize("solver", ["BDF", "IPCS"])
def test_forcing_changed_by_a_hook(tmp_path, solver):
    """A Constant mean pressure gradient that a hook changes is used from then
    on: the same as F0 given from the start."""
    (tmp_path / "hook").mkdir()
    (tmp_path / "hook" / "SetForcing.py").write_text(SET_FORCING)
    (tmp_path / "given").mkdir()
    run_twoasis(tmp_path / "hook", solver=solver, T=0.006, **dict(BUBBLE, problem="SetForcing"))
    run_twoasis(tmp_path / "given", solver=solver, T=0.006, F0="[1, 0]", **BUBBLE)
    assert_last_close(load_tdata(tmp_path / "hook"), load_tdata(tmp_path / "given"),
                      [("u0m", 1e-10), ("E_kin", 1e-10)], "forcing set by a hook")


def test_plain_pressure_update(tmp_path, bubble_bdf):
    """rotational=False leaves out -mu*div(u*) in the pressure after each
    pressure solve: the passes converge to the same solution, but more slowly."""
    run_twoasis(tmp_path, solver="BDF", T=0.02, rotational=False, **BUBBLE)
    td = load_tdata(tmp_path)
    assert_mass_conserved(td, MASS_TOL)
    assert_last_close(td, bubble_bdf[1], [("u0m", 1e-6), ("E_kin", 1e-6), ("E_int", 1e-6)], "plain")
    assert passes(tmp_path) >= 2 * passes(bubble_bdf[0])


def test_anderson(tmp_path, bubble_bdf):
    """Anderson acceleration of the passes: the same solution in fewer passes."""
    run_twoasis(tmp_path, solver="BDF", T=0.02, anderson_depth=3, **BUBBLE)
    td = load_tdata(tmp_path)
    assert_mass_conserved(td, MASS_TOL)
    assert_last_close(td, bubble_bdf[1], [("u0m", 1e-6), ("E_kin", 1e-6), ("E_int", 1e-6)], "anderson")
    assert passes(tmp_path) < 0.8 * passes(bubble_bdf[0])


def test_velocity_tolerance_factor(tmp_path, bubble_bdf):
    """Velocity solves stopped relative to their initial residual: the passes
    still converge, to the same solution, in about as many passes."""
    run_twoasis(tmp_path, solver="BDF", T=0.02, velocity_tolerance_factor=0.1, **BUBBLE)
    td = load_tdata(tmp_path)
    assert_last_close(td, bubble_bdf[1], [("u0m", 1e-6), ("E_kin", 1e-6), ("E_int", 1e-6)], "tolerance")
    assert passes(tmp_path) <= 1.1 * passes(bubble_bdf[0])


def test_rotational_pressure_update_refused_by_naive(tmp_path):
    """Naive has no rotational update; asked for one, it refuses before the first
    step (by default it runs without)."""
    with pytest.raises(AssertionError, match="the Naive solver has no rotational"):
        run_twoasis(tmp_path, solver="Naive", T=0.002, rotational=True, **BUBBLE)


@pytest.mark.slow
def test_hysing_case_2_starts(tmp_path):
    """Bubble2D case=2 (densities 1000 and 1) through t = 0.02, on a 1/64 mesh.
    With the phase field and chemical potential in P2 the light phase blows up
    before then; in P1, which case=2 sets, the kinetic energy stays at 0.012."""
    run_twoasis(tmp_path, problem="Bubble2D", case=2, T=0.02, res=1/64, stat_interval=1,
                plot_interval=1000000, timestamps_interval=1000000, save_step=1000000,
                checkpoint=1000000, folder="results")
    td = load_tdata(tmp_path)
    assert np.isfinite(td).all()
    assert td[-1, 5] < 0.1          # E_kin; Bubble2D's tdata.dat has E_grav at 8
