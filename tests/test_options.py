"""Solver options that the default runs do not use. Each is compared with the
corresponding default run, so that an option that stops working is noticed."""
import numpy as np
import pytest

from conftest import (BUBBLE, BUBBLE_P1, COLS, MASS_TOL, assert_close, assert_mass_conserved,
                      load_tdata, run_twoasis)


def compare(td, td_ref, rtols, label):
    assert td.shape == td_ref.shape
    assert np.isfinite(td).all()
    for name, rtol in rtols:
        # atol: tdata.dat holds 8 decimals
        assert_close(label + " " + name, td[-1, COLS[name]], td_ref[-1, COLS[name]], rtol, 2e-8)


def test_phase_field_fieldsplit(tmp_path, bubble_bdf):
    """The Schur fieldsplit preconditioner solves the same phase-field system;
    results differ only at the level of the Krylov tolerances."""
    run_twoasis(tmp_path, solver="BDF", T=0.02, phase_field_fieldsplit=True, **BUBBLE)
    td = load_tdata(tmp_path)
    assert_mass_conserved(td, MASS_TOL)
    compare(td, bubble_bdf[1], [("u0m", 1e-3), ("phim", 1e-6), ("E_int", 1e-5)], "fieldsplit")


def test_consistent_mass_velocity_update(tmp_path, bubble_p1_bdf):
    """method='default' (consistent mass) instead of the default lumped mass for
    P1. The velocity update only applies the last pressure increment of the
    inner iterations, which is small, so the results stay close."""
    run_twoasis(tmp_path, solver="BDF", T=0.02,
                velocity_update_solver='{"method": "default"}', **BUBBLE_P1)
    compare(load_tdata(tmp_path), bubble_p1_bdf[1],
            [("u0m", 1e-5), ("phim", 1e-8), ("E_int", 1e-7)], "consistent mass")


def test_ab_projection_pressure(tmp_path, bubble_bdf):
    """Extrapolating the pressure as the first guess changes only where the
    inner iterations start, not what they converge to."""
    run_twoasis(tmp_path, solver="BDF", T=0.02, AB_projection_pressure=True, **BUBBLE)
    compare(load_tdata(tmp_path), bubble_bdf[1],
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
    compare(td, bubble_p1_bdf[1], [("E_int", 2e-3)], "P1 phase field")
