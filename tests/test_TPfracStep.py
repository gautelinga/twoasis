"""Regression tests for the TPfracStep two-phase solvers.

Fast, coarse runs (see conftest.py for the case definition) checking:
  * runs complete and produce finite statistics (BDF, IPCS, serial + MPI),
  * mass conservation: the xi=1 row of the phase-field system is exact, so
    int(phi) may drift only at the Krylov-tolerance level,
  * golden values: selected scalars after 10 steps match tests/golden.json
    (regenerate deliberately with `pytest --regen-golden`),
  * BDF and IPCS agree on the same case to discretization accuracy,
  * checkpoint/restart reproduces an uninterrupted run.
"""
import shutil

import numpy as np
import pytest

from conftest import (BUBBLE, COLS, QUIET, assert_close, assert_mass_conserved,
                      golden_compare, load_tdata, run_twoasis)

# Scalars pinned by the golden tests.
GOLDEN_COLS = ["u0m", "phim", "E_kin", "E_int", "vol_c"]

# The scheme conserves int(phi) exactly. What drift remains comes from the
# phase-field Krylov tolerance (1e-8 relative), about 1e-8 per step. A real mass
# leak is orders of magnitude larger than this bound.
MASS_TOL = 1e-6


@pytest.fixture(params=["bubble_bdf", "bubble_ipcs"])
def bubble_run(request, bubble_bdf, bubble_ipcs):
    """(golden key, workdir, tdata) for each solver."""
    wd, td = {"bubble_bdf": bubble_bdf, "bubble_ipcs": bubble_ipcs}[request.param]
    return request.param, wd, td


def test_finite(bubble_run):
    _, _, td = bubble_run
    assert td.shape[0] == 10  # one stats row per step
    assert np.isfinite(td).all()
    # A blow-up shows in E_kin long before it produces NaNs.
    assert td[-1, COLS["E_kin"]] < 1.0


def test_mass_conservation(bubble_run):
    _, _, td = bubble_run
    assert_mass_conserved(td, MASS_TOL)


def test_golden(request, bubble_run):
    key, _, td = bubble_run
    values = {name: float(td[-1, COLS[name]]) for name in GOLDEN_COLS}
    golden_compare(request, key, values)


def test_bdf_ipcs_agree(bubble_bdf, bubble_ipcs):
    """Same case, different time discretizations: agreement to O(dt)."""
    _, td_b = bubble_bdf
    _, td_i = bubble_ipcs
    for name, rtol in [("u0m", 2e-2), ("phim", 1e-5), ("E_int", 1e-3)]:
        assert_close("IPCS " + name, td_i[-1, COLS[name]], td_b[-1, COLS[name]], rtol)


def test_restart_continuation(bubble_bdf_restarted, bubble_bdf_long):
    """Steps 11-20 after a restart must match the uninterrupted run.

    The checkpoint stores both time levels (current and previous), so BDF2
    resumes with its full history. Any remaining difference is at the level of
    the Krylov tolerances.
    """
    _, td_r = bubble_bdf_restarted
    _, td_l = bubble_bdf_long
    assert td_r.shape[0] == 20  # 10 original + 10 appended rows
    for name in GOLDEN_COLS:
        assert_close("restarted " + name, td_r[-1, COLS[name]], td_l[-1, COLS[name]],
                     1e-4, 1e-7)


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun not available")
def test_parallel(tmp_path, bubble_bdf):
    """Two-rank run: completes, conserves mass, agrees with serial."""
    run_twoasis(tmp_path, nprocs=2, solver="BDF", T=0.02, **BUBBLE)
    td = load_tdata(tmp_path)
    assert np.isfinite(td).all()
    assert_mass_conserved(td, MASS_TOL)
    _, td_s = bubble_bdf
    for name in ["u0m", "phim", "E_int"]:
        assert_close("np2 " + name, td[-1, COLS[name]], td_s[-1, COLS[name]], 1e-3, 1e-7)


@pytest.mark.parametrize("solver", ["BDF", "IPCS"])
def test_taylorbubble_smoke(tmp_path, solver):
    """Two steps of TaylorBubble2D at problem defaults: exercises walls,
    contact-angle terms and per-component velocity BCs (the Ai branch)."""
    run_twoasis(tmp_path, problem="TaylorBubble2D", solver=solver,
                T=0.04, dt=0.02, **QUIET)
    td = load_tdata(tmp_path)
    assert np.isfinite(td).all()
    assert_mass_conserved(td, 1e-5)
