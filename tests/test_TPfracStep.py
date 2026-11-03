"""Regression tests for the TPfracStep two-phase solvers.

Fast, coarse runs of the periodic bubble (see conftest.py for the two
variants) checking:
  * runs complete and produce finite statistics (BDF, IPCS, serial + MPI),
  * mass conservation: the xi=1 row of the phase-field system is exact, so
    int(phi) may drift only at the Krylov-tolerance level,
  * golden values: selected scalars after 10 steps match tests/golden.json
    (regenerate deliberately with `pytest --regen-golden`),
  * BDF and IPCS agree on the same case to discretization accuracy,
  * Naive, the plain reference implementation of IPCS's scheme, agrees with
    IPCS to solver tolerance,
  * checkpoint/restart reproduces an uninterrupted run, also from a checkpoint
    that lacks the previous time level, leaves no temporary checkpoint folders,
    and refuses a checkpoint written by another number of processes.
"""
import shutil
from os import path

import numpy as np
import pytest

from conftest import (BUBBLE, BUBBLE_P1, COLS, MASS_TOL, QUIET, assert_close,
                      assert_mass_conserved, copy_run, golden_compare, load_tdata,
                      require_mpi, run_twoasis)

# Scalars pinned by the golden tests.
GOLDEN_COLS = ["u0m", "phim", "E_kin", "E_int", "vol_c"]


@pytest.fixture(params=["bubble_bdf", "bubble_ipcs", "bubble_p1_bdf", "bubble_p1_ipcs"])
def bubble_run(request):
    """(golden key, workdir, tdata) for each bubble variant and solver."""
    wd, td = request.getfixturevalue(request.param)
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


@pytest.mark.parametrize("variant, rtols", [
    # BDF2 against IPCS (first order): they differ at O(dt).
    ("bubble", [("u0m", 2e-2), ("phim", 1e-5), ("E_int", 1e-3)]),
    # BDF1 against IPCS: both first order, but they scale the momentum equation
    # by the density differently, so they differ where the density varies.
    ("bubble_p1", [("u0m", 1e-3), ("phim", 1e-6), ("E_int", 1e-5)]),
])
def test_bdf_ipcs_agree(request, variant, rtols):
    """Same case, different time discretizations."""
    _, td_b = request.getfixturevalue(variant + "_bdf")
    _, td_i = request.getfixturevalue(variant + "_ipcs")
    for name, rtol in rtols:
        assert_close("IPCS " + name, td_i[-1, COLS[name]], td_b[-1, COLS[name]], rtol)


def test_naive_ipcs_agree(tmp_path):
    """Naive writes IPCS's scheme as plain UFL forms solved with LU, so the two
    differ only by IPCS's Krylov tolerances (1e-6 relative for the velocity and
    the pressure). Measured on this case, the statistics agree to the 8 decimals
    they are written with. The case has unequal properties, so the density and
    viscosity weighting and the grad(mu) term are exercised. A fixed number of
    inner iterations (max_error=0) makes both solvers take the same steps; it is
    cut to 10 to keep Naive fast, which is enough for these three steps.
    """
    params = dict(BUBBLE_P1, T=0.006, max_iter=10, iters_on_first_timestep=10, max_error=0)
    td = {}
    for solver in ("Naive", "IPCS"):
        wd = tmp_path / solver
        wd.mkdir()
        run_twoasis(wd, solver=solver, **params)
        td[solver] = load_tdata(wd)
    assert td["Naive"].shape == td["IPCS"].shape == (3, td["IPCS"].shape[1])
    cols = [COLS[name] for name in ("u0m", "phim", "E_kin", "E_int", "E_pot", "vol_c")]
    # atol: the rounding of the written statistics.
    np.testing.assert_allclose(td["Naive"][:, cols], td["IPCS"][:, cols], rtol=1e-6, atol=2e-8)


def test_restart_continuation(bubble_bdf_restarted, bubble_bdf_long):
    """Steps 11-20 after a restart must match the uninterrupted run.

    The checkpoint stores both time levels (current and previous), so BDF2
    resumes with its full history. Any remaining difference is at the level of
    the Krylov tolerances.
    """
    _, td_r, stdout = bubble_bdf_restarted
    _, td_l = bubble_bdf_long
    assert "BDF1 on first timestep: False" in stdout  # the previous level was restored
    assert td_r.shape[0] == 20  # 10 original + 10 appended rows
    for name in GOLDEN_COLS:
        assert_close("restarted " + name, td_r[-1, COLS[name]], td_l[-1, COLS[name]],
                     1e-4, 1e-7)


def test_checkpoint_folder(bubble_bdf_restarted):
    """The restarted run swaps its checkpoints into place and leaves no
    temporary folders behind."""
    wd, _, _ = bubble_bdf_restarted
    run = wd / "results" / "data" / "1"
    assert {f.name for f in (run / "Checkpoint").iterdir()} == \
        {"params.dat", "p.h5", "phig.h5", "u0.h5", "u1.h5"}
    assert not (run / "Checkpoint.new").exists() and not (run / "Checkpoint.old").exists()


def test_restart_from_a_copied_checkpoint(tmp_path_factory, bubble_bdf):
    """A checkpoint copied away from its run, under another name: the restart
    writes next to it, its own checkpoint at step 20 into a new Checkpoint/,
    and leaves the copy alone."""
    wd, ckpt = copy_run(tmp_path_factory, "restart_copied", bubble_bdf)
    copied = wd / "elsewhere" / "Checkpoint_10"
    shutil.copytree(str(ckpt), str(copied))
    run_twoasis(wd, solver="BDF", T=0.04, restart_folder=str(copied), **BUBBLE)
    assert {f.name for f in (wd / "elsewhere").iterdir()} == \
        {"Checkpoint", "Checkpoint_10", "Stats", "Timeseries", "subdomains.h5", "subdomains.xdmf"}
    assert np.loadtxt(str(wd / "elsewhere" / "Stats" / "tdata.dat")).shape[0] == 10


def test_restart_time_series(tmp_path):
    """The time series and statistics across restarts: a run saves past its last checkpoint
    (step 10), a restart from that checkpoint replaces the later steps, and a
    second restart from the same step writes files of its own (_r1) and replaces
    the first restart's steps. The merged series is one history, and every step
    in it refers to data that exists."""
    import h5py
    from femtools.xdmf import grids
    params = dict(BUBBLE, solver="BDF", checkpoint=10)
    restart = dict(params, save_step=1, restart_folder="results/data/1/Checkpoint")
    run_twoasis(tmp_path, T=0.03, **dict(params, save_step=5))   # saves t = 0, 0.01, 0.02, 0.03
    run_twoasis(tmp_path, T=0.026, **restart)                     # steps 11-13 from step 10
    run_twoasis(tmp_path, T=0.024, **restart)                     # steps 11-12 from step 10 again
    series = tmp_path / "results" / "data" / "1" / "Timeseries"
    assert sorted(f.name for f in series.glob("u_*.xdmf")) == ["u_from_tstep_0.xdmf"]
    steps = grids(str(series / "u_from_tstep_0.xdmf"))
    assert [round(g["time"], 6) for g in steps] == [0., 0.01, 0.02, 0.022, 0.024]
    assert [path.basename(g["attributes"]["u"][0]) for g in steps] == \
        ["u_from_tstep_0.h5"] * 3 + ["u_from_tstep_10_r1.h5"] * 2
    for g in steps:
        h5file, dataset = g["attributes"]["u"]
        with h5py.File(h5file, "r") as f:
            assert dataset in f
    # The statistics too: one row a step, the last restart's from step 11 on.
    assert list(load_tdata(tmp_path)[:, 0]) == list(range(1, 13))


@pytest.mark.parametrize("nprocs", [1, 2])
def test_kill_file(tmp_path, nprocs):
    """A file killtwoasis in the output folder stops the run cleanly where it
    is seen (here at once: the initial state is saved before the first step),
    with a checkpoint there, and is removed."""
    import pickle
    if nprocs > 1:
        require_mpi(nprocs)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "killtwoasis").touch()
    run_twoasis(tmp_path, nprocs=nprocs, solver="BDF", T=1.0, **BUBBLE)
    run = tmp_path / "results" / "data" / "1"
    with open(str(run / "Checkpoint" / "params.dat"), "rb") as f:
        assert pickle.load(f)["tstep"] == 0
    assert not (run / "Stats" / "tdata.dat").exists()
    assert not (tmp_path / "results" / "killtwoasis").exists()


def test_unknown_parameter_is_refused(tmp_path):
    """A misspelt parameter stops the run, with the name it was meant to be."""
    with pytest.raises(AssertionError, match="unknown parameter chekpoint.*did you mean checkpoint"):
        run_twoasis(tmp_path, solver="BDF", T=0.004, chekpoint=5, **BUBBLE)


def test_restart_takes_the_problem_from_the_checkpoint(tmp_path_factory, bubble_bdf):
    """A restart without problem= continues the checkpoint's problem."""
    wd, ckpt = copy_run(tmp_path_factory, "restart_no_problem", bubble_bdf)
    params = {k: v for k, v in BUBBLE.items() if k != "problem"}
    proc = run_twoasis(wd, solver="BDF", T=0.024, restart_folder=str(ckpt), **params)
    assert "PeriodicBubble2D" in proc.stdout
    assert load_tdata(wd).shape[0] == 12


def test_restart_other_process_count(tmp_path_factory, bubble_bdf):
    """A checkpoint written in serial cannot be read on two processes (the dof
    order differs), so the restart stops with a clear error."""
    require_mpi(2)
    wd, ckpt = copy_run(tmp_path_factory, "restart_np2", bubble_bdf)
    with pytest.raises(AssertionError, match="written by 1 processes"):
        run_twoasis(wd, nprocs=2, solver="BDF", T=0.04, restart_folder=str(ckpt), **BUBBLE)


def test_restart_old_format(bubble_bdf_restarted_old_format, bubble_bdf_long):
    """Restart from a checkpoint without the previous level of p and phi.

    The run must notice the missing level, take a first-order step, and then
    continue close to the uninterrupted run (which used BDF2 throughout).
    """
    _, td_r, stdout = bubble_bdf_restarted_old_format
    _, td_l = bubble_bdf_long
    assert "BDF1 on first timestep: True" in stdout
    assert td_r.shape[0] == 20
    assert np.isfinite(td_r).all()
    assert_mass_conserved(td_r, MASS_TOL)
    for name, rtol in [("u0m", 1e-2), ("phim", 1e-6), ("E_int", 1e-4)]:
        assert_close("restarted " + name, td_r[-1, COLS[name]], td_l[-1, COLS[name]], rtol)


def test_parallel(tmp_path, bubble_bdf):
    """Two-rank run: completes, conserves mass, agrees with serial."""
    require_mpi(2)
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
