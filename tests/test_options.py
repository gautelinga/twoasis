"""Solver options that the default runs do not use. Each is compared with the
corresponding default run, so that an option that stops working is noticed."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from conftest import (BUBBLE, BUBBLE_P1, COLS, MASS_TOL, PORE, QUIET, assert_last_close,
                      assert_mass_conserved, golden_compare, load_tdata, passes, require_mpi,
                      run_concurrently, run_twoasis)
from twoasis.common import telemetry


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


def test_gradient_matrix_velocity_update(tmp_path, bubble_p1_bdf):
    """method='gradient_matrix' (the cells' mean gradients averaged onto the velocity
    dofs, a matrix built once) instead of the consistent-mass projection: as with
    lumping, only the last pressure increment goes through it, so the results stay
    close."""
    run_twoasis(tmp_path, solver="BDF", T=0.02,
                velocity_update_solver='{"method": "gradient_matrix"}', **BUBBLE_P1)
    td = load_tdata(tmp_path)
    assert_mass_conserved(td, MASS_TOL)
    assert_last_close(td, bubble_p1_bdf[1],
                      [("u0m", 1e-5), ("phim", 1e-8), ("E_int", 1e-7)], "gradient matrix")


def test_ab_projection_pressure(tmp_path, bubble_bdf):
    """Extrapolating the pressure as the first guess changes only where the
    inner iterations start, not what they converge to."""
    run_twoasis(tmp_path, solver="BDF", T=0.02, AB_projection_pressure=True, **BUBBLE)
    assert_last_close(load_tdata(tmp_path), bubble_bdf[1],
                      [("u0m", 1e-4), ("phim", 1e-7), ("E_int", 1e-6)], "AB projection")


@pytest.mark.parametrize("solver", ["BDF", "IPCS"])
def test_lower_degree_phase_field(tmp_path, bubble_p1_bdf, solver):
    """P2 velocity with the phase field and chemical potential in P1, their default
    degree: mass is conserved, and the interface energy is close to that of the all-P1
    run."""
    run_twoasis(tmp_path, solver=solver, T=0.02, **dict(BUBBLE_P1, velocity_degree=2))
    td = load_tdata(tmp_path)
    assert np.isfinite(td).all()
    assert_mass_conserved(td, MASS_TOL)
    assert_last_close(td, bubble_p1_bdf[1], [("E_int", 2e-3)], "P1 phase field")


def test_velocity_degree_phase_field(tmp_path, request):
    """The phase field and chemical potential in the velocity's degree, P2, asked for.
    With equal densities and viscosities the bulk capillary forces the P1 pressure cannot
    cancel stay small: mass is conserved, and the statistics match their golden values."""
    run_twoasis(tmp_path, solver="BDF", T=0.02, phase_field_degree=2, **BUBBLE)
    td = load_tdata(tmp_path)
    assert np.isfinite(td).all()
    assert_mass_conserved(td, MASS_TOL)
    golden_compare(request, "bubble_p2_phase_field_bdf",
                   {name: float(td[-1, COLS[name]])
                    for name in ["u0m", "phim", "E_kin", "E_int", "vol_c"]})


SET_FORCING = '''
from twoasis.backend import Constant
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
    before then; in P1, the default, the kinetic energy stays at 0.012."""
    run_twoasis(tmp_path, problem="Bubble2D", case=2, T=0.02, res=1/64, stat_interval=1,
                plot_interval=1000000, timestamps_interval=1000000, save_step=1000000,
                checkpoint=1000000, folder="results")
    td = load_tdata(tmp_path)
    assert np.isfinite(td).all()
    assert td[-1, 5] < 0.1          # E_kin; Bubble2D's tdata.dat has E_grav at 8


def test_unconverged_passes_warned_only_with_passes(tmp_path):
    """The warning that a step's inner passes stopped unconverged is for runs that allow
    more than one pass; with max_iter=1 every step would print it."""
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    warning = "inner passes without converging"
    run_twoasis(one, solver="BDF", T=0.006, max_iter=1, iters_on_first_timestep=1, **BUBBLE)
    assert warning not in (one / "stdout.txt").read_text()
    run_twoasis(two, solver="BDF", T=0.006, max_iter=2, iters_on_first_timestep=2,
                max_error=1e-30, max_error_rel=1e-30, **BUBBLE)
    assert (two / "stdout.txt").read_text().count(warning) == 3


def test_diverging_passes_stop_the_run(tmp_path):
    """Passes whose velocity change grows diverge (here Hysing's case 1 with a P2 phase
    field, at ten times its step on a coarse mesh, with the plain convection form): the
    run stops with an error, not with garbage. With the skew-symmetric form this case
    also breaks down, but its passes stop growing before the guard's five passes in a
    row and the phase-field solve fails a step later instead."""
    with pytest.raises(AssertionError, match="the inner passes diverge"):
        run_twoasis(tmp_path, problem="Bubble2D", res=0.0625, dt=0.02, T=0.13, max_iter=50,
                    iters_on_first_timestep=50, phase_field_degree=2, skew_convection=False,
                    **QUIET)


def test_preconditioner_steps(tmp_path):
    """Preconditioners rebuilt every fifth step only (the pressure's in Hysing's case 1,
    whose pressure matrix changes every step): the passes converge to the same
    solution with fewer setups."""
    every, fifth = tmp_path / "every", tmp_path / "fifth"
    common = dict(QUIET, problem="Bubble2D", res=1 / 32, T=0.02, telemetry=True)
    for wd, steps in ((every, 1), (fifth, 5)):
        wd.mkdir()
        run_twoasis(wd, pressure_preconditioner_steps=steps, velocity_preconditioner_steps=steps,
                    **common)
    setups = []
    for wd in (every, fifth):
        records = telemetry.read(str(wd / "results" / "data" / "1" / "Stats" / "telemetry.jsonl"))
        setups.append(sum(r["solvers"]["pressure"]["pc_setups"] for r in records))
    assert setups == [10, 2]
    assert_last_close(load_tdata(fifth), load_tdata(every),
                      [("u0m", 1e-6), ("E_kin", 1e-6), ("E_int", 1e-6)], "preconditioner steps")


def test_divergence_stats(bubble_bdf):
    """Stats/divergence.dat: |div u| and |grad u| every stat_interval steps (each step
    in BUBBLE), the velocity nearly divergence free."""
    d = np.loadtxt(str(bubble_bdf[0] / "results" / "data" / "1" / "Stats" / "divergence.dat"))
    assert d.shape == (10, 4)
    assert (d[:, 0] == np.arange(1, 11)).all()
    assert np.isfinite(d).all() and (d[:, 2] < d[:, 3]).all()


# Converged steps for comparing the coupled velocity-pressure solve with the passes it
# replaces: both tight enough that what remains of their difference is the Krylov noise.
# The phase field too: on several processes its solve is not bit-reproducible, and at
# its default tolerance two identical runs differ by ~1e-6 after a step.
TIGHT_PHASE_FIELD = dict(
    phase_field_krylov_solver='{"relative_tolerance": 1e-12, "maximum_iterations": 5000}',
    krylov_solvers='{"absolute_tolerance": 1e-15}')
TIGHT_PASSES = dict(
    TIGHT_PHASE_FIELD, max_error=1e-13, max_iter=400, iters_on_first_timestep=400,
    velocity_krylov_solver='{"relative_tolerance": 1e-12, "maximum_iterations": 2000}',
    pressure_krylov_solver='{"relative_tolerance": 1e-12, "maximum_iterations": 500}')
TIGHT_COUPLED = dict(
    TIGHT_PHASE_FIELD, velocity_pressure_solve="coupled",
    coupled_krylov_solver='{"relative_tolerance": 1e-12}')


def checkpoint_fields(wd):
    """The velocity, the mean-free pressure and the phase field of the run in wd, from
    its checkpoint, cell by cell in the order of the global cell numbers (each run on
    several processes numbers its dofs its own way; the checkpoint's dof maps say which
    dofs each cell has)."""
    import h5py
    ckpt = Path(wd) / "results" / "data" / "1" / "Checkpoint"
    fields = {}
    for name, dofs in (("u0", "u0"), ("u1", "u0"), ("p", "p"), ("phig", "phig")):
        with h5py.File(str(ckpt / (name + ".h5")), "r") as f:
            v = f["current/vector_0"][...]
        with h5py.File(str(ckpt / ("dofs_" + dofs + ".h5")), "r") as f:
            cells, offsets, cell_dofs = (f["dofs/" + k][...] for k in
                                         ("cells", "x_cell_dofs", "cell_dofs"))
        per_cell = cell_dofs.reshape(-1, offsets[1] - offsets[0])[np.argsort(cells)]
        fields[name] = v[per_cell].ravel()
    fields["p"] = fields["p"] - fields["p"].mean()
    return fields


def assert_fields_close(wd, wd_ref, rtol):
    a, b = checkpoint_fields(wd), checkpoint_fields(wd_ref)
    for name in a:
        diff = np.linalg.norm(a[name] - b[name]) / np.linalg.norm(b[name])
        assert diff < rtol, "{}: relative difference {:.2e}".format(name, diff)


@pytest.mark.parametrize("nprocs", [1, 2])
def test_coupled_equals_converged_passes(tmp_path, nprocs):
    """velocity_pressure_solve='coupled' solves the system the passes converge to: on the
    periodic bubble (P2-P1, walls, the pressure fixed only up to a constant), converged
    passes and the coupled solve give the same step, in a few iterations against tens of
    passes."""
    if nprocs > 1:
        require_mpi(nprocs)
    passes_wd, coupled_wd = tmp_path / "passes", tmp_path / "coupled"
    common = dict(BUBBLE, solver="BDF", T=0.02, checkpoint=10, telemetry=True)
    for wd in (passes_wd, coupled_wd):
        wd.mkdir()
    run_concurrently(lambda: run_twoasis(passes_wd, nprocs, **dict(common, **TIGHT_PASSES)),
                     lambda: run_twoasis(coupled_wd, nprocs, **dict(common, **TIGHT_COUPLED)))
    assert_fields_close(coupled_wd, passes_wd, 1e-8)
    records = telemetry.read(str(coupled_wd / "results" / "data" / "1" / "Stats" /
                                 "telemetry.jsonl"))
    its = [r["solvers"]["coupled"]["its"] for r in records]
    assert len(its) == 10 and max(its) < 30, its
    assert passes(coupled_wd) == 0


def test_coupled_with_prescribed_pressure(tmp_path, lattice_meshes):
    """The coupled solve with the pressure prescribed at both ends (SimplePore2D, P2-P1):
    the prescribed rows are the identity, and the step is that of converged passes."""
    passes_wd, coupled_wd = tmp_path / "passes", tmp_path / "coupled"
    common = dict(PORE, solver="BDF", velocity_degree=2, checkpoint=5)
    for wd in (passes_wd, coupled_wd):
        shutil.copytree(str(lattice_meshes), str(wd / "meshes"))
    run_concurrently(lambda: run_twoasis(passes_wd, **dict(common, **TIGHT_PASSES)),
                     lambda: run_twoasis(coupled_wd, **dict(common, **TIGHT_COUPLED)))
    assert_fields_close(coupled_wd, passes_wd, 1e-8)


@pytest.mark.parametrize("params, message", [
    (dict(BUBBLE_P1, solver="BDF"), "needs an inf-sup stable pair"),
    (dict(BUBBLE, solver="IPCS"), "the IPCS solver has none")])
def test_coupled_refused(tmp_path, params, message):
    """The coupled solve needs an inf-sup stable pair (with P1-P1 the coupled system is
    not stabilised) and the BDF solver; otherwise the run refuses before the first step."""
    with pytest.raises(AssertionError, match=message):
        run_twoasis(tmp_path, T=0.002, velocity_pressure_solve="coupled", **params)


@pytest.mark.slow
def test_coupled_iterations_independent_of_mesh(tmp_path):
    """The coupled solve's iterations do not grow as the mesh is refined: Hysing's case
    1 at h = 1/32 and 1/128, five steps at the same dt."""
    its = []
    for res in (32, 128):
        wd = tmp_path / "res{}".format(res)
        wd.mkdir()
        run_twoasis(wd, problem="Bubble2D", res=1 / res, dt=0.002, T=0.0099, telemetry=True,
                    velocity_pressure_solve="coupled", **QUIET)
        records = telemetry.read(str(wd / "results" / "data" / "1" / "Stats" /
                                     "telemetry.jsonl"))
        its.append([r["solvers"]["coupled"]["its"] for r in records])
    coarse, fine = its
    assert len(fine) == 5 and max(fine) <= 1.5 * max(coarse) and max(fine) < 25, its
