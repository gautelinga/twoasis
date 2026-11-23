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


def test_adaptive_dt(tmp_path):
    """dt_adapt (solvers/TPfracStep/adapt.py): from a small first step the rising bubble's
    dt grows by at most 2 a step, the measured errors stay near their tolerance once it
    has, the last step ends at T, and the result stays near the fixed-dt run's."""
    fixed, adaptive = tmp_path / "fixed", tmp_path / "adaptive"
    common = dict(QUIET, problem="Bubble2D", res=1 / 32, T=0.02, telemetry=True)
    fixed.mkdir()
    run_twoasis(fixed, dt=0.001, **common)
    adaptive.mkdir()
    run_twoasis(adaptive, dt=0.0002,
                dt_adapt='{"tol_x": 1e-3, "rtol_u": 1e-2, "atol_u": 1e-4, "dt_max": 0.004}',
                **common)
    records = telemetry.read(str(adaptive / "results" / "data" / "1" / "Stats" /
                                 "telemetry.jsonl"))
    dts = np.array([r["dt"] for r in records])
    assert dts[0] == pytest.approx(0.0002) and dts.max() > 5 * dts[0]
    assert np.all(dts[1:] / dts[:-1] <= 2 + 1e-12)
    assert records[-1]["t"] == pytest.approx(0.02, abs=1e-12)
    assert all(r["retries"] == 0 for r in records)
    e = [r["e"] for r in records[5:] if r["e"] is not None]
    assert max(e) < 2.0
    td, td_ref = load_tdata(adaptive), load_tdata(fixed)
    assert td[-1, COLS["t"]] == pytest.approx(td_ref[-1, COLS["t"]])
    for name, rtol in (("u0m", 5e-2), ("E_int", 1e-3), ("y_cm", 1e-3)):
        assert td[-1, COLS[name]] == pytest.approx(td_ref[-1, COLS[name]], rel=rtol), name


def test_coupled_chns_implicit_physical_pressure(tmp_path, lattice_meshes):
    """The coupled CH-NS step (chns_solve="coupled") on PoreThroat2D's open ends: with
    the physical pressure's boundary rows implicit in g a meniscus released at dt 102.4,
    far past the meniscus's own period, settles to rest."""
    params = dict(QUIET, problem="PoreThroat2D", Lx=4.0, res=PORE["res"], epsilon=0.1,
                  M=1e-3, rho=[1, 1], dp_start=0.8, dp=0.8, chns_solve="coupled",
                  dt=102.4, T=12 * 102.4, stat_interval=1)
    shutil.copytree(str(lattice_meshes), str(tmp_path / "meshes"))
    run_twoasis(tmp_path, **params)
    rows = np.loadtxt(str(tmp_path / "results" / "data" / "1" / "Stats" / "throat.dat"))
    assert len(rows) >= 12 and np.isfinite(rows[:, 5]).all()
    assert rows[-1, 5] < 1e-3 * rows[0, 5]


def test_coupled_chns_flow_solve(tmp_path, lattice_meshes):
    """The iterative coupled CH-NS step's flow solve (monolithic.py): with its tolerance
    against the outer residual and the projection's pressure as its first guess, a
    meniscus at dt 102.4 takes a quarter of the outer iteration limit at most (with the
    tolerance against the flow solve's own right-hand side and a zero guess it stopped at
    the limit), and ends at the direct solve's result."""
    params = dict(QUIET, problem="PoreThroat2D", Lx=4.0, res=PORE["res"], epsilon=0.1,
                  M=1e-3, rho=[1.4, 1], dp_start=0.8, dp=0.8, velocity_pressure_solve="coupled",
                  chns_solve="coupled", dt=102.4, T=4.5 * 102.4, telemetry=True,
                  checkpoint=4)
    runs = {"direct": '{"method": "direct"}', "default": '{}'}
    for name in runs:
        shutil.copytree(str(lattice_meshes), str(tmp_path / name / "meshes"))
    run_concurrently(*[lambda n=n, c=c: run_twoasis(tmp_path / n, chns_solver=c, **params)
                       for n, c in runs.items()])
    data = tmp_path / "{}" / "results" / "data" / "1"
    its = [r["solvers"]["chns"]["its"] for r in
           telemetry.read(str(data).format("default") + "/Stats/telemetry.jsonl")]
    assert len(its) >= 4 and max(its) < 200 / 4, its
    ref, f = checkpoint_fields(tmp_path / "direct"), checkpoint_fields(tmp_path / "default")
    err = {k: np.linalg.norm(f[k] - ref[k]) / np.linalg.norm(ref[k]) for k in f}
    assert max(err.values()) < 2e-4, err


class FakeKSP:
    """What the phase-field preconditioner's reuse functions use of a petsc4py KSP."""

    class PC:
        flag = None

        def setReusePreconditioner(self, flag):
            self.flag = flag

    def __init__(self):
        self.handle = object()
        self.pc = self.PC()

    def getPC(self):
        return self.pc


def test_reuse_phase_field_preconditioner():
    """The phase field's preconditioner is kept only when asked and while alpha0 stays
    within 10 % of the alpha0 it was built with (an adaptive dt rebuilds it)."""
    from twoasis.solvers.TPfracStep import reuse_phase_field_preconditioner as reuse

    ksp = FakeKSP()
    steps = [(False, 1.0, False),   # first build
             (True, 1.05, True),    # within 10 %
             (False, 1.05, False),  # a rebuild step: built at 1.05 now
             (True, 1.2, False),    # alpha0 moved by > 10 %: rebuilt at 1.2
             (True, 1.25, True)]
    for ask, alpha, keep in steps:
        assert reuse(ksp, ask, alpha) is keep, (ask, alpha)
        assert ksp.pc.flag is keep


def test_phase_field_preconditioner_steps(tmp_path):
    """phase_field_preconditioner_steps=5: the phase field's preconditioner is built on
    the first step of every five, and on step 2, where alpha0 changes from BDF1's to
    BDF2's; the run's result is unchanged (the Krylov solves converge to the same
    tolerance)."""
    every, fifth = tmp_path / "every", tmp_path / "fifth"
    common = dict(QUIET, problem="Bubble2D", res=1 / 32, T=0.02, telemetry=True,
                  velocity_pressure_solve="coupled")
    for wd, steps in ((every, 1), (fifth, 5)):
        wd.mkdir()
    run_concurrently(lambda: run_twoasis(every, phase_field_preconditioner_steps=1, **common),
                     lambda: run_twoasis(fifth, phase_field_preconditioner_steps=5, **common))
    built = []
    for wd in (every, fifth):
        records = telemetry.read(str(wd / "results" / "data" / "1" / "Stats" / "telemetry.jsonl"))
        built.append([r["tstep"] for r in records if r["solvers"]["phase_field"]["pc_setups"]])
    assert built == [list(range(1, 11)), [1, 2, 6]]
    assert_last_close(load_tdata(fifth), load_tdata(every),
                      [("u0m", 1e-6), ("E_kin", 1e-6), ("E_int", 1e-6)],
                      "phase field preconditioner steps")


@pytest.mark.parametrize("chns_solve", ["split", "coupled"])
def test_extrapolated_initial_guess(tmp_path, chns_solve):
    """initial_guess="extrapolated" (the BDF extrapolation as the Krylov solves' first
    guess): at the same tolerance the result is closer to a tight-tolerance run than with
    the previous step's solution as the guess (the stopping test is relative to the
    initial residual, so a closer guess ends closer)."""
    common = dict(BUBBLE, solver="BDF", T=0.02, velocity_pressure_solve="coupled",
                  chns_solve=chns_solve)
    if chns_solve == "coupled":
        common["chns_solver"] = '{"method": "iterative"}'
    tol = '{"relative_tolerance": %g, "absolute_tolerance": 1e-20, "maximum_iterations": 2000}'
    wds = {k: tmp_path / k for k in ("ref", "previous", "extrapolated")}
    for wd in wds.values():
        wd.mkdir()
    run_concurrently(
        lambda: run_twoasis(wds["ref"], krylov_solvers=tol % 1e-12, **common),
        lambda: run_twoasis(wds["previous"], initial_guess="previous",
                            krylov_solvers=tol % 1e-6, **common),
        lambda: run_twoasis(wds["extrapolated"], initial_guess="extrapolated",
                            krylov_solvers=tol % 1e-6, **common))
    ref = checkpoint_fields(wds["ref"])
    err = {}
    for k in ("previous", "extrapolated"):
        f = checkpoint_fields(wds[k])
        err[k] = max(np.linalg.norm(f[n] - ref[n]) / np.linalg.norm(ref[n]) for n in f)
    assert err["previous"] < 1e-4 and err["extrapolated"] < err["previous"] / 2, err


def test_adaptive_dt_ramp():
    """dt_adapt's step ratio: within [1/ramp, ramp] while the measure fast_cut_on names is
    <= fast_cut, cut by up to shrink above it, unchanged within the deadband; the last two
    steps before T equal. By default only e_phi cuts fast."""
    from twoasis.solvers.TPfracStep.adapt import DEFAULTS, Controller
    c = Controller.__new__(Controller)      # next_dt needs only the configuration
    c.cfg = dict(DEFAULTS, dt_min=1e-6, fast_cut_on="e")
    c.ceiling = None
    ramp, T = DEFAULTS["ramp"], 1e9
    assert c.next_dt(1.0, 1e-6, 0.0, T) == pytest.approx(ramp)          # small e: grow gently
    assert c.next_dt(1.0, 1.5, 0.0, T) == pytest.approx(1 / ramp)       # e above 1: shrink gently
    assert c.next_dt(1.0, 0.81, 0.0, T) == 1.0                          # f = 1.0: the deadband
    assert c.next_dt(1.0, 3.0, 0.0, T) == pytest.approx(0.9 / 3 ** 0.5)  # e > fast_cut: cut
    assert c.next_dt(1.0, 100.0, 0.0, T) == pytest.approx(DEFAULTS["shrink"])
    assert c.next_dt(1.0, 1e-6, 0.0, 1.5) == pytest.approx(0.75)        # 1.5 left: two of 0.75
    assert c.next_dt(0.75, 1e-6, 0.75, 1.5) == pytest.approx(0.75)
    c.cfg["fast_cut_on"] = DEFAULTS["fast_cut_on"]                     # e_phi; e = e_u: gentle
    c.last = dict(e=3.0, e_phi=0.5, e_u=3.0)
    assert c.next_dt(1.0, 3.0, 0.0, T) == pytest.approx(1 / ramp)
    c.last = dict(e=3.0, e_phi=3.0, e_u=0.5)                           # e = e_phi: cut
    assert c.next_dt(1.0, 3.0, 0.0, T) == pytest.approx(0.9 / 3 ** 0.5)


def test_adaptive_dt_limit():
    """next_dt's `limit` (chns_solve="auto"'s cap) gets the proposed dt and the step the
    error allows (`wanted`, safety e^(-1/2) dt), and acts before the last steps are evened
    out."""
    from twoasis.solvers.TPfracStep.adapt import DEFAULTS, Controller
    c = Controller.__new__(Controller)
    c.cfg = dict(DEFAULTS, dt_min=1e-6)
    c.ceiling = None
    c.last = dict(e=0.25, e_phi=0.25, e_u=0.25)
    seen = []
    assert c.next_dt(1.0, 0.25, 0.0, 1e9, limit=lambda n, w: seen.append((n, w)) or 0.5) == 0.5
    assert seen == [(pytest.approx(DEFAULTS["ramp"]), pytest.approx(0.9 / 0.25 ** 0.5))]
    assert c.wanted == pytest.approx(1.8)
    # 1.2 left, capped at 1.0: two steps of 0.6, not 1.0 and 0.2
    assert c.next_dt(1.0, 0.25, 0.0, 1.2, limit=lambda n, w: min(n, 1.0)) == pytest.approx(0.6)
    c.next_dt(1.0, None, 0.0, 1e9)
    assert c.wanted is None


def test_chns_switch_rules():
    """chns_solve="auto"'s choice of step (switch.py): split at the start when dt <=
    switch_down dt_split_max, its dt capped there; to coupled only when the cap holds the
    split step back and the step the error allows exceeds cost_ratio times the cap; back
    to split at a dt <= switch_down times the cap; at most one switch in `hold` steps;
    on_switch told each switch; the mode given (a restart's) kept."""
    from twoasis.solvers.TPfracStep.switch import Switch

    class Quick(Switch):
        cost_ratio, hold = 2.0, 3

    told = []
    s = Quick(10.0, 1.0, on_switch=told.append)
    assert s.mode == "split"
    assert s.next_dt(5.0, 50.0) == 5.0 and s.mode == "split"      # below the cap: split
    assert s.next_dt(11.0, 15.0) == 10.0 and s.mode == "split"    # capped; wanted < 2 cap
    assert s.next_dt(11.0, 25.0) == 11.0 and s.mode == "coupled"  # capped, wanted > 2 cap
    assert s.next_dt(5.0, 5.0) == 5.0 and s.mode == "coupled"     # held: 1 step after
    assert s.next_dt(5.0) == 5.0 and s.mode == "coupled"          # 2
    assert s.next_dt(9.5) == 9.5 and s.mode == "coupled"          # 9.5 > 0.9 cap: coupled
    assert s.next_dt(5.0) == 5.0 and s.mode == "split"            # back
    assert told == ["coupled", "split"] and s.switches == 2
    assert Switch(10.0, 9.5).mode == "coupled"
    assert Switch(10.0, 50.0, mode="split").mode == "split"
    fixed = Switch(10.0, 1.0)
    assert [fixed.next_dt(1.0) for _ in range(30)] == [1.0] * 30 and fixed.switches == 0
    for bad in (None, 0.0):
        with pytest.raises(ValueError):
            Switch(bad, 1.0)
    with pytest.raises(ValueError):
        Switch(1.0, 1.0, mode="passes")


def test_forget_phase_field_preconditioner():
    """After forget_phase_field_preconditioner the phase field's preconditioner is
    rebuilt on its next solve even at the alpha0 it was built with (a switch of
    chns_solve="auto" changes the operator, not alpha0)."""
    from twoasis.solvers.TPfracStep import (forget_phase_field_preconditioner as forget,
                                           reuse_phase_field_preconditioner as reuse)

    ksp = FakeKSP()
    assert reuse(ksp, False, 1.0) is False
    assert reuse(ksp, True, 1.0) is True
    forget(ksp)
    assert reuse(ksp, True, 1.0) is False
    assert reuse(ksp, True, 1.0) is True


def test_chns_solver_settings():
    """chns_solver's settings are completed with monolithic.DEFAULTS; an unknown key or
    value is refused."""
    from twoasis.solvers.TPfracStep.monolithic import DEFAULTS, settings
    assert settings({}) == DEFAULTS
    assert settings(dict(method="direct", relative_tolerance=1e-10))["inner"] == "krylov"
    with pytest.raises(ValueError, match="unknown keys"):
        settings(dict(inner_tolerence=1e-2))
    with pytest.raises(ValueError, match="method"):
        settings(dict(method="lu"))


def test_check_chns_options():
    """The CH-NS step's options are refused when unknown or when they do not fit: the
    coupled steps have no dual mesh, dt_split_max is chns_solve="auto"'s."""
    from twoasis.solvers.TPfracStep import check_chns_options as check
    check("split", "previous", None, 1)
    check("auto", "extrapolated", 25.6, 0)
    for args, message in ((("stabilised", "previous", None, 0), "chns_solve"),
                          (("coupled", "previous", None, 1), "dual mesh"),
                          (("coupled", "previous", 25.6, 0), "dt_split_max"),
                          (("split", "last", None, 0), "initial_guess")):
        with pytest.raises(ValueError, match=message):
            check(*args)


@pytest.mark.parametrize("method", ["direct", "iterative"])
def test_chns_auto(tmp_path, method):
    """chns_solve="auto" on the rising bubble, dt_adapt from a small first step: split
    steps until the controller wants more than dt_split_max, then coupled ones, one
    switch (telemetry's mode); the split steps' phase-field solves keep their tolerance
    (an iterative coupled step has a phase-field solver of its own); the result stays
    near the coupled run's. A restart from a checkpoint taken in coupled mode continues
    coupled, and with dt held below the cap switches to split."""
    adapt = '{"tol_x": 1e-3, "rtol_u": 1e-2, "atol_u": 1e-4, "dt_max": 0.004}'
    cap = 0.0004    # the switch to coupled comes when the error allows 2.5 cap = 0.001
    common = dict(QUIET, problem="Bubble2D", res=1 / 32, T=0.02, telemetry=True, dt=0.0002,
                  dt_adapt=adapt, chns_solver='{"method": "%s"}' % method)
    auto, coupled = tmp_path / "auto", tmp_path / "coupled"
    for wd in (auto, coupled):
        wd.mkdir()
    run_concurrently(
        lambda: run_twoasis(auto, chns_solve="auto", dt_split_max=cap,
                            **dict(common, checkpoint=30)),
        lambda: run_twoasis(coupled, chns_solve="coupled", **common))
    stats = "results/data/1/Stats/telemetry.jsonl"
    records = telemetry.read(str(auto / stats))
    modes = [r["mode"] for r in records]
    first = modes.index("coupled")
    assert 5 < first < 30 and set(modes[:first]) == {"split"}, modes
    assert set(modes[first:]) == {"coupled"}, modes
    dts = np.array([r["dt"] for r in records])
    assert dts[:first].max() <= cap * (1 + 1e-12) and dts[first:].max() > cap
    assert all("chns" not in r["solvers"] or not r["solvers"]["chns"]["solves"]
               for r in records[:first])
    assert min(r["solvers"]["phase_field"]["its"] for r in records[5:first]) >= 5
    td, td_ref = load_tdata(auto), load_tdata(coupled)
    assert td[-1, COLS["t"]] == pytest.approx(td_ref[-1, COLS["t"]])
    for name, rtol in (("u0m", 5e-2), ("E_int", 1e-3), ("y_cm", 1e-3)):
        assert td[-1, COLS[name]] == pytest.approx(td_ref[-1, COLS[name]], rel=rtol), name
    # the restart (from step 30, into the same run folder): coupled first (the
    # checkpoint's step was), then split below the cap (dt_max merges into the
    # checkpoint's dt_adapt)
    run_twoasis(auto, restart_folder="results/data/1/Checkpoint", T=0.016, dt=0.0003,
                dt_adapt='{"dt_max": 0.0003}', **{
                    k: v for k, v in common.items() if k not in ("T", "dt", "dt_adapt")})
    records = telemetry.read(str(auto / stats))
    start = len(records) - 1 - [r["tstep"] for r in records][::-1].index(31)  # restart
    modes = [r["mode"] for r in records[start:]]
    assert modes[0] == "coupled" and modes[1:] and set(modes[1:]) == {"split"}, modes
