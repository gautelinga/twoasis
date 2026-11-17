"""Shared fixtures: short coarse twoasis runs, executed once per session.

The workhorse case is PeriodicBubble2D on a 64x32 mesh (res=1/32) with the
interface parameters scaled consistently (epsilon ~ 1.6*h, M ~ epsilon^2, small
R so the periodic images do not overlap). It runs in ~10 s serial.

It comes in two variants:
  * BUBBLE: the problem's own settings, i.e. equal densities and viscosities,
    P2 velocity and BDF2. This exercises the matched-property shortcuts.
  * BUBBLE_P1: unequal densities and viscosities, P1 velocity and BDF1, the
    settings of most production problems. This exercises the general
    variable-coefficient path. P1/P1 passes converge slowly; capped at 30 a
    step, the statistics are those of 200 to 8 decimals.

SimplePore2D (PORE) covers open boundaries with pressure and phase-field
Dirichlet conditions; its mesh is generated with meshes/lattice_mesh.py.
"""
import functools
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# Children run unbound to cores: under pytest-xdist several mpirun jobs at once would
# otherwise each bind from core 0.
os.environ.setdefault("OMPI_MCA_hwloc_base_binding_policy", "none")
# Unit tests import twoasis in-process; make sure it is this checkout.
sys.path.insert(0, str(REPO_ROOT))

# Column layout of Stats/tdata.dat written by PeriodicBubble2D's temporal_hook.
# Columns 0-7 are the same for the other problems that write these statistics;
# Bubble2D, RayleighTaylor2D and TaylorBubble2D have E_grav at 8, so later
# columns differ.
COLS = dict(tstep=0, t=1, u0m=2, u1m=3, phim=4, E_kin=5, E_int=6, E_pot=7,
            vol_c=8, y_cm=9, V_c=10)

# Tests that use dolfin's objects in the test process itself, or a dolfin-only format:
# with the suite on DOLFINx (TWOASIS_BACKEND=dolfinx) they are skipped, the default run
# covers them.
dolfin_only = pytest.mark.skipif(os.environ.get("TWOASIS_BACKEND") == "dolfinx",
                                 reason="uses dolfin itself; runs with the dolfin backend")

# Write statistics every step and no other output. BUBBLE turns checkpoints
# back on, since the restart test needs them.
QUIET = dict(stat_interval=1, plot_interval=1000000, timestamps_interval=1000000,
             save_step=1000000, checkpoint=1000000, folder="results")

BUBBLE = dict(QUIET, problem="PeriodicBubble2D", dt=0.002, res=0.03125, epsilon=0.05,
              R=0.25, M=0.000125, checkpoint=10, print_velocity_pressure_convergence=True)

BUBBLE_P1 = dict(BUBBLE, rho=[1, 0.1], mu=[1, 0.1], velocity_degree=1, bdf_order=1,
                 max_iter=30, iters_on_first_timestep=30)

# Five steps of SimplePore2D on a coarse mesh (lattice_mesh.py -res 0.1).
PORE = dict(QUIET, problem="SimplePore2D", res=0.1, dt=0.02, T=0.1, timestamps_interval=5)

GOLDEN_FILE = Path(__file__).parent / "golden.json"


def pytest_addoption(parser):
    parser.addoption("--regen-golden", action="store_true", default=False,
                     help="Regenerate tests/golden.json from this platform's "
                          "results instead of comparing against it.")


def run_twoasis(workdir, nprocs=1, timeout=900, **params):
    """Run `twoasis TPfracStep key=value ...` in workdir; fail loudly."""
    cmd = ["twoasis", "TPfracStep"] + ["{}={}".format(k, v) for k, v in params.items()]
    if nprocs > 1:
        cmd = ["mpirun", "-np", str(nprocs)] + cmd
    # PYTHONPATH makes the runs exercise this checkout even when an installed
    # (non-editable) twoasis exists in site-packages.
    env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1",
               PYTHONPATH=str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run(cmd, cwd=str(workdir), env=env,
                          capture_output=True, text=True, timeout=timeout)
    (Path(workdir) / "stdout.txt").write_text(proc.stdout)
    if proc.returncode != 0:
        raise AssertionError(
            "twoasis exited with {} for {}\n--- stdout tail ---\n{}\n--- stderr tail ---\n{}".format(
                proc.returncode, " ".join(cmd), proc.stdout[-3000:], proc.stderr[-3000:]))
    return proc


def run_concurrently(*calls):
    """Call each of `calls` (without arguments) at the same time, each in a thread of
    its own (as each waits on a child process); their results, in order. An error is
    raised once all have ended."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(calls)) as pool:
        futures = [pool.submit(call) for call in calls]
    return [f.result() for f in futures]


def passes(workdir):
    """The number of inner passes of the run in workdir, whose output prints one
    line per pass (print_velocity_pressure_convergence)."""
    return (Path(workdir) / "stdout.txt").read_text().count("Iter =")


# The scheme conserves int(phi) exactly. What drift remains comes from the
# phase-field Krylov tolerance (1e-8 relative), about 1e-8 per step. A real mass
# leak is orders of magnitude larger than this bound.
MASS_TOL = 1e-6


@functools.lru_cache(maxsize=None)
def mpirun_size(nprocs=2):
    """Number of ranks dolfin sees when started with `mpirun -np nprocs`, or 0
    if mpirun is missing or fails. With a launcher that does not match the MPI
    library (e.g. MPICH built for PMIx, started with its PMI-1 hydra mpirun),
    each process runs on its own and this returns 1."""
    if shutil.which("mpirun") is None:
        return 0
    script = "from dolfin import MPI; print(MPI.size(MPI.comm_world))"
    # Pass os.environ explicitly: once dolfin has initialised MPI in this
    # process (the unit tests import it), the inherited C-level environment
    # holds MPI variables that make a nested mpirun fail. os.environ does not.
    proc = subprocess.run(["mpirun", "-np", str(nprocs), sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=300, env=dict(os.environ))
    sizes = proc.stdout.split()
    return int(sizes[0]) if proc.returncode == 0 and sizes else 0


def load_tdata(workdir, run=1):
    f = Path(workdir) / "results" / "data" / str(run) / "Stats" / "tdata.dat"
    data = np.loadtxt(str(f))
    return data.reshape(1, -1) if data.ndim == 1 else data


def once(tmp_path_factory, name, make):
    """The folder `name`, filled by `make(folder)` once per test session, also when
    pytest-xdist spreads the session over workers: the first worker to ask makes it,
    under a file lock in the session's shared temporary folder, and the others wait
    for it and use it (or fail, if making it failed)."""
    import contextlib
    base = tmp_path_factory.getbasetemp()
    shared = bool(os.environ.get("PYTEST_XDIST_WORKER"))
    if shared:
        from filelock import FileLock
        base = base.parent
    wd, done, failed = (base / (name + end) for end in ("", ".done", ".failed"))
    with FileLock(str(wd) + ".lock") if shared else contextlib.nullcontext():
        if failed.exists():
            pytest.fail("making {} failed in another worker:\n{}".format(name, failed.read_text()))
        if not done.exists():
            wd.mkdir()
            try:
                make(wd)
            except BaseException as e:
                failed.write_text(str(e)[-5000:])
                raise
            done.touch()
    return wd


def bubble_run(tmp_path_factory, name, **params):
    """A coarse bubble run made once per session: (workdir, tdata)."""
    wd = once(tmp_path_factory, name, lambda wd: run_twoasis(wd, **params))
    return wd, load_tdata(wd)


@pytest.fixture(scope="session")
def bubble_bdf(tmp_path_factory):
    """10 BDF2 steps of the coarse periodic bubble; also the restart source."""
    return bubble_run(tmp_path_factory, "bubble_bdf", solver="BDF", T=0.02, **BUBBLE)


@pytest.fixture(scope="session")
def bubble_ipcs(tmp_path_factory):
    return bubble_run(tmp_path_factory, "bubble_ipcs", solver="IPCS", T=0.02, **BUBBLE)


@pytest.fixture(scope="session")
def bubble_p1_bdf(tmp_path_factory):
    """10 BDF1 steps of the bubble with unequal properties and P1 velocity."""
    return bubble_run(tmp_path_factory, "bubble_p1_bdf", solver="BDF", T=0.02, **BUBBLE_P1)


@pytest.fixture(scope="session")
def bubble_p1_ipcs(tmp_path_factory):
    return bubble_run(tmp_path_factory, "bubble_p1_ipcs", solver="IPCS", T=0.02, **BUBBLE_P1)


@pytest.fixture(scope="session")
def bubble_bdf_long(tmp_path_factory):
    """20 uninterrupted BDF2 steps; reference for the restart test."""
    return bubble_run(tmp_path_factory, "bubble_bdf_long", solver="BDF", T=0.04, **BUBBLE)


def restarted(tmp_path_factory, name, run, prepare=lambda ckpt: None):
    """Steps 11-20 restarted from `run`'s checkpoint at step 10, on a copy of it (the
    restart appends to the run folder the checkpoint path names, which must stay
    pristine); `prepare(checkpoint folder)` first. Returns (workdir, tdata, stdout)."""
    def make(wd):
        shutil.copytree(str(Path(run[0]) / "results"), str(wd / "results"))
        ckpt = wd / "results" / "data" / "1" / "Checkpoint"
        prepare(ckpt)
        run_twoasis(wd, solver="BDF", T=0.04, restart_folder=str(ckpt), **BUBBLE)
    wd = once(tmp_path_factory, name, make)
    return wd, load_tdata(wd), (wd / "stdout.txt").read_text()


@pytest.fixture(scope="session")
def bubble_bdf_restarted(tmp_path_factory, bubble_bdf):
    """Steps 11-20 restarted from bubble_bdf's checkpoint at step 10."""
    return restarted(tmp_path_factory, "bubble_bdf_restarted", bubble_bdf)


@pytest.fixture(scope="session")
def bubble_bdf_restarted_bare_vectors(tmp_path_factory, bubble_bdf):
    """Like bubble_bdf_restarted, from a checkpoint of bare dof vectors, with
    only the current level for the pressure and the phase field.
    Returns (workdir, tdata, stdout)."""
    return restarted(tmp_path_factory, "bubble_bdf_restarted_bare_vectors", bubble_bdf,
                     lambda ckpt: to_bare_vectors(ckpt, drop_previous=("p", "phig")))


def to_bare_vectors(ckpt, drop_previous=()):
    """Rewrite a serial checkpoint as bare vectors, as HDF5File.write(vector)
    stores them: each level a bare vector in dof order (on one process, the
    Function layout's vector_0), and no previous level for the fields in
    drop_previous."""
    if importlib.util.find_spec("h5py") is None:
        pytest.skip("h5py not available")
    # Edit the files in a separate process: h5py may bundle its own HDF5
    # library, which clashes with dolfin's if both are loaded in one process.
    script = (
        "import sys, h5py\n"
        "drop = sys.argv[1].split(',')\n"
        "for f in sys.argv[2:]:\n"
        "    with h5py.File(f, 'a') as h5f:\n"
        "        for level in ('current', 'previous'):\n"
        "            v = h5f[level + '/vector_0'][...]\n"
        "            del h5f[level]\n"
        "            if level == 'current' or f.split('/')[-1][:-3] not in drop:\n"
        "                h5f[level] = v\n")
    files = sorted(str(f) for f in Path(ckpt).glob("*.h5") if not f.name.startswith("dofs_"))
    proc = subprocess.run([sys.executable, "-c", script, ",".join(drop_previous)] + files,
                          capture_output=True, text=True, timeout=60, env=dict(os.environ))
    assert proc.returncode == 0, "editing the checkpoint failed:\n" + proc.stderr[-3000:]


def copy_run(tmp_path_factory, name, run):
    """Copy a finished run into a new workdir, so that a restart (which appends
    to the run folder named by the checkpoint path) leaves the original intact.
    Returns (workdir, checkpoint folder)."""
    src_wd, _ = run
    wd = tmp_path_factory.mktemp(name)
    shutil.copytree(str(Path(src_wd) / "results"), str(Path(wd) / "results"))
    return wd, Path(wd) / "results" / "data" / "1" / "Checkpoint"


@pytest.fixture(scope="session")
def lattice_meshes(tmp_path_factory):
    """Folder with the SimplePore2D mesh for PORE, made by meshes/lattice_mesh.py."""
    pytest.importorskip("gmsh")
    pytest.importorskip("meshio")

    def make(wd):
        (wd / "meshes").mkdir()
        subprocess.run([sys.executable, str(REPO_ROOT / "meshes" / "lattice_mesh.py"),
                        "-res", str(PORE["res"])],
                       cwd=str(wd / "meshes"), check=True, capture_output=True, timeout=300)
    return once(tmp_path_factory, "lattice", make) / "meshes"


def run_pore(tmp_path_factory, lattice_meshes, solver):
    def make(wd):
        shutil.copytree(str(lattice_meshes), str(wd / "meshes"))
        run_twoasis(wd, solver=solver, **PORE)
    wd = once(tmp_path_factory, "pore_" + solver.lower(), make)
    return wd, load_tdata(wd)


@pytest.fixture(scope="session")
def pore_bdf(tmp_path_factory, lattice_meshes):
    return run_pore(tmp_path_factory, lattice_meshes, "BDF")


@pytest.fixture(scope="session")
def pore_ipcs(tmp_path_factory, lattice_meshes):
    return run_pore(tmp_path_factory, lattice_meshes, "IPCS")


def golden_compare(request, key, values, rtol=1e-4, atol=1e-7):
    """Compare a dict of scalars against tests/golden.json[key].

    Run `pytest --regen-golden` to rewrite the stored values after an
    intentional change (and eyeball the diff before committing it).
    """
    if request.config.getoption("--regen-golden"):
        data = json.loads(GOLDEN_FILE.read_text()) if GOLDEN_FILE.exists() else {}
        data[key] = values
        GOLDEN_FILE.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        pytest.skip("regenerated golden values for '{}'".format(key))
    assert GOLDEN_FILE.exists(), "tests/golden.json missing; run pytest --regen-golden"
    golden = json.loads(GOLDEN_FILE.read_text())
    assert key in golden, "no golden values for '{}'; run pytest --regen-golden".format(key)
    for name, val in values.items():
        assert_close("{}[{}]".format(key, name), val, golden[key][name], rtol, atol)


def assert_close(label, value, reference, rtol, atol=0.):
    """Assert |value - reference| <= rtol*|reference| + atol."""
    assert abs(value - reference) <= rtol * abs(reference) + atol, \
        "{} = {:.10g}, expected {:.10g} (rtol={}, atol={})".format(
            label, value, reference, rtol, atol)


def assert_last_close(td, td_ref, rtols, label):
    """Assert that two runs' statistics have the same shape, are finite, and
    agree at the last step in the columns named in rtols, each to its rtol."""
    assert td.shape == td_ref.shape
    assert np.isfinite(td).all()
    for name, rtol in rtols:
        # atol: tdata.dat holds 8 decimals
        assert_close(label + " " + name, td[-1, COLS[name]], td_ref[-1, COLS[name]], rtol, 2e-8)


def assert_mass_conserved(td, tol):
    """Assert that the mean of phi stays within tol of its initial value."""
    phim = td[:, COLS["phim"]]
    assert np.max(np.abs(phim - phim[0])) < tol


def require_mpi(nprocs=2):
    """Skip the test unless mpirun starts nprocs ranks; with TWOASIS_REQUIRE_MPI
    set (as in CI), fail instead, so that a broken MPI setup is noticed."""
    size = mpirun_size(nprocs)
    if size != nprocs:
        message = "mpirun -np {} does not start a {}-rank job here (dolfin sees {} rank(s))".format(
            nprocs, nprocs, size)
        if os.environ.get("TWOASIS_REQUIRE_MPI"):
            pytest.fail(message)
        pytest.skip(message)
