"""Shared fixtures: short coarse twoasis runs, executed once per session.

The workhorse case is PeriodicBubble2D on a 64x32 mesh (res=1/32) with the
interface parameters scaled consistently (epsilon ~ 1.6*h, M ~ epsilon^2, small
R so the periodic images do not overlap). It runs in ~10 s serial and exercises
the matched-property fast paths (rho and mu are matched in this problem).
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

# Column layout of Stats/tdata.dat written by PeriodicBubble2D's temporal_hook.
# (TaylorBubble2D writes more columns, but the shared ones come first.)
COLS = dict(tstep=0, t=1, u0m=2, u1m=3, phim=4, E_kin=5, E_int=6, E_pot=7,
            vol_c=8, y_cm=9, V_c=10)

# Write statistics every step and no other output. BUBBLE turns checkpoints
# back on, since the restart test needs them.
QUIET = dict(stat_interval=1, plot_interval=1000000, timestamps_interval=1000000,
             save_step=1000000, checkpoint=1000000, folder="results")

BUBBLE = dict(QUIET, problem="PeriodicBubble2D", dt=0.002, res=0.03125, epsilon=0.05,
              R=0.25, M=0.000125, checkpoint=10)

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
    repo_root = Path(__file__).resolve().parent.parent
    env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1",
               PYTHONPATH=str(repo_root) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run(cmd, cwd=str(workdir), env=env,
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise AssertionError(
            "twoasis exited with {} for {}\n--- stdout tail ---\n{}\n--- stderr tail ---\n{}".format(
                proc.returncode, " ".join(cmd), proc.stdout[-3000:], proc.stderr[-3000:]))
    return proc


def load_tdata(workdir, run=1):
    f = Path(workdir) / "results" / "data" / str(run) / "Stats" / "tdata.dat"
    data = np.loadtxt(str(f))
    return data.reshape(1, -1) if data.ndim == 1 else data


@pytest.fixture(scope="session")
def bubble_bdf(tmp_path_factory):
    """10 BDF2 steps of the coarse periodic bubble; also the restart source."""
    wd = tmp_path_factory.mktemp("bubble_bdf")
    run_twoasis(wd, solver="BDF", T=0.02, **BUBBLE)
    return wd, load_tdata(wd)


@pytest.fixture(scope="session")
def bubble_ipcs(tmp_path_factory):
    wd = tmp_path_factory.mktemp("bubble_ipcs")
    run_twoasis(wd, solver="IPCS", T=0.02, **BUBBLE)
    return wd, load_tdata(wd)


@pytest.fixture(scope="session")
def bubble_bdf_long(tmp_path_factory):
    """20 uninterrupted BDF2 steps; reference for the restart test."""
    wd = tmp_path_factory.mktemp("bubble_bdf_long")
    run_twoasis(wd, solver="BDF", T=0.04, **BUBBLE)
    return wd, load_tdata(wd)


@pytest.fixture(scope="session")
def bubble_bdf_restarted(tmp_path_factory, bubble_bdf):
    """Steps 11-20 restarted from bubble_bdf's checkpoint at step 10.

    The restart machinery appends to the run folder named by the checkpoint
    path (data/1), so run on a copy to keep the bubble_bdf fixture pristine.
    """
    src_wd, _ = bubble_bdf
    wd = tmp_path_factory.mktemp("bubble_bdf_restarted")
    shutil.copytree(str(Path(src_wd) / "results"), str(Path(wd) / "results"))
    ckpt = Path(wd) / "results" / "data" / "1" / "Checkpoint"
    run_twoasis(wd, solver="BDF", T=0.04, restart_folder=str(ckpt), **BUBBLE)
    return wd, load_tdata(wd)


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


def assert_mass_conserved(td, tol):
    """Assert that the mean of phi stays within tol of its initial value."""
    phim = td[:, COLS["phim"]]
    assert np.max(np.abs(phim - phim[0])) < tol
