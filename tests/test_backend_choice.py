"""Choosing the backend (twoasis/backend/__init__.py): backend= on the command line, else
TWOASIS_BACKEND, else auto, which follows a restart's checkpoint and is dolfin otherwise."""
import importlib.util
import os
import pickle
import subprocess
import sys

import pytest

from conftest import BUBBLE, REPO_ROOT, run_twoasis
from twoasis.backend import choose


def params_dat(folder, **params):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "params.dat", "wb") as f:
        pickle.dump(dict(params), f)
    return folder


def test_default_is_dolfin():
    assert choose([], {}) == "dolfin"
    assert choose(["backend=auto"], {}) == "dolfin"


def test_command_line_then_environment():
    assert choose(["backend=dolfinx"], {}) == "dolfinx"
    assert choose([], {"TWOASIS_BACKEND": "dolfinx"}) == "dolfinx"
    assert choose(["backend=dolfin"], {"TWOASIS_BACKEND": "dolfinx"}) == "dolfin"
    assert choose(['backend="dolfinx"'], {}) == "dolfinx"


def test_auto_follows_the_checkpoint(tmp_path):
    x = params_dat(tmp_path / "x", backend="dolfinx", T=1.0)
    old = params_dat(tmp_path / "old", T=1.0)          # written before backends were recorded
    assert choose(["restart_folder=" + str(x)], {}) == "dolfinx"
    assert choose(["restart_folder=" + str(old)], {}) == "dolfin"
    assert choose(["restart_folder=" + str(tmp_path / "none")], {}) == "dolfin"
    assert choose(["restart_folder=" + str(x), "backend=dolfin"], {}) == "dolfin"


def test_unknown_backend_is_refused():
    with pytest.raises(ValueError):
        choose(["backend=fenics"], {})


@pytest.mark.skipif(importlib.util.find_spec("dolfinx") is None, reason="needs DOLFINx")
def test_run_on_the_command_line_choice(tmp_path, monkeypatch):
    monkeypatch.delenv("TWOASIS_BACKEND", raising=False)
    proc = run_twoasis(tmp_path, solver="BDF", T=0.004, backend="dolfinx",
                       **dict(BUBBLE, checkpoint=1000000))
    assert "Backend: dolfinx" in proc.stdout
    (params,) = tmp_path.glob("results/data/*/Timeseries/params.dat")
    with open(params, "rb") as f:
        assert pickle.load(f)["backend"] == "dolfinx"


@pytest.mark.skipif(importlib.util.find_spec("dolfinx") is None, reason="needs DOLFINx")
def test_dolfinx_run_imports_no_legacy_fenics(tmp_path):
    """A run on DOLFINx with dolfin, ufl_legacy, ffc, dijitso and FIAT unimportable."""
    args = ["TPfracStep", "backend=dolfinx", "T=0.004", "folder=results", "stat_interval=1",
            "checkpoint=1000000", "telemetry=true"]
    args += ["{}={}".format(k, v) for k, v in BUBBLE.items()
             if k not in ("folder", "checkpoint", "stat_interval")]
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=str(REPO_ROOT))
    env.pop("TWOASIS_BACKEND", None)
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "tests" / "nodolfin_run.py")] + args,
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
    assert list(tmp_path.glob("results/data/*/Stats/tdata.dat"))
