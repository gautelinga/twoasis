"""Checkpoints across the backends: a checkpoint written on one backend (on one process)
restarts on the other (on three), with BDF2 from the first step, and continues as an
uninterrupted run on the other backend does. Both write dolfin's HDF5 layout
(feluga.dolfin_layout, feluga.dolfinx.checkpoint)."""
import importlib.util

import pytest

from conftest import BUBBLE, assert_close, COLS, load_tdata, require_mpi, run_twoasis

pytestmark = pytest.mark.skipif(importlib.util.find_spec("dolfinx") is None,
                                reason="needs DOLFINx")

PARAMS = dict(BUBBLE, solver="BDF")


@pytest.mark.parametrize("writer,reader", [("dolfin", "dolfinx"), ("dolfinx", "dolfin")])
def test_restart_on_the_other_backend(tmp_path, monkeypatch, writer, reader):
    require_mpi(3)
    monkeypatch.delenv("TWOASIS_BACKEND", raising=False)
    (tmp_path / "long").mkdir()
    (tmp_path / "ckpt").mkdir()
    run_twoasis(tmp_path / "long", backend=reader, T=0.024, **dict(PARAMS, checkpoint=1000))
    run_twoasis(tmp_path / "ckpt", backend=writer, T=0.02, **dict(PARAMS, checkpoint=10))
    ckpt = tmp_path / "ckpt" / "results" / "data" / "1" / "Checkpoint"
    proc = run_twoasis(tmp_path / "ckpt", nprocs=3, backend=reader, T=0.024,
                       restart_folder=str(ckpt), **PARAMS)
    assert "Backend: " + reader in proc.stdout
    assert "BDF1 on first timestep: False" in proc.stdout
    td, td_long = load_tdata(tmp_path / "ckpt"), load_tdata(tmp_path / "long")
    assert list(td[:, 0]) == list(td_long[:, 0])
    # The first ten steps ran on the other backend: the backends agree to round-off,
    # which shows in the 8th printed decimal.
    for name in COLS:
        assert_close("{} -> {} {}".format(writer, reader, name), td[-1, COLS[name]],
                     td_long[-1, COLS[name]], 1e-6, 1e-7)


def test_auto_follows_a_dolfinx_checkpoint(tmp_path, monkeypatch):
    """backend=auto (the default) restarts a DOLFINx checkpoint on DOLFINx."""
    monkeypatch.delenv("TWOASIS_BACKEND", raising=False)
    run_twoasis(tmp_path, backend="dolfinx", T=0.02, **dict(PARAMS, checkpoint=10))
    ckpt = tmp_path / "results" / "data" / "1" / "Checkpoint"
    proc = run_twoasis(tmp_path, T=0.022, restart_folder=str(ckpt), **PARAMS)
    assert "Backend: dolfinx" in proc.stdout
