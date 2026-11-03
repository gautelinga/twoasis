"""Smoke tests: every problem runs two timesteps with both solvers (slow; run
with `pytest -m slow`), and a quick run of a configuration that once crashed.

These only check that a problem sets up and steps without errors, which catches
problem files that have fallen out of sync with the solvers. Also: TaylorBubble2D's
controller across a restart.
"""
import numpy as np
import pytest

from conftest import QUIET, load_tdata, run_twoasis

# Problems that build their own mesh. PeriodicBubble2D (the bubble fixtures),
# TaylorBubble2D (test_taylorbubble_smoke) and TaylorGreen2D (test_convergence)
# run elsewhere with both solvers.
PROBLEMS = ["Bubble2D", "BubbleOpen", "LaplaceBubble2D", "OscBubble2D", "PlaneCouette",
            "RayleighTaylor2D"]

# Problems that read a mesh file, which has to be generated first with the
# scripts in meshes/ (or Bernaise's, for Porous2D, Porous2DFlux and CylArr2D).
MESH_PROBLEMS = ["BeadContact3D", "CylArr2D", "Porous2D", "Porous2DFlux", "Porous3D",
                 "Porous3DFlux", "SimplePore2D", "Tube3D"]


@pytest.mark.slow
@pytest.mark.parametrize("solver", ["BDF", "IPCS"])
@pytest.mark.parametrize("problem", PROBLEMS + [
    pytest.param(p, marks=pytest.mark.skip(reason="needs a mesh file from meshes/"))
    for p in MESH_PROBLEMS])
def test_problem_runs(tmp_path, problem, solver):
    run_twoasis(tmp_path, problem=problem, solver=solver, dt=1e-4, T=2e-4, **QUIET)


def test_bubble_without_gravity(tmp_path):
    """Bubble2D's statistics with g0 = 0, where the gravitational energy's
    integrand vanishes."""
    run_twoasis(tmp_path, problem="Bubble2D", g0="[0,0]", res=0.125, dt=1e-4, T=2e-4, **QUIET)


def test_taylorbubble_controller_restart(tmp_path):
    """TaylorBubble2D's controller continues across a restart: the driving
    pressure gradient (last column) of steps 11-20 equals that of an
    uninterrupted run."""
    params = dict(QUIET, problem="TaylorBubble2D", res=0.0625, dt=0.02, regulator="y_cm",
                  stat_interval=2, checkpoint=10)
    (tmp_path / "long").mkdir()
    (tmp_path / "restart").mkdir()
    run_twoasis(tmp_path / "long", T=0.4, **params)
    run_twoasis(tmp_path / "restart", T=0.2, **params)
    run_twoasis(tmp_path / "restart", T=0.4, restart_folder="results/data/1/Checkpoint", **params)
    long, restarted = (load_tdata(tmp_path / d) for d in ("long", "restart"))
    assert restarted.shape == long.shape == (10, long.shape[1])
    assert np.allclose(restarted[:, -1], long[:, -1], rtol=1e-6)
