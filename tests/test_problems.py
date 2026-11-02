"""Smoke tests: every problem runs two timesteps with both solvers (slow; run
with `pytest -m slow`).

These only check that a problem sets up and steps without errors, which catches
problem files that have fallen out of sync with the solvers.
"""
import pytest

from conftest import QUIET, run_twoasis

# Problems that build their own mesh.
PROBLEMS = ["Bubble2D", "BubbleOpen", "OscBubble2D", "PeriodicBubble2D", "PlaneCouette",
            "RayleighTaylor2D", "TaylorBubble2D", "TaylorGreen2D"]

# Problems that read a mesh file, which has to be generated first with the
# scripts in meshes/.
MESH_PROBLEMS = ["BeadContact3D", "CylArr2D", "Porous2D", "Porous2DFlux", "Porous3D",
                 "Porous3DFlux", "SimplePore2D", "Tube3D"]


@pytest.mark.slow
@pytest.mark.parametrize("solver", ["BDF", "IPCS"])
@pytest.mark.parametrize("problem", PROBLEMS + [
    pytest.param(p, marks=pytest.mark.skip(reason="needs a mesh file from meshes/"))
    for p in MESH_PROBLEMS])
def test_problem_runs(tmp_path, problem, solver):
    run_twoasis(tmp_path, problem=problem, solver=solver, dt=1e-4, T=2e-4, **QUIET)
