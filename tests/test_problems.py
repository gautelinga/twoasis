"""Smoke tests: every problem runs two timesteps with both solvers (slow; run
with `pytest -m slow`), and a quick run of a configuration that once crashed.

These only check that a problem sets up and steps without errors, which catches
problem files that have fallen out of sync with the solvers. The problems that
read a mesh file run on small meshes made here (meshgen.py, and
meshes/tube_mesh.py for Tube3D), two of them also on two processes. Also:
TaylorBubble2D's controller across a restart.
"""
import subprocess
import sys

import numpy as np
import pytest

from conftest import QUIET, REPO_ROOT, load_tdata, require_mpi, run_twoasis

# Problems that build their own mesh. PeriodicBubble2D (the bubble fixtures),
# TaylorBubble2D (test_taylorbubble_smoke) and TaylorGreen2D (test_convergence)
# run elsewhere with both solvers.
PROBLEMS = ["Bubble2D", "BubbleOpen", "LaplaceBubble2D", "OscBubble2D", "PlaneCouette",
            "RayleighTaylor2D"]



def porous2d(meshes):
    """A periodic square with two disks (Porous2D's walls are its obstacles)."""
    from meshgen import obstacle_mesh, write_obstacles
    from twoasis.problems.TPfracStep.Porous2D import get_fname
    params = dict(Lx=2, Ly=2, rad=0.3, R=0.5, N=2, res=0.15)
    obstacles = [(-0.4, -0.3, 0.3), (0.5, 0.5, 0.25)]
    obstacle_mesh(meshes / get_fname(ext="h5", **params)[len("meshes/"):], (-1, -1), (1, 1),
                  obstacles, (True, True), params["res"])
    write_obstacles(meshes / get_fname(ext="dat", **params)[len("meshes/"):], obstacles)
    return params


def cylarr2d(meshes):
    """CylArr2D's cell (Lx wide, sqrt(3)*Lx high) with one cylinder in the middle."""
    from meshgen import obstacle_mesh, write_obstacles
    from twoasis.problems.TPfracStep.CylArr2D import get_fname
    params = dict(Lx=1, rad=0.33, res=0.1)
    obstacles = [(0, 0, params["rad"])]
    half = (0.5, np.sqrt(3) / 2)
    obstacle_mesh(meshes / get_fname(ext="h5", **params)[len("meshes/"):],
                  (-half[0], -half[1]), half, obstacles, (True, True), params["res"])
    write_obstacles(meshes / get_fname(ext="dat", **params)[len("meshes/"):], obstacles)
    return params


def porous3d(meshes):
    """The unit cube, periodic on all axes, with one sphere."""
    from meshgen import obstacle_mesh, write_obstacles
    from twoasis.problems.TPfracStep.Porous3D import get_fname
    obstacles = [(0.5, 0.5, 0.5, 0.25)]
    obstacle_mesh(meshes / get_fname(1, 0, 0, 0, "h5")[len("meshes/"):], (0, 0, 0), (1, 1, 1),
                  obstacles, (True, True, True), 0.2)
    write_obstacles(meshes / get_fname(1, 0, 0, 0, "obst")[len("meshes/"):], obstacles)
    return dict(L=1)


def porous3dflux(meshes):
    """A column periodic in x and y, flow along z, with one sphere."""
    from meshgen import obstacle_mesh, write_obstacles
    obstacles = [(0.5, 0.5, 1.0, 0.3)]
    obstacle_mesh(meshes / "column.h5", (0, 0, 0), (1, 1, 2), obstacles, (True, True, False), 0.25)
    write_obstacles(meshes / "column.obst", obstacles)
    return dict(meshfile="meshes/column.h5")


def beadcontact3d(meshes):
    """A bead between two plates, periodic in x and y."""
    from meshgen import obstacle_mesh
    from twoasis.problems.TPfracStep.BeadContact3D import get_fname
    params = dict(Lx=1.5, Ly=2.0, H=1.0, R=0.35, res=4)
    obstacle_mesh(meshes / get_fname(ext="h5", **params)[len("meshes/"):], (-0.75, -1, -0.5),
                  (0.75, 1, 0.5), [(0, 0, 0, params["R"])], (True, True, False), 0.25)
    return params


def tube3d(meshes):
    """A tube periodic along z, made by meshes/tube_mesh.py."""
    subprocess.run([sys.executable, str(REPO_ROOT / "meshes" / "tube_mesh.py"),
                    "-L", "2", "-R", "0.5", "-res", "0.25"],
                   cwd=str(meshes), check=True, capture_output=True, timeout=300)
    return dict(meshfile="meshes/tube.h5", Lb=1, Rb=0.3)


# Problems that read a mesh file, with what makes a small one in a meshes/
# folder and returns the parameters that point the problem to it. Porous2DFlux
# reads Porous2D's mesh. SimplePore2D runs in test_open_boundary.
MESH_PROBLEMS = dict(BeadContact3D=beadcontact3d, CylArr2D=cylarr2d, Porous2D=porous2d,
                     Porous2DFlux=porous2d, Porous3D=porous3d, Porous3DFlux=porous3dflux,
                     Tube3D=tube3d)


def with_mesh(tmp_path, problem):
    meshes = tmp_path / "meshes"
    meshes.mkdir()
    return MESH_PROBLEMS.get(problem, lambda meshes: {})(meshes)


@pytest.mark.slow
@pytest.mark.parametrize("solver", ["BDF", "IPCS"])
@pytest.mark.parametrize("problem", PROBLEMS + sorted(MESH_PROBLEMS))
def test_problem_runs(tmp_path, problem, solver):
    params = with_mesh(tmp_path, problem)
    run_twoasis(tmp_path, problem=problem, solver=solver, dt=1e-4, T=2e-4, **QUIET, **params)


@pytest.mark.slow
@pytest.mark.parametrize("problem", ["Porous2D", "Porous3D"])
def test_mesh_problem_in_parallel(tmp_path, problem):
    """On two processes: the mesh read (Porous3D: partitioned by femtools) and the
    periodic partition repaired and checked."""
    require_mpi(2)
    params = with_mesh(tmp_path, problem)
    run_twoasis(tmp_path, nprocs=2, problem=problem, dt=1e-4, T=2e-4, **QUIET, **params)
    assert np.isfinite(load_tdata(tmp_path)).all()


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


def test_timestamps_across_a_restart(tmp_path):
    """A restart writes the step it starts from to Timestamps/ again: it is listed
    once. A run stopped after its last checkpoint has listed later steps too; a
    restart from that checkpoint drops them, list entries and files, so that
    timestamps.dat is one run's history."""
    params = dict(QUIET, problem="Porous2D", dt=1e-4, timestamps_interval=2, checkpoint=4,
                  **with_mesh(tmp_path, "Porous2D"))
    run = tmp_path / "results" / "data" / "1"
    stamps = run / "Timestamps" / "timestamps.dat"

    def listed():
        return [line.split()[1] for line in stamps.read_text().splitlines()]

    run_twoasis(tmp_path, T=6e-4, **params)      # checkpoint at step 4, stamps to 6
    assert listed() == ["up_{}.h5".format(n) for n in (0, 2, 4, 6)], stamps.read_text()
    restart = dict(params, restart_folder="results/data/1/Checkpoint")
    run_twoasis(tmp_path, T=5e-4, **restart)     # from step 4, stopping at 5
    assert listed() == ["up_{}.h5".format(n) for n in (0, 2, 4)], stamps.read_text()
    assert not (run / "Timestamps" / "up_6.h5").exists()
    run_twoasis(tmp_path, T=8e-4, **restart)     # from step 4 again, on to 8
    assert listed() == ["up_{}.h5".format(n) for n in (0, 2, 4, 6, 8)], stamps.read_text()
