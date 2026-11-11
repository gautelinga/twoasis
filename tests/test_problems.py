"""Smoke tests: every problem runs two timesteps with both solvers (slow; run
with `pytest -m slow`), and a quick run of a configuration that once crashed.

These only check that a problem sets up and steps without errors, which catches
problem files that have fallen out of sync with the solvers. The problems that
read a mesh file run on small meshes made here (meshgen.py, and
meshes/tube_mesh.py for Tube3D), two of them also restarted in parallel. Also:
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
    from meshgen import obstacle_mesh
    from twoasis.problems.TPfracStep.Porous2D import get_fname
    params = dict(Lx=2, Ly=2, rad=0.3, R=0.5, N=2, res=0.15)
    obstacle_mesh(meshes / get_fname(ext="h5", **params)[len("meshes/"):], (-1, -1), (1, 1),
                  [(-0.4, -0.3, 0.3), (0.5, 0.5, 0.25)], (True, True), params["res"])
    return params


def cylarr2d(meshes):
    """CylArr2D's cell (Lx wide, sqrt(3)*Lx high) with one cylinder in the middle."""
    from meshgen import obstacle_mesh
    from twoasis.problems.TPfracStep.CylArr2D import get_fname
    params = dict(Lx=1, rad=0.33, res=0.1)
    half = (0.5, np.sqrt(3) / 2)
    obstacle_mesh(meshes / get_fname(ext="h5", **params)[len("meshes/"):],
                  (-half[0], -half[1]), half, [(0, 0, params["rad"])], (True, True), params["res"])
    return params


def porous3d(meshes):
    """The unit cube, periodic on all axes, with one sphere."""
    from meshgen import obstacle_mesh
    from twoasis.problems.TPfracStep.Porous3D import get_fname
    obstacle_mesh(meshes / get_fname(1, 0, 0, 0, "h5")[len("meshes/"):], (0, 0, 0), (1, 1, 1),
                  [(0.5, 0.5, 0.5, 0.25)], (True, True, True), 0.2)
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
@pytest.mark.parametrize("problem,nprocs", [("Porous2D", 2), ("Porous3D", 3)])
def test_mesh_problem_restart_on_other_process_count(tmp_path, problem, nprocs):
    """A checkpoint written in serial, restarted on nprocs processes (the mesh
    read and its periodic partition repaired, Porous3D: partitioned by
    femtools), continues as the uninterrupted serial run does. The checkpoint is
    the run's second, whose dof tables are those of its first, linked."""
    require_mpi(nprocs)
    params = dict(QUIET, problem=problem, dt=1e-4, checkpoint=2, **with_mesh(tmp_path, problem))
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "meshes").symlink_to(tmp_path / "meshes")
    run_twoasis(reference, T=6e-4, **params)
    run_twoasis(tmp_path, T=4e-4, **params)
    run_twoasis(tmp_path, nprocs=nprocs, T=6e-4,
                restart_folder="results/data/1/Checkpoint", **params)
    td, td_ref = load_tdata(tmp_path), load_tdata(reference)
    assert td.shape == td_ref.shape
    np.testing.assert_allclose(td, td_ref, rtol=1e-6, atol=1e-10)


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
    timestamps.dat is one run's history. Each run writes the dof tables once, to
    dofs_<its first step>.h5, and its files link to them."""
    import h5py
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
    assert sorted(f.name for f in (run / "Timestamps").glob("dofs_*")) == \
        ["dofs_0.h5", "dofs_4.h5"]
    for n, tables in ((2, "dofs_0.h5"), (8, "dofs_4.h5")):
        with h5py.File(str(run / "Timestamps" / "up_{}.h5".format(n)), "r") as h5:
            for name in ("u", "p", "phi"):
                link = h5.get(name + "/cell_dofs", getlink=True)
                assert isinstance(link, h5py.ExternalLink) and link.filename == tables
                assert h5[name + "/cell_dofs"].size > 0      # read through the link


def test_partrac_case_walls(tmp_path):
    """CylArr2D's partrac case: Timestamps/boundaries.xdmf holds exactly the
    exterior facets marked 1 (the no-slip walls), dolfin_params.dat names them
    for wall_upgrade, and 1 and 3 processes write the same facets."""
    import shutil
    import dolfin
    from femtools.boundaries import read
    from partractools.case import read_params
    from twoasis.problems.TPfracStep.CylArr2D import get_fname
    from twoasis.problems.TPfracStep.shared import load_mesh, mark_walls
    require_mpi(3)
    params = with_mesh(tmp_path, "CylArr2D")
    mesh = load_mesh(str(tmp_path / get_fname(ext="h5", **params)))
    marked = dolfin.MeshFunction("size_t", mesh, 1, 0)
    mark_walls(marked, mesh)
    exterior = dolfin.MeshFunction("size_t", mesh, 1, 0)
    dolfin.DomainBoundary().mark(exterior, 1)
    expected = int(((marked.array() == 1) & (exterior.array() == 1)).sum())
    facets = {}
    for nprocs in (1, 3):
        wd = tmp_path / "np{}".format(nprocs)
        shutil.copytree(str(tmp_path / "meshes"), str(wd / "meshes"))
        run_twoasis(wd, nprocs=nprocs, problem="CylArr2D", dt=1e-4, T=2e-4, **QUIET, **params)
        case = wd / "results" / "data" / "1" / "Timestamps"
        prm = read_params(str(case / "dolfin_params.dat"))
        assert (prm["wall_upgrade"], prm["wall_facets"], prm["wall_markers"]) == \
            ("true", "boundaries.xdmf", "1")
        points, cells, markers = read(str(case / "boundaries.xdmf"))
        assert len(cells) == expected and (markers == 1).all()
        facets[nprocs] = sorted(tuple(sorted(tuple(np.round(points[i], 10)) for i in cell))
                                for cell in cells)
    assert facets[1] == facets[3]
    print("wall facets:", expected)
