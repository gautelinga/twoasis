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

from conftest import (MASS_TOL, QUIET, REPO_ROOT, assert_mass_conserved, load_tdata, require_mpi,
                      run_twoasis)

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


def simplepore2d(meshes):
    """SimplePore2D's lattice, made by meshes/lattice_mesh.py."""
    pytest.importorskip("gmsh")
    pytest.importorskip("meshio")
    subprocess.run([sys.executable, str(REPO_ROOT / "meshes" / "lattice_mesh.py"), "-res", "0.1"],
                   cwd=str(meshes), check=True, capture_output=True, timeout=300)
    return dict(res=0.1)


# The problems the dual-mesh smoke test adds to PROBLEMS and MESH_PROBLEMS, with what
# makes them small.
DUAL_EXTRA = dict(PeriodicBubble2D=lambda m: {}, TaylorBubble2D=lambda m: {},
                  TaylorGreen2D=lambda m: {}, SimplePore2D=simplepore2d,
                  Bubble3D=lambda m: dict(res=0.25))


@pytest.mark.slow
@pytest.mark.parametrize("problem", PROBLEMS + sorted(MESH_PROBLEMS) + sorted(DUAL_EXTRA))
def test_problem_runs_on_dual_mesh(tmp_path, problem):
    """Every problem with its phase field on the flow mesh refined once
    (phase_refine=1, twoasis/common/dual.py): its markers, contact angles, Dirichlet
    conditions, statistics and timestamps on the right meshes."""
    params = with_mesh(tmp_path, problem)
    if problem in DUAL_EXTRA:
        params = DUAL_EXTRA[problem](tmp_path / "meshes")
    run_twoasis(tmp_path, problem=problem, phase_refine=1, dt=1e-4, T=2e-4, **QUIET,
                **params)


@pytest.mark.slow
@pytest.mark.parametrize("problem,nprocs", [("Porous2D", 2), ("Porous3D", 3)])
def test_mesh_problem_restart_on_other_process_count(tmp_path, problem, nprocs):
    """A checkpoint written in serial, restarted on nprocs processes (the mesh
    read and its periodic partition repaired, Porous3D: partitioned by
    feluga), continues as the uninterrupted serial run does. The checkpoint is
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
    # atol: one unit of the 8th decimal the statistics are printed with, which the
    # order of a parallel sum can flip
    np.testing.assert_allclose(td, td_ref, rtol=1e-6, atol=1.1e-8)


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
    from feluga.boundaries import read
    from partractools.case import read_params
    from twoasis.problems.TPfracStep.CylArr2D import get_fname
    # The expected walls from dolfin, whichever backend the runs are on
    from feluga.backend.dolfin import mark_walls, read_mesh_file as load_mesh
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


def test_porous2d_wetting_direction(tmp_path):
    """The contact angle drives the wetting the right way (twoasisx's Porous2D gate):
    from the same start, theta = pi/4 (phase phi = +1 wets the obstacles) raises the
    fraction of the walls that phase covers and theta = 3*pi/4 lowers it, and the mass
    of each phase is kept."""
    params = dict(QUIET, problem="Porous2D", T=0.25, **with_mesh(tmp_path, "Porous2D"))
    change = {}
    for theta in (np.pi / 4, 3 * np.pi / 4):
        wd = tmp_path / "theta{:.3f}".format(theta)
        wd.mkdir()
        (wd / "meshes").symlink_to(tmp_path / "meshes")
        run_twoasis(wd, theta=theta, **params)
        wet = np.loadtxt(wd / "results" / "data" / "1" / "Stats" / "wetting.dat", ndmin=2)
        change[theta] = wet[-1, 2] - wet[0, 2]
        assert_mass_conserved(load_tdata(wd), MASS_TOL)
    assert change[np.pi / 4] > 0.05 and change[3 * np.pi / 4] < -0.05, change


def test_tube3d_plug_position_and_wall_energy(tmp_path):
    """Tube3D's statistics on a known phase field: a bubble across the periodic ends keeps
    one position (z_c and z_w at the ends, not mid-tube), its wetted length is the band's;
    no bubble gives nan positions and no wetted length; and the walls' energy of a uniform
    phi is -sigma*cos(theta)*(3*phi - phi^3)/4 times the wall's area, with theta as an angle
    or as a Function of cos(theta)."""
    from twoasis.backend import (assemble, BoundaryMarkers, Constant, constant_on, Expression,
                                 FiniteElement, Function, FunctionSpace, Measure, MixedElement)
    from twoasis.common.utilities import set_subfunction, subfunction
    from twoasis.problems.TPfracStep import Tube3D
    from twoasis.problems.TPfracStep.shared import wall_energy
    meshes = tmp_path / "meshes"
    meshes.mkdir()
    subprocess.run([sys.executable, str(REPO_ROOT / "meshes" / "tube_mesh.py"),
                    "-L", "2", "-R", "0.5", "-res", "0.1"],
                   cwd=str(meshes), check=True, capture_output=True, timeout=300)
    mesh = Tube3D.mesh(str(meshes / "tube.h5"))
    box = Tube3D.constrained_domain(mesh)
    markers = BoundaryMarkers(mesh)
    markers.set_all(0)
    ns = Tube3D.mark_subdomains(markers, mesh)
    z0, z1 = ns["z0"], ns["z1"]
    ds = Measure("ds", domain=mesh, subdomain_data=markers)
    el = FiniteElement("CG", mesh.ufl_cell(), 1)
    q_ = {"phig": Function(FunctionSpace(mesh, MixedElement([el, el]), constrained_domain=box))}
    k = 2 * np.pi / (z1 - z0)

    # a bubble filling the tube within 0.4 of the periodic ends
    set_subfunction(q_["phig"], 0, Expression("cos(k*(x[2] - z0)) > cos(k*w) ? -1.0 : 1.0",
                                              k=k, z0=z0, w=0.4, degree=1))
    z_c, z_w, L_w = Tube3D.plug_position(q_, mesh, ds, z0, z1)
    for z in (z_c, z_w):
        assert min(z - z0, z1 - z) < 0.02, (z, z0, z1)
    assert L_w == pytest.approx(0.8, abs=0.1)

    set_subfunction(q_["phig"], 0, Constant(1.))
    z_c, z_w, L_w = Tube3D.plug_position(q_, mesh, ds, z0, z1)
    assert np.isnan(z_c) and np.isnan(z_w) and L_w == 0

    phi = 0.5
    set_subfunction(q_["phig"], 0, Constant(phi))
    phi_ = subfunction(q_["phig"], 0)
    area = assemble(constant_on(1., mesh) * ds(1))
    expect = -2.0 * np.cos(1.0) * (3 * phi - phi**3) / 4 * area
    assert wall_energy(phi_, 2.0, [(1.0, 1)], ds) == pytest.approx(expect, rel=1e-10)
    costheta = Function(FunctionSpace(mesh, "CG", 1, constrained_domain=box))
    costheta.vector()[:] = np.cos(1.0)
    assert wall_energy(phi_, 2.0, [(costheta, 1)], ds) == pytest.approx(expect, rel=1e-10)
    assert wall_energy(phi_, 2.0, [], ds) == 0


@pytest.mark.parametrize("mesh_kind,theta", [("structured", None), ("structured", 1.0),
                                             ("unstructured", 1.0)])
def test_channel2d(tmp_path, mesh_kind, theta):
    """Channel2D runs on its structured mesh and on meshes/channel_mesh.py's, with and
    without a contact angle: its statistics end with the plug's position and wetted length,
    the flow in its frame, and the walls' energy only when there is a contact angle."""
    params = dict(mesh_kind=mesh_kind, res=0.1, L=2, width=1, theta=theta)
    if mesh_kind == "unstructured":
        pytest.importorskip("gmsh")
        pytest.importorskip("meshtools")
        meshes = tmp_path / "meshes"
        meshes.mkdir()
        subprocess.run([sys.executable, str(REPO_ROOT / "meshes" / "channel_mesh.py"),
                        "-L", "2", "-W", "1", "-res", "0.1"],
                       cwd=str(meshes), check=True, capture_output=True, timeout=300)
    run_twoasis(tmp_path, problem="Channel2D", dt=0.1, T=0.2, **dict(QUIET, **params))
    td = load_tdata(tmp_path)
    assert td.shape[1] == 2 + 6 + 3 + 4 + (0 if theta is None else 1), td.shape
    assert np.isfinite(td).all()


def test_comoving_stats():
    """comoving_stats: a uniform flow U0 is U_p = U0 with no flow in the plug's frame; for a
    nonuniform flow U_p, u_band and u_co equal the integrals assembled directly and u_max the
    largest nodal size."""
    from twoasis.backend import (assemble, constant_on, dx, Expression, FiniteElement, Function,
                                 FunctionSpace, interpolate, MixedElement)
    from twoasis.common.utilities import compute_ind, set_subfunction, subfunction
    from twoasis.problems.TPfracStep.shared import comoving_stats, rectangle_mesh
    mesh = rectangle_mesh(2, 1, 0.1, centred=True)
    el = FiniteElement("CG", mesh.ufl_cell(), 1)
    q_ = {"phig": Function(FunctionSpace(mesh, MixedElement([el, el])))}
    set_subfunction(q_["phig"], 0, Expression("tanh((std::abs(x[0]) - 0.5)/0.1)", degree=2))
    phi_ = subfunction(q_["phig"], 0)
    V = FunctionSpace(mesh, "CG", 2)
    dX = dx(domain=mesh)
    volume = assemble(constant_on(1., mesh) * dX)

    u_ = [interpolate(Expression("0.3", degree=0), V), interpolate(Expression("0.0", degree=0), V)]
    q_.update(u0=u_[0], u1=u_[1])
    U_p, u_band, u_co, u_max = comoving_stats(q_, u_, phi_, volume, mesh)
    assert U_p == pytest.approx(0.3, rel=1e-12)
    assert u_band < 1e-7 and u_co < 1e-7 and u_max < 1e-7

    u_ = [interpolate(Expression("0.3 + x[1] + 0.2*x[0]*x[0]", degree=2), V),
          interpolate(Expression("x[0]*x[1]", degree=2), V)]
    q_.update(u0=u_[0], u1=u_[1])
    U_p, u_band, u_co, u_max = comoving_stats(q_, u_, phi_, volume, mesh)
    ind_ = compute_ind(q_, -1)
    assert U_p == pytest.approx(assemble(ind_ * u_[0] * dX) / assemble(ind_ * dX), rel=1e-12)
    U = constant_on(U_p, mesh)
    d2 = (u_[0] - U)**2 + u_[1]**2
    band = 1 - phi_**2
    assert u_band == pytest.approx(np.sqrt(assemble(band * d2 * dX) / assemble(band * dX)),
                                   rel=1e-10)
    assert u_co == pytest.approx(np.sqrt(assemble(d2 * dX) / volume), rel=1e-10)
    nodal = np.sqrt((u_[0].vector().get_local() - U_p)**2 + u_[1].vector().get_local()**2)
    assert u_max == pytest.approx(nodal.max(), rel=1e-12)
