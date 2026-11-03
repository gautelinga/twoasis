"""Building blocks shared by the TPfracStep problems."""
__author__ = "Gaute Linga <gaute.linga@ntnu.no>"
__date__ = "2026-11-02"
__copyright__ = "Copyright (C) 2026 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from os import makedirs, path, remove

import numpy as np
from dolfin import (
    assemble, CompiledSubDomain, Constant, DomainBoundary, dx, Function, grad, HDF5File,
    inner, Mesh, MPI, Point, RectangleMesh, SpatialCoordinate, SubDomain)

from femtools.periodic import periodic_box

from twoasis.common.utilities import compute_ind, subfunction, subspace

# How near a face of the box a point must be to lie on it: above the rounding of
# mesh generators (gmsh leaves points ~1e-14 off a face), far below any cell.
FACE_TOL = 1e-10


def rectangle_mesh(Lx, Ly, res, centred=False):
    """Structured triangle mesh of [0, Lx] x [0, Ly], or of [-Lx/2, Lx/2] x
    [-Ly/2, Ly/2] if centred, with cells of size about res."""
    x0, y0 = (-Lx / 2, -Ly / 2) if centred else (0., 0.)
    return RectangleMesh(Point(x0, y0), Point(x0 + Lx, y0 + Ly),
                         int(round(Lx / res)), int(round(Ly / res)))


def centred(L):
    """Bounds (-L/2, L/2) of an interval of length L centred at 0."""
    return (-L / 2, L / 2)


def load_mesh(fname, periodic=None):
    """Read the mesh stored as "mesh" in the HDF5 file fname.

    `periodic` gives the axes (x, y, z) the problem identifies. Under MPI the mesh
    is then read in blocks and partitioned with femtools.periodic_partition, so that
    no process holds a vertex of a periodic face without knowing that it is periodic
    (dolfin's periodic dofmap would give that vertex a wrong dof). Without it, the
    mesh is read by dolfin and post_import_problem repairs the partition, which
    costs another build of the mesh when cells have to move.
    """
    if periodic is not None and MPI.size(MPI.comm_world) > 1:
        from femtools.periodic_partition import read_mesh
        # L=None: the upper faces are at the largest coordinates, which is all the
        # partition needs.
        return read_mesh(fname, None, periodic, MPI.comm_world, log=_info)
    mesh = Mesh()
    with HDF5File(mesh.mpi_comm(), fname, "r") as h5f:
        h5f.read(mesh, "mesh", False)
    return mesh


def _info(s):
    if MPI.rank(MPI.comm_world) == 0:
        print(s)


def PeriodicBox(periodic):
    """Periodic boundary conditions on a box (femtools.periodic.periodic_box).

    `periodic` maps each periodic axis to its (lower, upper) bounds, e.g.
    {0: (0, Lx)} for periodicity in x. The face at the upper bound is identified
    with the face at the lower bound. The box carries `lower`, `upper` and
    `periodic` (one entry per axis x, y, z).
    """
    lower, upper = ([periodic[i][k] if i in periodic else 0. for i in range(3)] for k in (0, 1))
    # map_tol: dolfin's default for a SubDomain.
    return periodic_box(upper, FACE_TOL, [i in periodic for i in range(3)], lower,
                        map_tol=1e-10)


class Faces(SubDomain):
    """The boundary where coordinate `axis` equals one of `values`."""

    def __init__(self, axis, *values):
        self.axis, self.values = axis, values
        super().__init__()

    def inside(self, x, on_boundary):
        return on_boundary and any(abs(x[self.axis] - v) < FACE_TOL for v in self.values)


def mark_walls(subdomains, mesh, marker=1):
    """Mark the walls of the obstacles inside the mesh's bounding box `marker`:
    every boundary facet, except those on the faces of the box (left 0; a
    problem marks its inlets and outlets there afterwards). Compiled, so fast on
    large meshes, and it needs no list of the obstacles."""
    DomainBoundary().mark(subdomains, marker)
    lo, hi = mesh_extrema(mesh)
    faces = " || ".join("near(x[{0}], lo{0}, tol) || near(x[{0}], hi{0}, tol)".format(i)
                        for i in range(mesh.geometry().dim()))
    bounds = {"lo{}".format(i): v for i, v in enumerate(lo)}
    bounds.update({"hi{}".format(i): v for i, v in enumerate(hi)})
    CompiledSubDomain("on_boundary && ({})".format(faces), tol=FACE_TOL,
                      **bounds).mark(subdomains, 0)


def mesh_extrema(mesh):
    """Lower and upper corners of the mesh's bounding box, over all processes."""
    x = mesh.coordinates()
    comm = mesh.mpi_comm()
    lo = np.array([MPI.min(comm, float(v)) for v in x.min(axis=0)])
    hi = np.array([MPI.max(comm, float(v)) for v in x.max(axis=0)])
    return lo, hi


def interface_energy(phi_, sigma, epsilon, dx):
    """The gradient and double-well parts of the interface free energy,
    integrated over the domain:

        sigma_bar*epsilon/2 * |grad(phi)|^2  and  sigma_bar/(4*epsilon) * (1 - phi^2)^2

    with sigma_bar = 3*sigma/(2*sqrt(2)), the scaling the solvers use, so that a
    flat interface at equilibrium has the energy sigma per unit area (half in
    each part). The wall energy from contact angles is not included."""
    sigma_bar = 3 * sigma / (2 * np.sqrt(2))
    E_grad = 0.5 * sigma_bar * epsilon * assemble(inner(grad(phi_), grad(phi_)) * dx)
    E_dw = 0.25 * sigma_bar / epsilon * assemble((1 - phi_**2)**2 * dx)
    return E_grad, E_dw


def phase_moments(q_, mesh, dx, phase=-1):
    """Volume of the phase phi = phase (+1 or -1), measured with a smoothed
    indicator, and its centroid and mean velocity (one entry per dimension).
    A centroid along a periodic direction is meaningless once the phase
    crosses the periodic boundary."""
    ind_ = compute_ind(q_, phase)
    X = SpatialCoordinate(mesh)
    dim = mesh.geometry().dim()
    vol = assemble(ind_ * dx)
    centroid = [assemble(ind_ * X[i] * dx) / vol for i in range(dim)]
    velocity = [assemble(ind_ * q_["u{}".format(i)] * dx) / vol for i in range(dim)]
    return vol, centroid, velocity


def makedirs_on_root(*folders):
    """Create folders (on rank 0 only) if they do not exist."""
    if MPI.rank(MPI.comm_world) == 0:
        for folder in folders:
            if not path.exists(folder):
                makedirs(folder)


def setup_stats(mesh, newfolder):
    """Create the folder newfolder/Stats (on rank 0) and measure the volume of
    the mesh. Returns dict(statsfolder=..., volume=...), to be returned (or
    merged into the dict returned) by a pre_solve_hook."""
    statsfolder = path.join(newfolder, "Stats")
    makedirs_on_root(statsfolder)
    volume = assemble(Constant(1.) * dx(domain=mesh))
    return dict(statsfolder=statsfolder, volume=volume)


def standard_stats(q_, u_, phi_, rho_, sigma, epsilon, volume, mesh):
    """The mean velocity components (u0m, u1m and, in 3D, u2m), the mean phase
    field phim and the energies per volume: kinetic E_kin, interface gradient
    E_int and double-well E_pot parts (see interface_energy). Returns them as a
    list in this order. All processes must call it (it assembles)."""
    dX = dx(domain=mesh)
    dim = mesh.geometry().dim()
    um = [assemble(q_['u{}'.format(i)] * dX) / volume for i in range(dim)]
    phim = assemble(phi_ * dX) / volume
    E_kin = 0.5 * assemble(rho_ * sum(u_[i]**2 for i in range(dim)) * dX) / volume
    E_int, E_pot = (E / volume for E in interface_energy(phi_, sigma, epsilon, dX))
    return um + [phim, E_kin, E_int, E_pot]


def gravitational_energy(rho_, g0, volume, mesh):
    """The gravitational energy per volume, -rho g0.x, with g0 = None meaning no
    gravity. All processes must call it (it assembles)."""
    if g0 is None:
        return 0.
    X = SpatialCoordinate(mesh)
    dim = mesh.geometry().dim()
    return assemble(-rho_ * sum(g0[i] * X[i] for i in range(dim)) * dx(domain=mesh)) / volume


# The statistics files appended to by this run.
_APPENDED = set()


def append_row(fname, tstep, line):
    """Append line, the row of time step tstep, to the statistics file fname (call
    on one process). A restart continues the file of the run it restarts, which
    may hold rows from tstep on, written after the checkpoint: the first row this
    run appends drops those, so that the file is one run's history."""
    if fname not in _APPENDED and path.exists(fname):
        with open(fname) as f:
            rows = f.readlines()
        keep = [r for r in rows if not r.split() or int(r.split()[0]) < tstep]
        if len(keep) < len(rows):
            with open(fname, "w") as f:
                f.writelines(keep)
    _APPENDED.add(fname)
    with open(fname, "a") as f:
        f.write(line)


def write_tdata(statsfolder, tstep, t, entries):
    """Append the line "tstep t entries..." to statsfolder/tdata.dat (on rank 0),
    with 8 decimals for t and each entry (append_row)."""
    if MPI.rank(MPI.comm_world) == 0:
        append_row(path.join(statsfolder, "tdata.dat"), tstep,
                   ("{:d} {:.8f} " + " ".join("{:.8f}" for _ in entries) + "\n").format(
                       tstep, t, *entries))


def write_timestamps_info(timestampsfolder, mesh, params, velocity_degree, pressure_degree,
                          phase_field_degree, constrained_domain):
    """Describe the files that write_timestamp adds to timestampsfolder:
    params.dat lists the given parameters (a dict), dolfin_params.dat the
    function spaces and the periodic directions, and mesh.h5 holds the mesh."""
    from partractools import case
    periodic = getattr(constrained_domain, "periodic", [False] * 3)
    if MPI.rank(MPI.comm_world) == 0:
        with open(path.join(timestampsfolder, "params.dat"), "w") as ofile:
            for key, val in params.items():
                ofile.write("{}={}\n".format(key, val))
        case.write_params(path.join(timestampsfolder, "dolfin_params.dat"), dict(
            velocity_space="P{}".format(velocity_degree),
            pressure_space="P{}".format(pressure_degree),
            phase_field_space="P{}".format(phase_field_degree),
            timestamps="timestamps.dat", mesh="mesh.h5",
            **{"periodic_" + axis: periodic[i] for i, axis in enumerate("xyz")},
            rho=1.0))
    case.write_mesh(path.join(timestampsfolder, "mesh.h5"), mesh)


def read_phase_field(fname, mesh, q_):
    """The phase field stored as "phi" in the HDF5 file fname (e.g. a file
    written by write_timestamp)."""
    phi__ = Function(subspace(q_['phig'].function_space(), 0))
    with HDF5File(mesh.mpi_comm(), fname, "r") as h5f:
        h5f.read(phi__, "phi")
    return phi__


def write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder):
    """Write velocity, pressure and phase field to up_<tstep>.h5 and list the file
    in timestamps.dat, as its last step.

    A restart writes the step it starts from again, and an earlier run may have
    listed later steps (a restart from an older checkpoint). Those steps belong
    to the other history: they are dropped from the list and their files
    deleted, so that the list is always one run's history."""
    uv()
    h5fname = "up_{}.h5".format(tstep)
    phi__ = subfunction(q_['phig'], 0)
    phi__.rename("phi", "phi")
    from partractools import case
    case.write_fields(path.join(timestampsfolder, h5fname), {"u": uv, "p": p_, "phi": phi__})
    if MPI.rank(MPI.comm_world) == 0:
        stamps = path.join(timestampsfolder, "timestamps.dat")
        listed = case.read_stamps(stamps) if path.exists(stamps) else []
        keep = [(nums, f) for nums, f in listed if _stamp_step(f) < tstep]
        if len(keep) == len(listed):
            case.write_stamps(stamps, [("{:.6f}".format(t), h5fname)], mode="a")
        else:
            for _, f in listed:
                if _stamp_step(f) > tstep and path.exists(path.join(timestampsfolder, f)):
                    remove(path.join(timestampsfolder, f))
            case.write_stamps(stamps, keep + [("{:.6f}".format(t), h5fname)], mode="w")


def _stamp_step(fname):
    """The time step of a file up_<tstep>.h5 that write_timestamp wrote."""
    return int(path.splitext(fname)[0].split("_")[-1])
