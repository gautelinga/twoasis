"""Building blocks shared by the TPfracStep problems."""
__author__ = "Gaute Linga <gaute.linga@ntnu.no>"
__date__ = "2026-11-02"
__copyright__ = "Copyright (C) 2026 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from os import makedirs, path
import pickle

import numpy as np
from dolfin import (
    assemble, DOLFIN_EPS_LARGE, grad, HDF5File, inner, Mesh, MPI, Point, RectangleMesh,
    SpatialCoordinate, SubDomain)

from twoasis.common.utilities import compute_ind


def rectangle_mesh(Lx, Ly, res, centred=False):
    """Structured triangle mesh of [0, Lx] x [0, Ly], or of [-Lx/2, Lx/2] x
    [-Ly/2, Ly/2] if centred, with cells of size about res."""
    x0, y0 = (-Lx / 2, -Ly / 2) if centred else (0., 0.)
    return RectangleMesh(Point(x0, y0), Point(x0 + Lx, y0 + Ly),
                         int(round(Lx / res)), int(round(Ly / res)))


def centred(L):
    """Bounds (-L/2, L/2) of an interval of length L centred at 0."""
    return (-L / 2, L / 2)


def load_mesh(fname):
    """Read the mesh stored as "mesh" in the HDF5 file fname."""
    mesh = Mesh()
    with HDF5File(mesh.mpi_comm(), fname, "r") as h5f:
        h5f.read(mesh, "mesh", False)
    return mesh


class PeriodicBox(SubDomain):
    """Periodic boundary conditions on a box.

    `periodic` maps each periodic axis to its (lower, upper) bounds, e.g.
    {0: (0, Lx)} for periodicity in x. The face at the upper bound is identified
    with the face at the lower bound.
    """

    def __init__(self, periodic):
        self.periodic = dict(periodic)
        super().__init__()

    def inside(self, x, on_boundary):
        # On a lower face, but not on an upper face (those are mapped).
        return bool(on_boundary
                    and any(abs(x[i] - lo) < DOLFIN_EPS_LARGE for i, (lo, hi) in self.periodic.items())
                    and not any(abs(x[i] - hi) < DOLFIN_EPS_LARGE for i, (lo, hi) in self.periodic.items()))

    def map(self, x, y):
        for i in range(len(x)):
            y[i] = x[i]
        for i, (lo, hi) in self.periodic.items():
            if abs(x[i] - hi) < DOLFIN_EPS_LARGE:
                y[i] = x[i] - (hi - lo)


class Faces(SubDomain):
    """The boundary where coordinate `axis` equals one of `values`."""

    def __init__(self, axis, *values):
        self.axis, self.values = axis, values
        super().__init__()

    def inside(self, x, on_boundary):
        return on_boundary and any(abs(x[self.axis] - v) < DOLFIN_EPS_LARGE for v in self.values)


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


def load_restart_parameters(NS_parameters, commandline_kwargs):
    """If the command line gives a restart_folder, load the parameters stored
    there into NS_parameters and return True; otherwise return False."""
    if "restart_folder" not in commandline_kwargs:
        return False
    restart_folder = path.abspath(commandline_kwargs["restart_folder"])
    with open(path.join(restart_folder, "params.dat"), "rb") as f:
        NS_parameters.update(pickle.load(f))
    NS_parameters["restart_folder"] = restart_folder
    return True


def write_timestamps_info(timestampsfolder, mesh, params, velocity_degree, pressure_degree,
                          constrained_domain):
    """Describe the files that write_timestamp adds to timestampsfolder:
    params.dat lists the given parameters (a dict), dolfin_params.dat the
    function spaces and the periodic directions, and mesh.h5 holds the mesh."""
    periodic = getattr(constrained_domain, "periodic", {})
    if MPI.rank(MPI.comm_world) == 0:
        with open(path.join(timestampsfolder, "params.dat"), "a+") as ofile:
            for key, val in params.items():
                ofile.write("{}={}\n".format(key, val))
        with open(path.join(timestampsfolder, "dolfin_params.dat"), "a+") as ofile:
            ofile.write("velocity_space=P{}\n".format(velocity_degree))
            ofile.write("pressure_space=P{}\n".format(pressure_degree))
            ofile.write("phase_field_space=P{}\n".format(velocity_degree))
            ofile.write("timestamps=timestamps.dat\n")
            ofile.write("mesh=mesh.h5\n")
            for i, axis in enumerate("xyz"):
                ofile.write("periodic_{}={}\n".format(axis, str(i in periodic).lower()))
            ofile.write("rho=1.0\n")
    with HDF5File(mesh.mpi_comm(), path.join(timestampsfolder, "mesh.h5"), "w") as h5f:
        h5f.write(mesh, "mesh")


def read_phase_field(fname, mesh, q_):
    """The phase field stored as "phi" in the HDF5 file fname (e.g. a file
    written by write_timestamp)."""
    phi__, _ = q_['phig'].split(deepcopy=True)
    with HDF5File(mesh.mpi_comm(), fname, "r") as h5f:
        h5f.read(phi__, "phi")
    return phi__


def write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder):
    """Write velocity, pressure and phase field to up_<tstep>.h5 and list the file
    in timestamps.dat."""
    uv()
    h5fname = "up_{}.h5".format(tstep)
    phi__, _ = q_['phig'].split(deepcopy=True)
    phi__.rename("phi", "phi")
    with HDF5File(mesh.mpi_comm(), path.join(timestampsfolder, h5fname), "w") as h5f:
        h5f.write(uv, "u")
        h5f.write(p_, "p")
        h5f.write(phi__, "phi")
    if MPI.rank(MPI.comm_world) == 0:
        with open(path.join(timestampsfolder, "timestamps.dat"), "a+") as ofile:
            ofile.write("{:.6f} {}\n".format(t, h5fname))
