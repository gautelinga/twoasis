"""A plug of one fluid spanning a straight channel, periodic along x, driven by a body force
F0 along it: Tube3D's case in 2D, for how the discretisation moves the plug (walls at
y = +-width/2 are straight lines, so no facets). mesh_kind "unstructured" reads a gmsh mesh
(meshes/channel_mesh.py), "structured" builds a rectangle mesh that repeats exactly every cell
along x. The plug's caps are circular arcs of radius Rb, their apexes H from its centre (Rb = 1
with width 1: the static menisci of a 60 degree contact angle; a large Rb: flat ones)."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2026-11-25"
__copyright__ = ("Copyright (C) 2026 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"

import numpy as np
from twoasis.backend import (
    assemble, cos, DirichletBC, dx, Expression, extent, hmax, sin, SpatialCoordinate, SubDomain)

from twoasis.common.utilities import compute_ind, set_subfunction
from twoasis.problems import info_red
from twoasis.problems.TPfracStep.shared import (
    load_mesh, PeriodicBox, rectangle_mesh, standard_stats, start_timestamps, wall_energy,
    write_tdata)


class Walls(SubDomain):
    """The walls: boundary points at |y| >= width/2 - tol."""
    def __init__(self, width, tol):
        self.width, self.tol = width, tol
        super().__init__()

    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1]) > self.width / 2 - self.tol


def meshfile_name(L, width, res):
    return "meshes/channel_L{}_W{}_res{}.h5".format(L, width, res)


def mesh(L, width, res, mesh_kind, **params):
    if mesh_kind == "structured":
        return rectangle_mesh(L, width, res, centred=True)
    if mesh_kind == "unstructured":
        return load_mesh(meshfile_name(L, width, res), periodic=(True, False))
    raise ValueError("mesh_kind {!r}: 'structured' or 'unstructured'".format(mesh_kind))


def problem_parameters(NS_parameters, **NS_namespace):
    NS_parameters.update(
        T=1000.0,
        L=4.0, width=1.0, res=0.04, mesh_kind="unstructured",
        H=1.25, Rb=1.0,              # the plug: cap apexes H from its centre, caps of radius Rb
        plug_shift=0.0,              # the plug's centre: this far from the channel's middle
        dt=0.5,
        rho=[2595, 1898], mu=[1.0, 0.92], sigma=1.0, epsilon=0.08, M=1.6e-3,
        theta=np.pi / 3,
        F0=[0., 0.], g0=[0., 0.],
        velocity_degree=2, bdf_order=2, solver="BDF",
        velocity_pressure_solve="coupled", chns_solve="coupled",
        folder="channel2d_results", telemetry=True,
        plot_interval=1000000, stat_interval=1, timestamps_interval=1000000,
        save_step=1000000, checkpoint=1000000, print_intermediate_info=1000000,
    )


def constrained_domain(mesh, **params):
    xmin, xmax = (x[0] for x in extent(mesh))
    return PeriodicBox({0: (xmin, xmax)})


def mark_subdomains(subdomains, mesh, width, **NS_namespace):
    """Mark the walls 1 and add the channel's ends x0, x1 to the namespace."""
    lo, hi = extent(mesh)
    Walls(width, hmax(mesh)**2 / width + 1e-10).mark(subdomains, 1)
    return dict(x0=lo[0], x1=hi[0])


def create_bcs(V, subdomains, **NS_namespace):
    bc_u_wall = DirichletBC(V, 0, subdomains, 1)
    return dict(u0=[bc_u_wall], u1=[bc_u_wall], p=[], phig=[])


def initialize(q_, q_1, q_2, x_1, x_2, bcs, epsilon, VV, x0, x1, H, Rb, plug_shift,
               **NS_namespace):
    # The plug (phi = -1) in the middle: inside the stadium whose caps are arcs of radius Rb
    # with their apexes at x = xc +- H.
    set_subfunction(q_['phig'], 0, Expression(
        "tanh((sqrt(pow(x[1], 2) + pow(std::max(0.0, std::abs(x[0]-xc) - (H-Rb)), 2)) - Rb)"
        "/(sqrt(2)*epsilon))", epsilon=epsilon, Rb=Rb, H=H, xc=(x0 + x1) / 2 + plug_shift, degree=2))
    q_1['phig'].vector()[:] = q_['phig'].vector()
    q_2['phig'].vector()[:] = q_['phig'].vector()
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def plug_position(q_, mesh, ds, x0, x1):
    """Where the plug (phi = -1) is: x_c, its centroid, and x_w, its centroid on the walls,
    each the circular mean of x over the period weighted by its indicator; and L_w, its wetted
    length on each wall (its wall length over the two walls)."""
    ind_ = compute_ind(q_, -1)
    k = 2 * np.pi / (x1 - x0)
    x = SpatialCoordinate(mesh)[0]

    def circular_mean(measure):
        c, s = (assemble(ind_ * f(k * (x - x0)) * measure) for f in (cos, sin))
        return x0 + (np.arctan2(s, c) % (2 * np.pi)) / k
    return [circular_mean(dx(domain=mesh)), circular_mean(ds(1)), assemble(ind_ * ds(1)) / 2]


def pre_solve_hook(**NS_namespace):
    return start_timestamps(
        ["F0", "g0", "mu", "rho", "sigma", "M", "theta", "epsilon", "dt", "res", "mesh_kind",
         "H", "Rb", "plug_shift"], **NS_namespace)


def temporal_hook(q_, tstep, t, u_, phi_, rho_, sigma, epsilon, volume, statsfolder,
                  stat_interval, mesh, ds, x0, x1, theta, **NS_namespace):
    """Write to Stats/tdata.dat: tstep, t, u0m, u1m, phim, the energies per area (kinetic,
    interface gradient and double-well parts), the plug's position (x_c, x_w, L_w) and the
    walls' energy per area (E_wall)."""
    info_red("tstep = {}".format(tstep))
    if tstep % stat_interval == 0:
        write_tdata(statsfolder, tstep, t,
                    standard_stats(q_, u_, phi_, rho_, sigma, epsilon, volume, mesh)
                    + plug_position(q_, mesh, ds, x0, x1)
                    + [wall_energy(phi_, sigma, theta, ds(1)) / volume])

