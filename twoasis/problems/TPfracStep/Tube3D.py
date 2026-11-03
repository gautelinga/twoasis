"""An elongated bubble in a periodic circular tube (a Taylor bubble in 3D), driven
along the tube by a mean pressure gradient (mesh from meshes/tube_mesh.py; the tube's
length and radius are read from the mesh)."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2022-11-22"
__copyright__ = ("Copyright (C) 2022 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"

from os import path

import numpy as np
from dolfin import DirichletBC, Expression, SubDomain

from twoasis.common.utilities import set_subfunction
from twoasis.problems import info_red
from twoasis.problems.TPfracStep.shared import (
    load_mesh, makedirs_on_root, mesh_extrema, PeriodicBox, setup_stats, standard_stats,
    write_tdata, write_timestamp, write_timestamps_info)


class Walls(SubDomain):
    """The tube wall: boundary points at distance >= R - tol from the axis. tol
    must exceed the sag of a facet below the circle, ~h^2/(8R)."""
    def __init__(self, R, tol):
        self.R, self.tol = R, tol
        super().__init__()

    def inside(self, x, on_boundary):
        return on_boundary and np.sqrt(x[0]**2 + x[1]**2) > self.R - self.tol


def mesh(meshfile, **params):
    return load_mesh(meshfile, periodic=(False, False, True))


def problem_parameters(NS_parameters, **NS_namespace):
    NS_parameters.update(
        T=100.0,
        meshfile="meshes/tube.h5",  # a tube along z, centred on the z axis
        Lb=3,                       # bubble length (cylinder with hemispherical caps)
        Rb=0.35,                    # bubble radius
        dt=0.01,
        rho=[1, 0.1],
        mu=[10., 1.0],
        theta=np.pi/2,
        epsilon=0.03,
        sigma=5.0,
        M=0.0001,
        F0=[0., 0., -10.],
        g0=[0., 0., 0.],
        velocity_degree=1,
        folder="tube3d_results",
        plot_interval=10,
        stat_interval=10,
        timestamps_interval=100,
        save_step=10,
        checkpoint=10,
        print_intermediate_info=10,
        use_krylov_solvers=True,
        solver="BDF",
        bdf_order=1,
        AB_projection_pressure=False,
        max_iter=20,                 # Number of inner pressure velocity iterations on timestep
        max_error=1e-3,               # Tolerance for inner iterations (pressure velocity iterations)
        iters_on_first_timestep=100,  # Number of iterations on first timestep
    )


def constrained_domain(mesh, **params):
    zmin, zmax = (x[2] for x in mesh_extrema(mesh))
    return PeriodicBox({2: (zmin, zmax)})


def mark_subdomains(subdomains, mesh, **NS_namespace):
    """Mark the wall 1 and add the tube's ends z0, z1 and radius R to the namespace."""
    lo, hi = mesh_extrema(mesh)
    R = hi[0]
    Walls(R, mesh.hmax()**2 / R).mark(subdomains, 1)
    return dict(z0=lo[2], z1=hi[2], R=R)


def create_bcs(V, subdomains, **NS_namespace):
    bc_u_wall = DirichletBC(V, 0, subdomains, 1)
    return dict(u0=[bc_u_wall],
                u1=[bc_u_wall],
                u2=[bc_u_wall],
                p=[],
                phig=[])


def initialize(q_, q_1, q_2, x_1, x_2, bcs, epsilon, VV, z0, z1, Lb, Rb, **NS_namespace):
    # The bubble (phi = -1) in the middle of the tube.
    set_subfunction(q_['phig'], 0, Expression(
        "tanh((sqrt(pow(x[0], 2)+pow(x[1], 2)+pow(std::max(0.0, std::abs(x[2]-zc)-Lb/2), 2))-Rb)/(sqrt(2)*epsilon))",
        epsilon=epsilon, Rb=Rb, zc=(z0 + z1) / 2, Lb=Lb, degree=2))
    q_1['phig'].vector()[:] = q_['phig'].vector()
    q_2['phig'].vector()[:] = q_['phig'].vector()
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(tstep, t, q_, p_, mesh, u_, newfolder, velocity_degree, pressure_degree,
                   phase_field_degree, AssignedVectorFunction, constrained_domain, **NS_namespace):
    stats = setup_stats(mesh, newfolder)
    timestampsfolder = path.join(newfolder, "Timestamps")
    uv = AssignedVectorFunction(u_, name="u")
    makedirs_on_root(timestampsfolder)
    write_timestamps_info(timestampsfolder, mesh,
                          {key: NS_namespace[key] for key in
                           ["F0", "g0", "mu", "rho", "sigma", "M", "theta", "epsilon", "dt"]},
                          velocity_degree, pressure_degree, phase_field_degree,
                          constrained_domain)
    write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)
    return dict(uv=uv, timestampsfolder=timestampsfolder, **stats)


def temporal_hook(q_, tstep, t, u_, p_, phi_, rho_, sigma, epsilon, volume, statsfolder,
                  timestampsfolder, stat_interval, timestamps_interval, uv, mesh, **NS_namespace):
    """Write to Stats/tdata.dat: tstep, t, u0m, u1m, u2m, phim and the energies per
    volume (kinetic, interface gradient and double-well parts); and u, p and phi
    to Timestamps/ every timestamps_interval steps."""
    info_red("tstep = {}".format(tstep))
    if tstep % stat_interval == 0:
        write_tdata(statsfolder, tstep, t,
                    standard_stats(q_, u_, phi_, rho_, sigma, epsilon, volume, mesh))
    if tstep % timestamps_interval == 0:
        write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)
