"""A bubble rising in a liquid column, test case 1 of the benchmark of Hysing et al.
(2009), with no-slip walls at top and bottom and periodic sides."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2022-08-19"
__copyright__ = ("Copyright (C) 2022 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"

from os import path

import numpy as np
from dolfin import (
    assemble, assign, Constant, DirichletBC, dx, Expression, interpolate, MPI,
    SpatialCoordinate)

from twoasis.problems import info_red
from twoasis.problems.TPfracStep.shared import (
    Faces, interface_energy, makedirs_on_root, PeriodicBox, phase_moments,
    rectangle_mesh)


def mesh(Lx, Ly, res, **params):
    return rectangle_mesh(Lx, Ly, res)


def constrained_domain(Lx, **params):
    return PeriodicBox({0: (0, Lx)})


def problem_parameters(NS_parameters, **NS_namespace):
    NS_parameters.update(
        T=3.0,
        Lx=1,
        Ly=2,
        res=1./64,
        R=0.25,
        dt=0.002,
        rho=[1000., 100.],
        mu=[10., 1.],
        theta=np.pi/2,
        epsilon=0.02,
        sigma=24.5,
        M=0.00002,
        F0=[0., 0.],
        g0=[0., -0.98],
        velocity_degree=2,
        folder="bubble2d_results",
        plot_interval=10,
        stat_interval=10,
        timestamps_interval=10,
        save_step=10,
        checkpoint=10,
        print_intermediate_info=10,
        use_krylov_solvers=True,
        solver="BDF",
        max_iter=100,                 # Number of inner pressure velocity iterations on timestep
        max_error=1e-6,               # Tolerance for inner iterations (pressure velocity iterations)
        iters_on_first_timestep=200,  # Number of iterations on first timestep
    )


def mark_subdomains(subdomains, Ly, **NS_namespace):
    Faces(1, 0, Ly).mark(subdomains, 2)  # top and bottom walls


def contact_angles(theta, **NS_namespace):
    return [(theta, 2)]


def create_bcs(V, subdomains, **NS_namespace):
    bc_u_tb = DirichletBC(V, 0, subdomains, 2)
    return dict(u0=[bc_u_tb],
                u1=[bc_u_tb],
                p=[],
                phig=[])


def initialize(q_, q_1, q_2, x_1, x_2, bcs, epsilon, VV, Lx, Ly, R, **NS_namespace):
    phig_init = interpolate(Expression(
        "tanh((sqrt(pow(x[0]-Lx/2, 2)+pow(x[1]-Ly/4, 2))-R)/(sqrt(2)*epsilon))",
        epsilon=epsilon, Lx=Lx, Ly=Ly, R=R, degree=2), VV['phig'].sub(0).collapse())
    assign(q_['phig'].sub(0), phig_init)
    q_1['phig'].vector()[:] = q_['phig'].vector()
    q_2['phig'].vector()[:] = q_['phig'].vector()
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(mesh, newfolder, **NS_namespace):
    statsfolder = path.join(newfolder, "Stats")
    makedirs_on_root(statsfolder)
    volume = assemble(Constant(1.) * dx(domain=mesh))
    return dict(statsfolder=statsfolder, volume=volume)


def temporal_hook(q_, tstep, t, u_, phi_, rho_, sigma, epsilon, volume, g0, statsfolder,
                  stat_interval, mesh, **NS_namespace):
    """Write to Stats/tdata.dat: tstep, t, the mean velocity (u0m, u1m) and phase
    field (phim), the energies per volume (kinetic, interface gradient and
    double-well parts, gravitational), and the bubble's (phi = -1) area, height
    and vertical velocity."""
    info_red("tstep = {}".format(tstep))
    if tstep % stat_interval == 0:
        dX = dx(domain=mesh)
        X = SpatialCoordinate(mesh)
        u0m = assemble(q_['u0'] * dX) / volume
        u1m = assemble(q_['u1'] * dX) / volume
        phim = assemble(phi_ * dX) / volume
        vol_c, (_, y_cm), (_, V_c) = phase_moments(q_, mesh, dX)
        E_kin = 0.5 * assemble(rho_ * (u_[0]**2 + u_[1]**2) * dX) / volume
        E_grav = assemble(-rho_ * (g0[0] * X[0] + g0[1] * X[1]) * dX) / volume
        E_int, E_pot = (E / volume for E in interface_energy(phi_, sigma, epsilon, dX))
        if MPI.rank(MPI.comm_world) == 0:
            with open(path.join(statsfolder, "tdata.dat"), "a") as tfile:
                entries = [u0m, u1m, phim, E_kin, E_int, E_pot, E_grav, vol_c, y_cm, V_c]
                tfile.write(("{:d} {:.8f} " + " ".join("{:.8f}" for _ in entries) + "\n").format(
                    tstep, t, *entries))
