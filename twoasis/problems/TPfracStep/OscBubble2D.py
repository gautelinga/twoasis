"""A water drop in air (CGS units) whose circular shape is perturbed by a mode-3
cosine, oscillating as it relaxes, in a box with no-slip walls. The viscosities are
ten times those of water and air."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2022-09-18"
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
    Faces, interface_energy, makedirs_on_root, phase_moments, rectangle_mesh)


def mesh(Lx, Ly, res, **params):
    return rectangle_mesh(Lx, Ly, res)


def problem_parameters(NS_parameters, **NS_namespace):
    NS_parameters.update(
        T=3.0,
        Lx=5.0,
        Ly=5.0,
        res=1./16,
        amplitude=0.1,
        mode_order=3,
        R=1.25,
        dt=0.0005,
        rho=[0.998, 0.0012],
        mu=[0.101, 0.00182],
        theta=np.pi/2,
        epsilon=0.06,
        sigma=73.0,
        M=1e-2,
        F0=[0., 0.],
        g0=[0., 0.],
        velocity_degree=1,
        folder="oscbubble2d_results",
        stat_interval=0.002,   # in time units, like save_interval
        save_interval=0.02,
        checkpoint=10,
        print_intermediate_info=10,
        use_krylov_solvers=True,
        solver="IPCS",
        AB_projection_pressure=False,
        max_iter=100,                 # Number of inner pressure velocity iterations on timestep
        max_error=1e-3,               # Tolerance for inner iterations (pressure velocity iterations)
        iters_on_first_timestep=200,  # Number of iterations on first timestep
    )


def mark_subdomains(subdomains, Lx, Ly, **NS_namespace):
    Faces(0, 0, Lx).mark(subdomains, 1)
    Faces(1, 0, Ly).mark(subdomains, 1)


def create_bcs(V, subdomains, **NS_namespace):
    bc_u = DirichletBC(V, 0, subdomains, 1)
    return dict(u0=[bc_u],
                u1=[bc_u],
                p=[],
                phig=[])


def initialize(q_, q_1, q_2, x_1, x_2, bcs, epsilon, VV, Lx, Ly, R, amplitude, mode_order,
               **NS_namespace):
    # Water (phi = 1) inside r < R*(1 + amplitude*cos(mode_order*angle)).
    phig_init = interpolate(Expression(
        "-tanh((sqrt(pow(x[0]-Lx/2, 2)+pow(x[1]-Ly/2, 2))-R*(1+A*cos(n*atan2(x[1]-Ly/2, x[0]-Lx/2))))/(sqrt(2)*epsilon))",
        epsilon=epsilon, Lx=Lx, Ly=Ly, R=R, A=amplitude,
        n=mode_order, degree=2), VV['phig'].sub(0).collapse())
    assign(q_['phig'].sub(0), phig_init)
    q_1['phig'].vector()[:] = q_['phig'].vector()
    q_2['phig'].vector()[:] = q_['phig'].vector()
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(mesh, newfolder, dt, save_interval, stat_interval, **NS_namespace):
    statsfolder = path.join(newfolder, "Stats")
    makedirs_on_root(statsfolder)
    volume = assemble(Constant(1.) * dx(domain=mesh))
    return dict(statsfolder=statsfolder, volume=volume,
                save_step=round(save_interval / dt), stat_step=round(stat_interval / dt))


def temporal_hook(q_, tstep, t, u_, phi_, rho_, sigma, epsilon, volume, g0, statsfolder,
                  stat_step, mesh, **NS_namespace):
    """Write to Stats/tdata.dat: tstep, t, u0m, u1m, phim, the energies per volume
    (kinetic, interface gradient and double-well parts, gravitational), and the
    drop's (phi = 1) area, height and vertical velocity."""
    info_red("tstep = {}".format(tstep))
    if tstep % stat_step == 0:
        dX = dx(domain=mesh)
        X = SpatialCoordinate(mesh)
        u0m = assemble(q_['u0'] * dX) / volume
        u1m = assemble(q_['u1'] * dX) / volume
        phim = assemble(phi_ * dX) / volume
        vol_c, (_, y_cm), (_, V_c) = phase_moments(q_, mesh, dX, phase=1)
        E_kin = 0.5 * assemble(rho_ * (u_[0]**2 + u_[1]**2) * dX) / volume
        E_grav = assemble(-rho_ * (g0[0] * X[0] + g0[1] * X[1]) * dX) / volume
        E_int, E_pot = (E / volume for E in interface_energy(phi_, sigma, epsilon, dX))
        if MPI.rank(MPI.comm_world) == 0:
            with open(path.join(statsfolder, "tdata.dat"), "a") as tfile:
                entries = [u0m, u1m, phim, E_kin, E_int, E_pot, E_grav, vol_c, y_cm, V_c]
                tfile.write(("{:d} {:.8f} " + " ".join("{:.8f}" for _ in entries) + "\n").format(
                    tstep, t, *entries))
