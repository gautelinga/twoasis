"""A Taylor bubble, filling most of the width of a vertical channel that is periodic
along its length, seen in a frame moving with the bubble: the walls move at u_wall,
and a controller (regulator = "V_c" or "y_cm") adjusts the mean pressure gradient to
hold the bubble at u_target relative to the walls. Without the controller nothing
drives the flow (F0 = g0 = 0): the moving walls just drag the fluid along."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2022-08-30"
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


def constrained_domain(Ly, **params):
    return PeriodicBox({1: (0, Ly)})


def problem_parameters(NS_parameters, **NS_namespace):
    NS_parameters.update(
        T=100.0,
        Lx=1,
        Ly=2,
        res=1./64,
        R=0.4,
        dt=0.02,
        rho=[1000., 100.],
        mu=[10., 1.],
        theta=np.pi/2,
        epsilon=0.02,
        sigma=24.5,
        M=0.00002,
        u_wall=-0.5,
        u_target=0.5,
        F0=[0., 0.],
        g0=[0., 0.],
        velocity_degree=1,
        folder="taylorbubble2d_results",
        plot_interval=10,
        stat_interval=10,
        timestamps_interval=10,
        save_step=10,
        checkpoint=10,
        print_intermediate_info=10,
        use_krylov_solvers=True,
        regulator=None,  # "V_c" or "y_cm"
        solver="IPCS",
        max_iter=100,                 # Number of inner pressure velocity iterations on timestep
        max_error=1e-3,               # Tolerance for inner iterations (pressure velocity iterations)
        iters_on_first_timestep=200,  # Number of iterations on first timestep
    )


def mark_subdomains(subdomains, Lx, **NS_namespace):
    Faces(0, 0, Lx).mark(subdomains, 1)  # the walls


def create_bcs(V, subdomains, u_wall, **NS_namespace):
    bc_ux_lr = DirichletBC(V, 0, subdomains, 1)
    bc_uy_lr = DirichletBC(V, u_wall, subdomains, 1)
    return dict(u0=[bc_ux_lr],
                u1=[bc_uy_lr],
                p=[],
                phig=[])


def average_pressure_gradient(F0, **NS_namespace):
    # An Expression, so that the controller can change it during the run.
    return Expression(("0", "-forcing"), forcing=F0[1], degree=2)


def initialize(q_, q_1, q_2, x_1, x_2, bcs, epsilon, VV, Lx, Ly, R, u_wall, **NS_namespace):
    phig_init = interpolate(Expression(
        "tanh((sqrt(pow(x[0]-Lx/2, 2)+pow(x[1]-Ly/2, 2))-R)/(sqrt(2)*epsilon))",
        epsilon=epsilon, Lx=Lx, Ly=Ly, R=R, degree=2), VV['phig'].sub(0).collapse())
    assign(q_['phig'].sub(0), phig_init)
    q_1['phig'].vector()[:] = q_['phig'].vector()
    q_2['phig'].vector()[:] = q_1['phig'].vector()
    q_['u1'].vector()[:] = u_wall
    q_1['u1'].vector()[:] = u_wall
    q_2['u1'].vector()[:] = u_wall
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(mesh, newfolder, **NS_namespace):
    statsfolder = path.join(newfolder, "Stats")
    makedirs_on_root(statsfolder)
    volume = assemble(Constant(1.) * dx(domain=mesh))
    return dict(statsfolder=statsfolder, volume=volume, vel_prev=0, y_cm_prev=None)


def temporal_hook(q_, tstep, t, u_, phi_, rho_, sigma, epsilon, volume, g0, statsfolder,
                  stat_interval, mesh, dt, gradp_avg, u_wall, u_target, Lx, Ly, vel_prev, rho,
                  mu, regulator, y_cm_prev, **NS_namespace):
    """Write to Stats/tdata.dat: tstep, t, u0m, u1m, phim, the energies per volume
    (kinetic, interface gradient and double-well parts, gravitational), the
    bubble's (phi = -1) area vol_c, height y_cm and velocity V_c, the mean
    velocity V_tot, a zero column, and the driving pressure gradient."""
    returndict = dict()

    info_red("tstep = {}".format(tstep))
    if tstep % stat_interval == 0:
        dX = dx(domain=mesh)
        X = SpatialCoordinate(mesh)
        u0m = assemble(q_['u0'] * dX) / volume
        u1m = assemble(q_['u1'] * dX) / volume
        phim = assemble(phi_ * dX) / volume
        vol_c, (_, y_cm), (_, V_c) = phase_moments(q_, mesh, dX)
        V_tot = assemble(q_['u1'] * dX) / (Lx*Ly)

        E_kin = 0.5 * assemble(rho_ * (u_[0]**2 + u_[1]**2) * dX) / volume
        E_grav = assemble(-rho_ * (g0[0] * X[0] + g0[1] * X[1]) * dX) / volume
        E_int, E_pot = (E / volume for E in interface_energy(phi_, sigma, epsilon, dX))
        forcing = gradp_avg.forcing
        grav = 0
        if regulator is not None:
            # Adjust the driving force so that the bubble moves at u_target
            # relative to the wall. The gains alpha and beta are set from the
            # viscous time scale tau of the channel.
            k = 10.
            tau = Lx**2 / (12 * mu[0] / rho[0])
            alpha = rho[0] * 0.25 * (1+k)**2 / tau**2
            beta = rho[0] * k / tau

            if regulator == "V_c":
                vel = V_c
            elif regulator == "y_cm":
                if y_cm_prev is None:
                    # First sample of this run: there is no earlier position
                    # to compare with, so use the bubble velocity.
                    vel = V_c
                else:
                    # Remove the jump of ~Ly when the bubble crosses the
                    # periodic top/bottom boundary.
                    dy = y_cm - y_cm_prev
                    dy -= Ly * round(dy / Ly)
                    vel = dy / (stat_interval * dt)
                returndict["y_cm_prev"] = y_cm
            else:
                exit("Unknown regulator")

            forcing += alpha * (u_target + u_wall - vel) * stat_interval * dt - beta * (vel - vel_prev)
            gradp_avg.forcing = forcing
            returndict["vel_prev"] = vel

        if MPI.rank(MPI.comm_world) == 0:
            with open(path.join(statsfolder, "tdata.dat"), "a") as tfile:
                entries = [u0m, u1m, phim, E_kin, E_int, E_pot, E_grav, vol_c, y_cm, V_c, V_tot, grav, forcing]
                tfile.write(
                    ("{:d} {:.8f} " + " ".join(["{:.8f}" for _ in entries]) + "\n").format(
                    tstep, t, *entries))
    return returndict
