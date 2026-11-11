"""A Taylor bubble, filling most of the width of a vertical channel that is periodic
along its length, seen in a frame moving with the bubble: the walls move at u_wall,
and a controller (regulator = "V_c" or "y_cm") adjusts the driving force F0[1] to
hold the bubble at u_target relative to the walls. Without the controller nothing
drives the flow (F0 = g0 = 0): the moving walls just drag the fluid along.

With the default dt = 0.005, a 2-time-unit run is within 0.3% of one at dt = 0.0016
in kinetic energy and bubble position; at dt = 0.02 the kinetic energy of the
translating fluid decays by 13% over that time. The controller acts every
stat_interval*dt, 0.2 time units by default."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2022-08-30"
__copyright__ = ("Copyright (C) 2022 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"

import numpy as np
from dolfin import assemble, DirichletBC, dx, Expression

from twoasis.common.utilities import set_subfunction
from twoasis.problems import info_red
from twoasis.problems.TPfracStep.shared import (
    Faces, gravitational_energy, PeriodicBox, phase_moments, rectangle_mesh,
    setup_stats, standard_stats, write_tdata)


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
        dt=0.005,
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
        plot_interval=40,
        stat_interval=40,
        timestamps_interval=40,
        save_step=40,
        checkpoint=40,
        print_intermediate_info=40,
        regulator=None,  # "V_c" or "y_cm"
        # The controller's state (forcing, vel_prev, y_cm_prev), set during the
        # run. It is saved with the checkpoint's parameters, so that a restart
        # continues the control where it stopped.
        controller=None,
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


def average_pressure_gradient(F0, controller, **NS_namespace):
    """The mean pressure gradient, -F0, with its y component `forcing` (the
    driving force along y) an Expression parameter, so that the controller can
    change it during the run."""
    forcing = F0[1] if controller is None else controller["forcing"]
    return Expression(("-fx", "-forcing"), fx=F0[0], forcing=forcing, degree=2)


def initialize(q_, q_1, q_2, x_1, x_2, bcs, epsilon, VV, Lx, Ly, R, u_wall, **NS_namespace):
    set_subfunction(q_['phig'], 0, Expression(
        "tanh((sqrt(pow(x[0]-Lx/2, 2)+pow(x[1]-Ly/2, 2))-R)/(sqrt(2)*epsilon))",
        epsilon=epsilon, Lx=Lx, Ly=Ly, R=R, degree=2))
    q_1['phig'].vector()[:] = q_['phig'].vector()
    q_2['phig'].vector()[:] = q_1['phig'].vector()
    q_['u1'].vector()[:] = u_wall
    q_1['u1'].vector()[:] = u_wall
    q_2['u1'].vector()[:] = u_wall
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(mesh, newfolder, regulator, **NS_namespace):
    if regulator not in (None, "V_c", "y_cm"):
        raise ValueError("Unknown regulator {!r}; use None, 'V_c' or 'y_cm'".format(regulator))
    return setup_stats(mesh, newfolder)


def temporal_hook(q_, tstep, t, u_, phi_, rho_, sigma, epsilon, volume, g0, statsfolder,
                  stat_interval, mesh, dt, gradp_avg, u_wall, u_target, Lx, Ly, rho, mu,
                  regulator, NS_parameters, **NS_namespace):
    """Write to Stats/tdata.dat: tstep, t, u0m, u1m, phim, the energies per volume
    (kinetic, interface gradient and double-well parts, gravitational), the
    bubble's (phi = -1) area vol_c, height y_cm and velocity V_c, the mean
    velocity V_tot, a zero column, and the driving force along y (F0[1])."""
    info_red("tstep = {}".format(tstep))
    if tstep % stat_interval == 0:
        dX = dx(domain=mesh)
        entries = standard_stats(q_, u_, phi_, rho_, sigma, epsilon, volume, mesh)
        entries.append(gravitational_energy(rho_, g0, volume, mesh))
        vol_c, (_, y_cm), (_, V_c) = phase_moments(q_, mesh, dX)
        V_tot = assemble(q_['u1'] * dX) / (Lx*Ly)

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

            state = NS_parameters["controller"] or dict(vel_prev=0., y_cm_prev=None)
            if regulator == "y_cm" and state["y_cm_prev"] is not None:
                # Remove the jump of ~Ly when the bubble crosses the periodic
                # top/bottom boundary.
                dy = y_cm - state["y_cm_prev"]
                dy -= Ly * round(dy / Ly)
                vel = dy / (stat_interval * dt)
            else:
                # Regulating V_c, or the first sample of y_cm: there is no
                # earlier position to compare with, so use the bubble velocity.
                vel = V_c

            forcing += (alpha * (u_target + u_wall - vel) * stat_interval * dt
                        - beta * (vel - state["vel_prev"]))
            gradp_avg.forcing = forcing
            NS_parameters["controller"] = dict(forcing=forcing, vel_prev=vel, y_cm_prev=y_cm)

        write_tdata(statsfolder, tstep, t, entries + [vol_c, y_cm, V_c, V_tot, grav, forcing])
