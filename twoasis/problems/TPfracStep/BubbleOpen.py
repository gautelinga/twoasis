"""A bubble in a through-flow: the velocity is prescribed at the bottom (inflow) and
at the top (outflow), with periodic sides. The phase entering at the bottom is
injected_phase, by default the surrounding phase (phi = 1).

The interface energy is still sensitive to dt at the default dt = 0.005: it is ~2.5%
below a run at dt = 0.0025 while the bubble passes through (and ~5% at dt = 0.01)."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2023-08-08"
__copyright__ = ("Copyright (C) 2023 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"

import numpy as np
from dolfin import DirichletBC, Expression

from twoasis.common.utilities import set_subfunction
from twoasis.problems import info_blue, info_red
from twoasis.problems.TPfracStep.shared import (
    centred, Faces, PeriodicBox, read_phase_field, rectangle_mesh, setup_stats,
    standard_stats, write_tdata)


def mesh(Lx, Ly, res, **params):
    return rectangle_mesh(Lx, Ly, res, centred=True)


def constrained_domain(Lx, **params):
    return PeriodicBox({0: centred(Lx)})


def problem_parameters(NS_parameters, **NS_namespace):
    NS_parameters.update(
        T=100.0,
        Lx=4,
        Ly=4,
        res=0.1,
        R=1.0,
        dt=0.005,
        rho=[1, 1],
        mu=[1, 10],
        u0=1.0,
        theta=np.pi/2,
        epsilon=0.1,
        sigma=5.0,
        M=0.0001,
        F0=[0., 0.],
        g0=[0., 0.],
        velocity_degree=1,
        folder="bubbleopen_results",
        plot_interval=20,
        stat_interval=20,
        timestamps_interval=20,
        save_step=20,
        checkpoint=20,
        print_intermediate_info=20,
        use_krylov_solvers=True,
        solver="BDF",
        bdf_order=1,
        AB_projection_pressure=False,
        max_iter=10,                 # Number of inner pressure velocity iterations on timestep
        max_error=1e-3,               # Tolerance for inner iterations (pressure velocity iterations)
        iters_on_first_timestep=10,  # Number of iterations on first timestep
        initial_state=None,
        injected_phase=1,
    )


def contact_angles(**NS_namespace):
    return []  # no walls: the top and bottom are open


def mark_subdomains(subdomains, Ly, **NS_namespace):
    Faces(1, Ly/2).mark(subdomains, 1)   # top
    Faces(1, -Ly/2).mark(subdomains, 2)  # bottom


def create_bcs(V, u0, W, subdomains, injected_phase, **NS_namespace):
    bc_ux_top = DirichletBC(V, 0, subdomains, 1)
    bc_uy_top = DirichletBC(V, u0, subdomains, 1)
    bc_ux_btm = DirichletBC(V, 0, subdomains, 2)
    bc_uy_btm = DirichletBC(V, u0, subdomains, 2)
    bc_phig_btm = DirichletBC(W.sub(0), injected_phase, subdomains, 2)
    return dict(u0=[bc_ux_btm, bc_ux_top],
                u1=[bc_uy_btm, bc_uy_top],
                p=[],
                phig=[bc_phig_btm])


def initialize(q_, q_1, q_2, x_1, x_2, bcs, epsilon, VV, R, initial_state, mesh, **NS_namespace):
    if initial_state is None:
        phi_init = Expression(
            "tanh((sqrt(pow(x[0], 2)+pow(x[1], 2))-R)/(sqrt(2)*epsilon))",
            epsilon=epsilon, R=R, degree=2)
    else:
        info_blue("initializing from: " + initial_state)
        phi_init = read_phase_field(initial_state, mesh, q_)
    set_subfunction(q_['phig'], 0, phi_init)

    q_1['phig'].vector()[:] = q_['phig'].vector()
    q_2['phig'].vector()[:] = q_['phig'].vector()
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(mesh, newfolder, **NS_namespace):
    return setup_stats(mesh, newfolder)


def temporal_hook(q_, tstep, t, u_, phi_, rho_, sigma, epsilon, volume, statsfolder,
                  stat_interval, mesh, **NS_namespace):
    """Write to Stats/tdata.dat: tstep, t, u0m, u1m, phim and the energies per
    volume (kinetic, interface gradient and double-well parts)."""
    info_red("tstep = {}".format(tstep))
    if tstep % stat_interval == 0:
        write_tdata(statsfolder, tstep, t,
                    standard_stats(q_, u_, phi_, rho_, sigma, epsilon, volume, mesh))
