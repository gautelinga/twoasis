"""A bubble in a through-flow: the velocity is prescribed at the bottom (inflow) and
at the top (outflow), with periodic sides. The phase entering at the bottom is
injected_phase, by default the surrounding phase (phi = 1)."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2023-08-08"
__copyright__ = ("Copyright (C) 2023 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"

from os import path

import numpy as np
from dolfin import (
    assemble, assign, Constant, DirichletBC, dx, Expression, interpolate, MPI)

from twoasis.problems import info_blue, info_red
from twoasis.problems.TPfracStep.shared import (
    centred, Faces, interface_energy, makedirs_on_root, PeriodicBox, read_phase_field,
    rectangle_mesh)


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
        dt=0.01,
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
        plot_interval=10,
        stat_interval=10,
        timestamps_interval=10,
        save_step=10,
        checkpoint=10,
        print_intermediate_info=10,
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
        phi_init = interpolate(Expression(
            "tanh((sqrt(pow(x[0], 2)+pow(x[1], 2))-R)/(sqrt(2)*epsilon))",
            epsilon=epsilon, R=R, degree=2), VV['phig'].sub(0).collapse())
    else:
        info_blue("initializing from: " + initial_state)
        phi_init = read_phase_field(initial_state, mesh, q_)
    assign(q_['phig'].sub(0), phi_init)

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


def temporal_hook(q_, tstep, t, u_, phi_, rho_, sigma, epsilon, volume, statsfolder,
                  stat_interval, mesh, **NS_namespace):
    """Write to Stats/tdata.dat: tstep, t, u0m, u1m, phim and the energies per
    volume (kinetic, interface gradient and double-well parts)."""
    info_red("tstep = {}".format(tstep))
    if tstep % stat_interval == 0:
        dX = dx(domain=mesh)
        u0m = assemble(q_['u0'] * dX) / volume
        u1m = assemble(q_['u1'] * dX) / volume
        phim = assemble(phi_ * dX) / volume
        E_kin = 0.5 * assemble(rho_ * (u_[0]**2 + u_[1]**2) * dX) / volume
        E_int, E_pot = (E / volume for E in interface_energy(phi_, sigma, epsilon, dX))
        if MPI.rank(MPI.comm_world) == 0:
            with open(path.join(statsfolder, "tdata.dat"), "a") as tfile:
                entries = [u0m, u1m, phim, E_kin, E_int, E_pot]
                tfile.write(("{:d} {:.8f} " + " ".join("{:.8f}" for _ in entries) + "\n").format(
                    tstep, t, *entries))
