"""A water drop in air (CGS units) whose circular shape is perturbed by a mode-3
cosine, oscillating as it relaxes, in a box with no-slip walls.

For small amplitudes, mode n oscillates at the frequency
    omega_n^2 = (n^3 - n) sigma / ((rho_drop + rho_out) R^3),
Rayleigh's frequency for a cylindrical jet (Lord Rayleigh, On the capillary phenomena
of jets, Proc. R. Soc. Lond. 29, 71-97 (1879), doi:10.1098/rspl.1879.0015) extended to
an outer fluid by D. E. Fyfe, E. S. Oran and M. J. Fritts, Surface tension and
viscosity with Lagrangian hydrodynamics on a triangular mesh, J. Comput. Phys. 76,
349-384 (1988), doi:10.1016/0021-9991(88)90147-7. Here the period is 0.210, so the
kinetic energy has a minimum every 0.105.

Adapted from the oscillating 2D water drop of A. Lovrić, W. G. Dettmer and D. Perić,
Low order finite element methods for the Navier-Stokes-Cahn-Hilliard equations,
arXiv:1911.06718 (2019), which has the same densities and surface tension, a drop of
radius 0.0125 cm and modes n = 2, 3, 4.
A similar case, a water drop oscillating in air in 3D, is in the benchmark of
A. Lippert, T. Tolle, A. Dörr and T. Marić, A benchmark for surface-tension-driven
incompressible two-phase flows, arXiv:2212.02904 (2022)."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2022-09-18"
__copyright__ = ("Copyright (C) 2022 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"

import numpy as np
from dolfin import assign, DirichletBC, dx, Expression, interpolate

from twoasis.problems import info_red
from twoasis.problems.TPfracStep.shared import (
    Faces, gravitational_energy, phase_moments, rectangle_mesh, setup_stats,
    standard_stats, write_tdata)


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
        mu=[0.0101, 0.000182],
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
    return dict(save_step=max(1, round(save_interval / dt)),
                stat_step=max(1, round(stat_interval / dt)),
                **setup_stats(mesh, newfolder))


def temporal_hook(q_, tstep, t, u_, phi_, rho_, sigma, epsilon, volume, g0, statsfolder,
                  stat_step, mesh, **NS_namespace):
    """Write to Stats/tdata.dat: tstep, t, u0m, u1m, phim, the energies per volume
    (kinetic, interface gradient and double-well parts, gravitational), and the
    drop's (phi = 1) area, height and vertical velocity."""
    info_red("tstep = {}".format(tstep))
    if tstep % stat_step == 0:
        entries = standard_stats(q_, u_, phi_, rho_, sigma, epsilon, volume, mesh)
        entries.append(gravitational_energy(rho_, g0, volume, mesh))
        vol_c, (_, y_cm), (_, V_c) = phase_moments(q_, mesh, dx(domain=mesh), phase=1)
        write_tdata(statsfolder, tstep, t, entries + [vol_c, y_cm, V_c])
