"""A bubble rising in a liquid column, test case 1 of the benchmark of Hysing et al.
(2009), with no-slip walls at top and bottom and periodic sides.

S. Hysing, S. Turek, D. Kuzmin, N. Parolini, E. Burman, S. Ganesan and L. Tobiska,
Quantitative benchmark computations of two-dimensional bubble dynamics, Int. J. Numer.
Meth. Fluids 60, 1259-1288 (2009), doi:10.1002/fld.1934. The same case with
diffuse-interface (Cahn-Hilliard) models: S. Aland and A. Voigt, Benchmark computations
of diffuse interface models for two-dimensional bubble dynamics, Int. J. Numer. Meth.
Fluids 69, 747-761 (2012), doi:10.1002/fld.2611."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2022-08-19"
__copyright__ = ("Copyright (C) 2022 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"

import numpy as np
from dolfin import assign, DirichletBC, dx, Expression, interpolate

from twoasis.problems import info_red
from twoasis.problems.TPfracStep.shared import (
    Faces, gravitational_energy, PeriodicBox, phase_moments, rectangle_mesh,
    setup_stats, standard_stats, write_tdata)


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
    return setup_stats(mesh, newfolder)


def temporal_hook(q_, tstep, t, u_, phi_, rho_, sigma, epsilon, volume, g0, statsfolder,
                  stat_interval, mesh, **NS_namespace):
    """Write to Stats/tdata.dat: tstep, t, the mean velocity (u0m, u1m) and phase
    field (phim), the energies per volume (kinetic, interface gradient and
    double-well parts, gravitational), and the bubble's (phi = -1) area, height
    and vertical velocity."""
    info_red("tstep = {}".format(tstep))
    if tstep % stat_interval == 0:
        entries = standard_stats(q_, u_, phi_, rho_, sigma, epsilon, volume, mesh)
        entries.append(gravitational_energy(rho_, g0, volume, mesh))
        vol_c, (_, y_cm), (_, V_c) = phase_moments(q_, mesh, dx(domain=mesh))
        write_tdata(statsfolder, tstep, t, entries + [vol_c, y_cm, V_c])
