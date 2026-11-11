"""A bubble rising in a liquid column, the benchmark of Hysing et al. (2009), with
no-slip walls at top and bottom and periodic sides (the benchmark has free-slip sides).

case=1 (the default) is test case 1: densities 1000 and 100, viscosities 10 and 1,
surface tension 24.5. case=2 is test case 2: densities 1000 and 1, viscosities 10 and
0.1, surface tension 1.96. Case 2 needs the phase field and the chemical potential
in P1, the pressure's space (phase_field_degree = chemical_potential_degree = 1):
in P2 the capillary force -phi*grad(g) leaves bulk forces the P1 pressure cannot
cancel, and in the light phase (density 1) they blow the run up within the first
hundredth of a time unit, whatever the step. It also needs a finer mesh than case 1:
at res=1/64 (epsilon = 1.3 cells) the run blows up at t ~ 0.37. case=2 sets
res=1/128 and dt=0.001; any of its values can still be given on the command line.

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
from dolfin import DirichletBC, dx, Expression

from twoasis.common.utilities import set_subfunction
from twoasis.problems import info_red
from twoasis.problems.TPfracStep.shared import (
    Faces, gravitational_energy, PeriodicBox, phase_moments, rectangle_mesh,
    setup_stats, standard_stats, write_tdata)


def mesh(Lx, Ly, res, **params):
    return rectangle_mesh(Lx, Ly, res)


def constrained_domain(Lx, **params):
    return PeriodicBox({0: (0, Lx)})


def problem_parameters(NS_parameters, commandline_kwargs, **NS_namespace):
    case = commandline_kwargs.get("case", 1)
    if case not in (1, 2):
        raise ValueError("case={!r}: Hysing et al.'s test case 1 or 2".format(case))
    NS_parameters.update(
        case=case,
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
        solver="BDF",
        max_iter=100,                 # Number of inner pressure velocity iterations on timestep
        max_error=1e-6,               # Tolerance for inner iterations (pressure velocity iterations)
        iters_on_first_timestep=200,  # Number of iterations on first timestep
    )
    if case == 2:
        NS_parameters.update(
            rho=[1000., 1.],
            mu=[10., 0.1],
            sigma=1.96,
            res=1./128,
            dt=0.001,
            phase_field_degree=1,
            chemical_potential_degree=1,
            anderson_depth=2,
            folder="bubble2d_case2_results",
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
    set_subfunction(q_['phig'], 0, Expression(
        "tanh((sqrt(pow(x[0]-Lx/2, 2)+pow(x[1]-Ly/4, 2))-R)/(sqrt(2)*epsilon))",
        epsilon=epsilon, Lx=Lx, Ly=Ly, R=R, degree=2))
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
