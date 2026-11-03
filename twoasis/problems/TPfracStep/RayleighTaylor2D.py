"""The Rayleigh-Taylor instability: a heavy fluid above a light one under gravity, with
a cosine perturbation of the interface, periodic sides and no-slip top and bottom.

The setup (a 1 x 4 box, density ratio 3, a perturbation of amplitude 0.1) is that of
G. Tryggvason, Numerical simulations of the Rayleigh-Taylor instability, J. Comput.
Phys. 75, 253-282 (1988), doi:10.1016/0021-9991(88)90112-X. The same setup, with equal
viscosities, is a test case in M. Shadkhah et al., MPFBench: A large scale dataset for
SciML of multi-phase-flows: droplet and bubble dynamics, arXiv:2502.07080 (2025).
As a phase-field problem, with a surface tension that is small but not zero so that the
Cahn-Hilliard equation does not reduce to pure transport, it is posed by B. S. Hosseini,
S. Turek, M. Möller and C. Palmes, Isogeometric analysis of the Navier-Stokes-Cahn-
Hilliard equations with application to incompressible two-phase flows, J. Comput. Phys.
348, 171-194 (2017), doi:10.1016/j.jcp.2017.07.029, and computed with the same initial
interface by A. Lovrić, W. G. Dettmer and D. Perić, arXiv:1911.06718 (2019), with equal
viscosities 0.0031316 and g = 9.80665."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2022-09-08"
__copyright__ = ("Copyright (C) 2022 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"


from os import path

import numpy as np
from dolfin import assemble, cos, Expression, MPI, SpatialCoordinate

from twoasis.common.utilities import set_subfunction
from twoasis.problems.TPfracStep import Bubble2D


# Periodic sides, walls, boundary conditions and the setup of statistics as in Bubble2D.
constrained_domain = Bubble2D.constrained_domain
contact_angles = Bubble2D.contact_angles
mesh = Bubble2D.mesh
create_bcs = Bubble2D.create_bcs
mark_subdomains = Bubble2D.mark_subdomains
pre_solve_hook = Bubble2D.pre_solve_hook


def problem_parameters(NS_parameters, **NS_namespace):
    NS_parameters.update(
        T=3.0,
        Lx=1,
        Ly=4,
        res=1./64,
        A=0.1,
        dt=0.001,
        rho=[3., 1.],
        mu=[0.01, 0.01],
        theta=np.pi/2,
        epsilon=0.02,                 # about 1.3 cells
        sigma=1e-12,
        M=1e-1,
        F0=[0., 0.],
        g0=[0., -1.0],
        velocity_degree=1,
        folder="rayleightaylor2d_results",
        plot_interval=10,
        stat_interval=10,
        timestamps_interval=10,
        save_step=10,
        checkpoint=10,
        print_intermediate_info=10,
        use_krylov_solvers=True,
        solver="IPCS",
        max_iter=100,                 # Number of inner pressure velocity iterations on timestep
        max_error=1e-6,               # Tolerance for inner iterations (pressure velocity iterations)
        iters_on_first_timestep=200,  # Number of iterations on first timestep
    )


def initialize(q_, q_1, x_1, x_2, bcs, epsilon, VV, Lx, Ly, A, **NS_namespace):
    # Heavy fluid (phi = 1) on top of light fluid, with the interface perturbed
    # by one cosine wavelength of amplitude A across the periodic width.
    set_subfunction(q_['phig'], 0, Expression(
        "tanh((x[1] - Ly/2 - A*cos(2*M_PI*x[0]/Lx))/(sqrt(2)*epsilon))",
        epsilon=epsilon, Lx=Lx, Ly=Ly, A=A, degree=2))
    q_1['phig'].vector()[:] = q_['phig'].vector()
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def temporal_hook(tstep, t, phi_, dx, mesh, Lx, statsfolder, stat_interval, **NS_namespace):
    """Statistics as in Bubble2D, plus the amplitude a of the interface's cosine
    mode in Stats/amplitude.dat. For an interface at y = Ly/2 + a*cos(k*x) with
    heavy fluid (phi = 1) above it, -1/Lx times the integral of phi*cos(k*x)
    equals a, at any amplitude where the interface is a graph over x."""
    Bubble2D.temporal_hook(tstep=tstep, t=t, phi_=phi_, dx=dx, mesh=mesh, Lx=Lx,
                           statsfolder=statsfolder, stat_interval=stat_interval, **NS_namespace)
    if tstep % stat_interval == 0:
        x = SpatialCoordinate(mesh)
        a = -assemble(phi_ * cos(2 * np.pi * x[0] / Lx) * dx) / Lx
        if MPI.rank(MPI.comm_world) == 0:
            with open(path.join(statsfolder, "amplitude.dat"), "a") as ofile:
                ofile.write("{:d} {:.8f} {:.10e}\n".format(tstep, t, a))
