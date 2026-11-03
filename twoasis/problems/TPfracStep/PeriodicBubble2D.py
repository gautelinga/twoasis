"""A bubble carried by flow in a channel with no-slip walls, periodic along the channel
and driven by a body force along it.

The default dt is at the capillary time-step limit sqrt(rho h^3/(2 pi sigma)) ~ 2.5e-4
(h = res/2 for P2): with dt = 5e-4 or more the run blows up within ~0.03 time units."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2024-04-09"
__copyright__ = ("Copyright (C) 2024 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"


import numpy as np
from dolfin import DirichletBC, dx, Expression

from twoasis.common.utilities import set_subfunction
from twoasis.problems import info_red
from twoasis.problems.TPfracStep import Bubble2D
from twoasis.problems.TPfracStep.shared import phase_moments, standard_stats, write_tdata


# Periodic along the channel, walls at top and bottom and statistics set up as
# in Bubble2D.
mesh = Bubble2D.mesh
constrained_domain = Bubble2D.constrained_domain
contact_angles = Bubble2D.contact_angles
mark_subdomains = Bubble2D.mark_subdomains
pre_solve_hook = Bubble2D.pre_solve_hook


def problem_parameters(NS_parameters, **NS_namespace):
    NS_parameters.update(
        T=3.0,
        Lx=2,
        Ly=1,
        res=1./64,
        R=0.4,
        dt=0.00025,
        rho=[1., 1.],
        mu=[1., 1.],
        theta=np.pi/2,
        epsilon=0.02,
        sigma=10.0,
        M=0.00002,
        u0=0.0,
        F0=[0., 0.],
        g0=[1.0, 0],
        velocity_degree=2,
        folder="periodicbubble2d_results",
        plot_interval=80,
        stat_interval=80,
        timestamps_interval=80,
        save_step=80,
        checkpoint=80,
        print_intermediate_info=80,
        use_krylov_solvers=True,
        solver="BDF",
        max_iter=100,                 # Number of inner pressure velocity iterations on timestep
        max_error=1e-6,               # Tolerance for inner iterations (pressure velocity iterations)
        iters_on_first_timestep=200  # Number of iterations on first timestep
    )


# Specify boundary conditions
def create_bcs(V, subdomains, u0, **NS_namespace):
    # The walls move along the channel at u0 (no flow through them).
    return dict(u0=[DirichletBC(V, u0, subdomains, 2)],
                u1=[DirichletBC(V, 0, subdomains, 2)],
                p=[],
                phig=[])


def initialize(q_, q_1, q_2, x_1, x_2, bcs, epsilon, u0, VV, Lx, Ly, R, **NS_namespace):
    # Uniform flow, also at the previous time level, from which the solvers step
    for q in (q_, q_1):
        q['u0'].vector()[:] = u0
        q['u1'].vector()[:] = 0.
    # Centered
    set_subfunction(q_['phig'], 0, Expression(
        "tanh((sqrt(pow(x[0]-Lx/2, 2)+pow(x[1]-Ly/2, 2))-R)/(sqrt(2)*epsilon))",
        epsilon=epsilon, Lx=Lx, Ly=Ly, R=R, degree=2))
    q_1['phig'].vector()[:] = q_['phig'].vector()
    q_2['phig'].vector()[:] = q_['phig'].vector()
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def temporal_hook(q_, tstep, t, u_, phi_, rho_, sigma, epsilon, volume, statsfolder,
                  stat_interval, mesh, **NS_namespace):
    """Write to Stats/tdata.dat: tstep, t, the mean velocity (u0m, u1m) and phase
    field (phim), the energies per volume (kinetic, interface gradient and
    double-well parts), and the bubble's (phi = -1) area, height across the
    channel and velocity along it."""
    info_red("tstep = {}".format(tstep))
    if tstep % stat_interval == 0:
        entries = standard_stats(q_, u_, phi_, rho_, sigma, epsilon, volume, mesh)
        vol_c, (_, y_cm), (V_c, _) = phase_moments(q_, mesh, dx(domain=mesh))
        write_tdata(statsfolder, tstep, t, entries + [vol_c, y_cm, V_c])
