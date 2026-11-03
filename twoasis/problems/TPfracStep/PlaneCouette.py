"""A drop deforming in plane Couette flow between two walls moving in opposite
directions, periodic along the walls."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2022-03-18"
__copyright__ = ("Copyright (C) 2022 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"

from dolfin import DirichletBC, Expression, UnitSquareMesh

from twoasis.common.utilities import set_subfunction
from twoasis.problems import info_red
from twoasis.problems.TPfracStep.shared import PeriodicBox


def mesh(Nx=128, Ny=64, **params):
    m = UnitSquareMesh(Nx, Ny)
    m.coordinates()[:, 0] *= 2
    return m


def constrained_domain(**params):
    return PeriodicBox({0: (0, 2.0)})


top = "std::abs(x[1]-1) < 1e-8"
bottom = "std::abs(x[1]) < 1e-8"


def problem_parameters(NS_parameters, **NS_namespace):
    NS_parameters.update(
        T=10.0,
        dt=0.002,
        u_wall=1.0,
        rho=[1, 1.0],
        mu=[1, 0.1],
        epsilon=0.02,
        sigma=10,
        velocity_degree=1,
        folder="planecouette_results",
        plot_interval=10,
        save_step=10,
        checkpoint=10,
        print_intermediate_info=10,
        use_krylov_solvers=True,
        solver="IPCS",
        max_iter=100,                 # Number of inner pressure velocity iterations on timestep
        max_error=1e-3,               # Tolerance for inner iterations (pressure velocity iterations)
        iters_on_first_timestep=200,  # Number of iterations on first timestep
    )


def create_bcs(V, u_wall, **NS_namespace):
    bc_ux_top = DirichletBC(V, u_wall, top)
    bc_ux_bottom = DirichletBC(V, -u_wall, bottom)
    bc_uy_top = DirichletBC(V, 0, top)
    bc_uy_bottom = DirichletBC(V, 0, bottom)
    return dict(u0=[bc_ux_top, bc_ux_bottom],
                u1=[bc_uy_top, bc_uy_bottom],
                p=[],
                phig=[])


def initialize(q_, q_1, x_1, x_2, bcs, epsilon, VV, **NS_namespace):
    set_subfunction(q_['phig'], 0, Expression(
        "tanh((sqrt(pow(x[0]-1, 2)+pow(x[1]-0.5, 2))-0.3)/(sqrt(2)*epsilon))",
        epsilon=epsilon, degree=2))
    q_1['phig'].vector()[:] = q_['phig'].vector()
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def temporal_hook(tstep, **NS_namespace):
    info_red("tstep = {}".format(tstep))
