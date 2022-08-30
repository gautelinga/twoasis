__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..TPfracStep import *
import matplotlib.pyplot as plt

#set_log_active(False)
class PeriodicDomain(SubDomain):
    def inside(self, x, on_boundary):
        return bool(near(x[0], 0) and on_boundary)

    def map(self, x, y):
        if near(x[0], 2.0):
            y[0] = x[0] - 2.0
            y[1] = x[1]
        else:
            y[0] = x[0]
            y[1] = x[1]


# Create a mesh
def mesh(Nx=128, Ny=64, **params):
    m = UnitSquareMesh(Nx, Ny)
    x = m.coordinates()
    x[:, 0] *= 2
    #x[:] = (x - 0.5) * 2
    #x[:] = 0.5*(cos(pi*(x-1.) / 2.) + 1.)
    return m

top = "std::abs(x[1]-1) < 1e-8"
bottom = "std::abs(x[1]) < 1e-8"

# Override some problem specific parameters
def problem_parameters(NS_parameters, NS_expressions, scalar_components, Schmidt, **NS_namespace):
    NS_parameters.update(
        nu=0.01,
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
        iters_on_first_timestep=200,  # Number of iterations on first timestep)
    )
    #scalar_components += ["alfa", "beta"]
    #Schmidt["alfa"] = 1.
    #Schmidt["beta"] = 10.

    #NS_parameters['krylov_solvers'] = {'monitor_convergence': False,
    #                                   'report': False,
    #                                   'relative_tolerance': 1e-10,
    #                                   'absolute_tolerance': 1e-10}
    NS_expressions.update(dict(
        constrained_domain=PeriodicDomain()
    ))

# Specify boundary conditions
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
    phig_init = interpolate(Expression(
        "tanh((sqrt(pow(x[0]-1, 2)+pow(x[1]-0.5, 2))-0.3)/(sqrt(2)*epsilon))",
        epsilon=epsilon, degree=2), VV['phig'].sub(0).collapse())
    assign(q_['phig'].sub(0), phig_init)
    q_1['phig'].vector()[:] = q_['phig'].vector()
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(mesh, velocity_degree, **NS_namespace):
    Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree)
    return dict(uv=Function(Vv))


def temporal_hook(q_, tstep, u_, p_, phi_, plot_interval, **NS_namespace):
    info_red("tstep = {}".format(tstep))
    if tstep % plot_interval == 0 and False:
        plot(u_, title='Velocity')
        plt.show()
        plot(phi_, title='Phase field')
        plt.show()
        #plot(p_, title='Pressure')
        #plot(q_['alfa'], title='alfa')
        #plot(q_['beta'], title='beta')


def theend_hook(u_, p_, testing, **NS_namespace):
    if not testing:
        plot(u_, title='Velocity')
        plot(p_, title='Pressure')

    u_norm = norm(u_[0].vector())
    if MPI.rank(MPI.comm_world) == 0 and testing:
        print("Velocity norm = {0:2.6e}".format(u_norm))
