__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2023-06-5"
__copyright__ = "Copyright (C) 2023 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..TPfracStep import *
import matplotlib.pyplot as plt
import numpy as np
from os import makedirs
from twoasis.problems.TPfracStep.Porous2DFlux import PBC2, Bottom, Top, write_timestamp, read_phase_distribition

# Create a mesh
def mesh(Lx, Ly, res, **params):
    Nx = int(Lx/res)
    Ny = int(Ly/res)
    mesh = RectangleMesh(Point(-Lx/2, -Ly/2), Point(Lx/2, Ly/2), Nx, Ny)
    return mesh

# Override some problem specific parameters
def problem_parameters(NS_parameters, NS_expressions, commandline_kwargs, **NS_namespace):

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
        y0=-14,
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
    # Need to force this for proper PBCs
    if "Lx" in commandline_kwargs:
        NS_parameters["Lx"] = commandline_kwargs["Lx"]

    NS_expressions.update(dict(
        constrained_domain=PBC2(NS_parameters["Lx"])
    ))

def mark_subdomains(subdomains, Lx, Ly, **NS_namespace):
    top = Top(Ly)
    top.mark(subdomains, 1)
    btm = Bottom(Ly)
    btm.mark(subdomains, 2)
    return dict()

def contact_angles(theta, **NS_namespace):
    return [(theta, 1)]

# Specify boundary conditions
def create_bcs(V, Q, u0, W, subdomains, injected_phase, **NS_namespace):
    bc_ux_top = DirichletBC(V, 0, subdomains, 1)
    bc_uy_top = DirichletBC(V, u0, subdomains, 1)
    bc_ux_btm = DirichletBC(V, 0, subdomains, 2)
    bc_uy_btm = DirichletBC(V, u0, subdomains, 2)
    #bc_p_top = DirichletBC(Q, 0, subdomains, 1)
    #bc_p_btm = DirichletBC(Q, 0, subdomains, 2)
    #bc_phig_top = DirichletBC(W.sub(0), -1 * injected_phase, subdomains, 2)
    bc_phig_btm = DirichletBC(W.sub(0), 1 * injected_phase, subdomains, 2)
    return dict(u0=[bc_ux_btm, bc_ux_top],
                u1=[bc_uy_btm, bc_uy_top],
                p=[], # [bc_p_top],
                phig=[bc_phig_btm])


def average_pressure_gradient(F0, **NS_namespace):
    # average pressure gradient
    return Constant(tuple(F0))


def acceleration(g0, **NS_namespace):
    # (gravitational) acceleration
    return Constant(tuple(g0))


def initialize(q_, q_1, q_2, x_1, x_2, bcs, epsilon, VV, Ly, y0, R, initial_state, mesh, **NS_namespace):
    if initial_state is None:
        phi_init = interpolate(Expression(
            "tanh((sqrt(pow(x[0], 2)+pow(x[1], 2))-R)/(sqrt(2)*epsilon))",
            #"tanh((x[1]-0.25*Ly)/(sqrt(2)*epsilon))-tanh((x[1]+0.25*Ly)/(sqrt(2)*epsilon))+1",
            #"tanh((x[1]-y0)/(sqrt(2)*epsilon))",
            epsilon=epsilon, R=R, Ly=Ly, y0=y0, degree=2), VV['phig'].sub(0).collapse())
        assign(q_['phig'].sub(0), phi_init)
    else:
        info_blue("initializing from: " + initial_state)
        phi_init = read_phase_distribition(initial_state, mesh, q_)
        assign(q_['phig'].sub(0), phi_init)

    q_1['phig'].vector()[:] = q_['phig'].vector()
    q_2['phig'].vector()[:] = q_['phig'].vector()
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(tstep, t, q_, p_, mesh, u_, newfolder, AssignedVectorFunction, **NS_namespace):
    volume = assemble(Constant(1.) * dx(domain=mesh))
    statsfolder = path.join(newfolder, "Stats")
    timestampsfolder = path.join(newfolder, "Timestamps")

    # Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree)
    uv = AssignedVectorFunction(u_, name="u")

    if MPI.rank(MPI.comm_world) == 0 and not path.exists(statsfolder):
        makedirs(statsfolder)

    return dict(uv=uv, statsfolder=statsfolder, timestampsfolder=timestampsfolder, volume=volume)


def temporal_hook(q_, tstep, t, dx, u_, p_, phi_, rho_,
                  sigma, epsilon, volume, statsfolder, timestampsfolder,
                  plot_interval, stat_interval, timestamps_interval,
                  uv, mesh,
                  **NS_namespace):
    info_red("tstep = {}".format(tstep))
    if tstep % plot_interval == 0 and False:
        plot(u_, title='Velocity')
        plt.show()
        plot(phi_, title='Phase field')
        plt.show()
        #plot(p_, title='Pressure')
        #plot(q_['alfa'], title='alfa')
        #plot(q_['beta'], title='beta')
    if tstep % stat_interval == 0:
        u0m = assemble(q_['u0'] * dx) / volume
        u1m = assemble(q_['u1'] * dx) / volume
        phim = assemble(phi_ * dx) / volume
        E_kin = 0.5*assemble(rho_ * (u_[0]**2 + u_[1]**2) * dx) / volume
        E_int = 0.5 * sigma * epsilon * assemble((phi_.dx(0)**2 + phi_.dx(1)**2) * dx) / volume
        E_pot = 0.25 * sigma / epsilon * assemble((1-phi_**2)**2 * dx) / volume
        # Do not forget boundary term in E_int !
        if MPI.rank(MPI.comm_world) == 0:
            with open(statsfolder + "/tdata.dat", "a") as tfile:
                tfile.write("{:d} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n".format(
                    tstep, t, u0m, u1m, phim, E_kin, E_int, E_pot))

    return dict()

def theend_hook(u_, p_, testing, **NS_namespace):
    u_norm = norm(u_[0].vector())
    if MPI.rank(MPI.comm_world) == 0 and testing:
        print("Velocity norm = {0:2.6e}".format(u_norm))
