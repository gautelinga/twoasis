__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from copy import deepcopy
from ..TPfracStep import *
from twoasis.common.utilities import compute_ind
import matplotlib.pyplot as plt
import numpy as np
from os import makedirs
# from ufl_legacy import max_value, min_value

class LeftRight(SubDomain):
    def __init__(self, Lx):
        self.Lx = Lx
        super().__init__()

    def inside(self, x, on_bnd):
        return on_bnd and ( x[0] < DOLFIN_EPS_LARGE or x[0] > self.Lx - DOLFIN_EPS_LARGE )

class TopBottom(SubDomain):
    def __init__(self, Ly):
        self.Ly = Ly
        super().__init__()

    def inside(self, x, on_bnd):
        return on_bnd and x[1] < DOLFIN_EPS_LARGE

    def map(self, x, y):
        if near(x[1], self.Ly):
            y[0] = x[0]
            y[1] = x[1] - self.Ly
        else:  # near(x[2], Lz/2.):
            y[0] = x[0]
            y[1] = x[1]

# Create a mesh
def mesh(Lx, Ly, res, **params):
    Nx = int(Lx/res)
    Ny = int(Ly/res)
    mesh = RectangleMesh(Point(0., 0.), Point(Lx, Ly), Nx, Ny)
    return mesh

# Override some problem specific parameters
def problem_parameters(NS_parameters, NS_expressions, commandline_kwargs, **NS_namespace):
    NS_parameters.update(
        T=100.0,
        Lx=1,
        Ly=2,
        res=1./64,
        R=0.4,
        dt=0.02,
        rho=[1000., 100.],
        mu=[10., 1.],
        theta=np.pi/2,
        epsilon=0.02,
        sigma=24.5,
        M=0.00002,
        u_wall=-0.5,
        u_target=0.5,
        F0=[0., 0.],
        g0=[0., -0*0.98],
        velocity_degree=1,
        folder="taylorbubble2d_results",
        plot_interval=10,
        stat_interval=10,
        timestamps_interval=10,
        save_step=10,
        checkpoint=10,
        print_intermediate_info=10,
        use_krylov_solvers=True,
        solver="IPCS",
        max_iter=100,                 # Number of inner pressure velocity iterations on timestep
        max_error=1e-3,               # Tolerance for inner iterations (pressure velocity iterations)
        iters_on_first_timestep=200,  # Number of iterations on first timestep
    )

    NS_expressions.update(dict(
        constrained_domain=TopBottom(NS_parameters["Ly"])
    ))

def mark_subdomains(subdomains, Lx, Ly, **NS_namespace):
    lr = LeftRight(Lx)
    #tb = TopBottom(Ly)
    lr.mark(subdomains, 1)
    #tb.mark(subdomains, 2)
    return dict()

def contact_angles(theta, **NS_namespace):
    return [(theta, 1)]

# Specify boundary conditions
def create_bcs(V, subdomains, u_wall, **NS_namespace):
    #bc_u_tb = DirichletBC(V, 0, subdomains, 2)
    bc_ux_lr = DirichletBC(V, 0, subdomains, 1)
    bc_uy_lr = DirichletBC(V, u_wall, subdomains, 1)
    return dict(u0=[bc_ux_lr],
                u1=[bc_uy_lr],
                p=[],
                phig=[])


def average_pressure_gradient(g0, F0, q_, rho, dx, mesh, **NS_namespace):
    # average pressure gradient
    #return Constant(tuple(F0))
    return Expression(("0", "-forcing"), forcing=F0[1], degree=2)
    # rho__, g___ = q_['phig'].split(deepcopy=True)
    # cv = rho__.vector()[:]
    # cv += 1
    # cv *= 0.5
    # cv[cv < 0.0] = 0.0
    # cv[cv > 1.0] = 1.0
    # rho__.vector()[:] = rho[0]*cv + rho[1]*(1-cv)
    # rho_avg = assemble(rho__ * dx)/assemble(Constant(1)*dx(domain=mesh))
    # return Constant((rho[0]*g0[0], rho[0]*g0[1]))

def acceleration(g0, **NS_namespace):
    # (gravitational) acceleration
    return Constant(tuple(g0))
    #return Expression(("0", "grav"), grav=g0[1], degree=2)

def initialize(q_, q_1, x_1, x_2, bcs, epsilon, VV, Lx, Ly, R, u_wall, **NS_namespace):
    phig_init = interpolate(Expression(
        "tanh((sqrt(pow(x[0]-Lx/2, 2)+pow(x[1]-Ly/2, 2))-R)/(sqrt(2)*epsilon))",
        epsilon=epsilon, Lx=Lx, Ly=Ly, R=R, degree=2), VV['phig'].sub(0).collapse())
    assign(q_['phig'].sub(0), phig_init)
    q_1['phig'].vector()[:] = q_['phig'].vector()
    q_['u1'].vector()[:] = u_wall
    q_1['u1'].vector()[:] = u_wall
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(tstep, t, q_, p_, mesh, u_, newfolder, velocity_degree, 
                   pressure_degree, AssignedVectorFunction, 
                   F0, g0, mu, rho, sigma, M, theta, epsilon, res, dt, **NS_namespace):
    volume = assemble(Constant(1.) * dx(domain=mesh))
    statsfolder = path.join(newfolder, "Stats")
    timestampsfolder = path.join(newfolder, "Timestamps")

    # Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree)
    uv = AssignedVectorFunction(u_, name="u")

    if MPI.rank(MPI.comm_world) == 0 and not path.exists(statsfolder):
        makedirs(statsfolder)

    """
    if MPI.rank(MPI.comm_world) == 0 and not path.exists(timestampsfolder):
        makedirs(timestampsfolder)
    if MPI.rank(MPI.comm_world) == 0:
        with open(path.join(timestampsfolder, "params.dat"), "a+") as ofile:
            keys = ["F0", "g0", "mu", "rho", "sigma", "M", "theta", "epsilon", "res", "dt"]
            for key in keys:
                ofile.write("{}={}\n".format(key, eval(key)))
        with open(path.join(timestampsfolder, "dolfin_params.dat"), "a+") as ofile:
            ofile.write("velocity_space=P{}\n".format(velocity_degree))
            ofile.write("pressure_space=P{}\n".format(pressure_degree))
            ofile.write("phase_field_space=P{}\n".format(velocity_degree))
            ofile.write("timestamps=timestamps.dat\n")
            ofile.write("mesh=mesh.h5\n")
            ofile.write("periodic_x=true\n")
            ofile.write("periodic_y=true\n")
            ofile.write("periodic_z=true\n")
            ofile.write("rho=1.0\n")
    with HDF5File(mesh.mpi_comm(),
                  path.join(timestampsfolder, "mesh.h5"), "w") as h5f:
        h5f.write(mesh, "mesh")
    write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)
    """

    return dict(uv=uv, statsfolder=statsfolder, timestampsfolder=timestampsfolder, volume=volume, V_c_prev=0)

"""
def write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder):
    uv()

    h5fname = "up_{}.h5".format(tstep)

    phi__, g__ = q_['phig'].split(deepcopy=True)
    phi__.rename("phi", "phi")

    with HDF5File(mesh.mpi_comm(), path.join(timestampsfolder, h5fname), "w") as h5f:
        h5f.write(uv, "u")
        h5f.write(p_, "p")
        h5f.write(phi__, "phi")

    if MPI.rank(MPI.comm_world) == 0:
        with open(path.join(timestampsfolder, "timestamps.dat"), "a+") as ofile:
            ofile.write("{:.6f} {}\n".format(t, h5fname))
"""

def temporal_hook(q_, tstep, t, dx, u_, p_, phi_, rho_,
                  sigma, epsilon, volume, g0, statsfolder, timestampsfolder,
                  plot_interval, stat_interval, timestamps_interval,
                  uv, mesh, acc, dt, gradp_avg, u_wall, u_target, Lx, Ly, V_c_prev, rho, mu,
                  **NS_namespace):
    returndict = dict()

    info_red("tstep = {}".format(tstep))
    if tstep % stat_interval == 0:
        u0m = assemble(q_['u0'] * dx) / volume
        u1m = assemble(q_['u1'] * dx) / volume
        phim = assemble(phi_ * dx) / volume
        X = SpatialCoordinate(mesh)

        ind_ = compute_ind(q_)

        vol_c = assemble(ind_ * dx)
        y_cm = assemble(ind_ * X[1] * dx) / vol_c
        V_c = assemble(ind_ * q_['u1'] * dx) / vol_c
        V_tot = assemble(q_['u1'] * dx) / (Lx*Ly)

        E_kin = 0.5*assemble(rho_ * (u_[0]**2 + u_[1]**2) * dx) / volume
        E_grav = assemble(-rho_ * (g0[0] * X[0] + g0[1]*X[1]) * dx(domain=mesh)) / volume
        E_int = 0.5 * sigma * epsilon * assemble((phi_.dx(0)**2 + phi_.dx(1)**2) * dx) / volume
        E_pot = 0.25 * sigma / epsilon * assemble((1-phi_**2)**2 * dx) / volume
        # Do not forget boundary term in E_int !

        # both are adaptive
        Kp = 1.0

        #grav_prev = acc.grav
        #grav = grav_prev - Kp * V_c * stat_interval * dt 
        #acc.grav = grav
        grav=0

        k = 10.
        tau = Lx**2 / (12 * mu[0] / rho[0])
        alpha = rho[0] * 0.25 * (1+k)**2 / tau**2
        beta = rho[0] * k / tau

        forcing_prev = gradp_avg.forcing
        forcing = forcing_prev + alpha * (u_target + u_wall - V_c) * stat_interval * dt - beta * (V_c - V_c_prev)
        #forcing = forcing_prev - Kp * V_c * stat_interval * dt
        #forcing = 0
        gradp_avg.forcing = forcing

        if MPI.rank(MPI.comm_world) == 0:
            with open(statsfolder + "/tdata.dat", "a") as tfile:
                entries = [u0m, u1m, phim, E_kin, E_int, E_pot, E_grav, vol_c, y_cm, V_c, V_tot, grav, forcing]
                tfile.write(
                    ("{:d} {:.8f} " + " ".join(["{:.8f}" for _ in entries]) + "\n").format(
                    tstep, t, *entries))

        returndict.update(dict(V_c_prev=V_c))

    """
    if tstep % timestamps_interval == 0 and False:
        write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)
    """
    return returndict


def theend_hook(u_, p_, **NS_namespace):
    pass
    #u_norm = norm(u_[0].vector())
    #if MPI.rank(MPI.comm_world) == 0 and testing:
    #    print("Velocity norm = {0:2.6e}".format(u_norm))
