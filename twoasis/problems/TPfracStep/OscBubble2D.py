__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..TPfracStep import *
import matplotlib.pyplot as plt
import numpy as np
from os import makedirs
from ufl_legacy import max_value, min_value

class LeftRight(SubDomain):
    def __init__(self, Lx):
        self.Lx = Lx
        super().__init__()

    def inside(self, x, on_bnd):
        return on_bnd and (x[0] > self.Lx - DOLFIN_EPS_LARGE or x[0] < DOLFIN_EPS_LARGE)

class TopBottom(SubDomain):
    def __init__(self, Ly):
        self.Ly = Ly
        super().__init__()

    def inside(self, x, on_bnd):
        return on_bnd and (x[1] > self.Ly - DOLFIN_EPS_LARGE or x[1] < DOLFIN_EPS_LARGE)

# Create a mesh
def mesh(Lx, Ly, res, **params):
    Nx = int(Lx/res)
    Ny = int(Ly/res)
    print(f"{Nx}, {Ny}")
    mesh = RectangleMesh(Point(0., 0.), Point(Lx, Ly), Nx, Ny)
    return mesh

# Override some problem specific parameters
def problem_parameters(NS_parameters, NS_expressions, commandline_kwargs, **NS_namespace):
    NS_parameters.update(
        T=3.0,
        Lx=5.0,
        Ly=5.0,
        res=1./16,
        amplitude=0.1,
        mode_order=3,
        R=1.25,
        dt=0.0005,
        rho=[0.998, 0.0012], #
        mu=[0.101, 0.00182],
        theta=np.pi/2,
        epsilon=0.06,
        sigma=73.0,
        M=1e-2,
        F0=[0., 0.],
        g0=[0., 0.],
        velocity_degree=1,
        folder="oscbubble2d_results",
        # plot_interval=10,
        stat_interval=0.002,
        timestamps_interval=0.02,
        #save_step=10,
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
    lr = LeftRight(Lx)
    tb = TopBottom(Ly)
    lr.mark(subdomains, 1)
    tb.mark(subdomains, 1)
    return dict()

def contact_angles(theta, **NS_namespace):
    return [(theta, 1)]

# Specify boundary conditions
def create_bcs(V, subdomains, **NS_namespace):
    bc_u = DirichletBC(V, 0, subdomains, 1)
    #bc_ux_lr = DirichletBC(V, 0, subdomains, 1)
    return dict(u0=[bc_u],
                u1=[bc_u],
                p=[],
                phig=[])


def average_pressure_gradient(F0, **NS_namespace):
    # average pressure gradient
    return Constant(tuple(F0))


def acceleration(g0, **NS_namespace):
    # (gravitational) acceleration
    return Constant(tuple(g0))


def initialize(q_, q_1, q_2, x_1, x_2, bcs, epsilon, VV, Lx, Ly, R, 
               amplitude, mode_order, 
               **NS_namespace):
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


def pre_solve_hook(tstep, t, q_, p_, mesh, u_, newfolder, velocity_degree, pressure_degree, AssignedVectorFunction, 
                   F0, g0, mu, rho, sigma, M, theta, epsilon, res, dt, 
                   save_interval, stat_interval, timestamps_interval,
                   **NS_namespace):
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
    save_step = round(save_interval / dt)
    stat_step = round(stat_interval / dt)
    timestamps_step = round(timestamps_interval / dt)

    return dict(uv=uv, statsfolder=statsfolder, timestampsfolder=timestampsfolder, volume=volume,
                save_step=save_step, stat_step=stat_step, timestamps_step=timestamps_step)

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

def temporal_hook(q_, tstep, t, dt, dx, u_, p_, phi_, rho_,
                  sigma, epsilon, volume, g0, statsfolder, timestampsfolder,
                  stat_step, timestamps_step,
                  uv, mesh,
                  **NS_namespace):
    info_red("tstep = {}".format(tstep))
    if tstep % stat_step == 0:
        u0m = assemble(q_['u0'] * dx) / volume
        u1m = assemble(q_['u1'] * dx) / volume
        phim = assemble(phi_ * dx) / volume
        X = SpatialCoordinate(mesh)

        phi_2 = max_value(min_value((-phi_ + 1)/2, 1.), 0.)
        ind_ = 3*phi_2**2 - 2*phi_2**3

        vol_c = assemble(ind_ * dx)
        y_cm = assemble(ind_ * X[1] * dx) / vol_c
        V_c = assemble(ind_ * q_['u1'] * dx) / vol_c

        E_kin = 0.5*assemble(rho_ * (u_[0]**2 + u_[1]**2) * dx) / volume
        E_grav = 0. # assemble(-rho_ * (g0[0] * X[0] + g0[1]*X[1]) * dx) / volume
        E_int = 0.5 * sigma * epsilon * assemble((phi_.dx(0)**2 + phi_.dx(1)**2) * dx) / volume
        E_pot = 0.25 * sigma / epsilon * assemble((1-phi_**2)**2 * dx) / volume
        # Do not forget boundary term in E_int !
        if MPI.rank(MPI.comm_world) == 0:
            with open(statsfolder + "/tdata.dat", "a") as tfile:
                entries = [u0m, u1m, phim, E_kin, E_int, E_pot, E_grav, vol_c, y_cm, V_c]
                tfile.write(
                    ("{:d} {:.8f} " + " ".join(["{:.8f}" for _ in entries]) + "\n").format(
                    tstep, t, *entries))
    """
    if tstep % timestamps_interval == 0 and False:
        write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)
    """
    return dict()

def theend_hook(u_, p_, **NS_namespace):
    pass
    #u_norm = norm(u_[0].vector())
    #if MPI.rank(MPI.comm_world) == 0 and testing:
    #    print("Velocity norm = {0:2.6e}".format(u_norm))
