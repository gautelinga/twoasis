__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2023-06-5"
__copyright__ = "Copyright (C) 2023 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..TPfracStep import *
import numpy as np
from os import makedirs, getcwd
from .Porous2D import Walls
import pickle

class PBC2(SubDomain):
    def __init__(self, Ly):
        self.Ly = Ly
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        # return True if on left boundary AND NOT on the two slave edge
        return bool(near(x[1], -self.Ly / 2) and not near(x[1], self.Ly / 2) and on_boundary)

    def map(self, x, y):
        if near(x[1], self.Ly / 2):
            y[0] = x[0]
            y[1] = x[1] - self.Ly
        else:  # near(x[2], Lz/2.):
            y[0] = x[0]
            y[1] = x[1] #- self.Ly

class Left(SubDomain):
    def __init__(self, Lx):
        self.Lx = Lx
        super().__init__()

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], -self.Lx/2)

class Right(SubDomain):
    def __init__(self, Lx):
        self.Lx = Lx
        super().__init__()

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], self.Lx/2)

def get_fname(Lx, Ly, rad, res, ext):
    meshparams = dict(Lx=Lx, Ly=Ly, rad=rad, res=res, ext=ext)
    fname = "meshes/lattice_Lx{Lx}_Ly{Ly}_r{rad}_dx{res}.{ext}".format(**meshparams)    
    return fname

# Create a mesh
def mesh(Lx, Ly, rad, res, **params):
    mesh = Mesh()
    #fname = "meshes/periodic_porous_Lx20_Ly10_rad0.25_N300_dx0.05.h5"
    fname = get_fname(Lx, Ly, rad, res, "h5")
    with HDF5File(mesh.mpi_comm(), fname, "r") as h5f:
        h5f.read(mesh, "mesh", False)
    return mesh

# Override some problem specific parameters
def problem_parameters(NS_parameters, NS_expressions, commandline_kwargs, **NS_namespace):
    if "restart_folder" in commandline_kwargs.keys():
         restart_folder = commandline_kwargs["restart_folder"]
         restart_folder = path.join(getcwd(), restart_folder)
         f = open(path.join(path.dirname(path.abspath(__file__)), restart_folder, 'params.dat'), 'rb')
         NS_parameters.update(pickle.load(f))
         NS_parameters['restart_folder'] = restart_folder
         globals().update(NS_parameters)
    else:
        NS_parameters.update(
            T=100.0,
            Lx=4.0,
            Ly=2.0,
            rad=0.5,
            N=88,
            res=0.05,
            dt=0.02,
            rho=[1, 1],
            mu=[1, 1],
            Dp0=-5.0,
            theta=np.pi/2,
            epsilon=0.05,
            sigma=5.0,
            M=0.0001,
            F0=[0., 0.],
            g0=[0., 0.],
            velocity_degree=1,
            folder="simplepore2d_results",
            plot_interval=10,
            stat_interval=10,
            timestamps_interval=10,
            save_step=20,
            checkpoint=10000,
            print_intermediate_info=10,
            use_krylov_solvers=True,
            solver="BDF",
            bdf_order=1,
            AB_projection_pressure=False,
            max_iter=10,                  # Number of inner pressure velocity iterations on timestep
            max_error=1e-3,               # Tolerance for inner iterations (pressure velocity iterations)
            iters_on_first_timestep=20,  # Number of iterations on first timestep
            injected_phase=-1,
        )
        # Need to force this for proper PBCs
        if "Lx" in commandline_kwargs:
            NS_parameters["Lx"] = commandline_kwargs["Lx"]

    #scalar_components += ["alfa", "beta"]
    #Schmidt["alfa"] = 1.
    #Schmidt["beta"] = 10.

    #NS_parameters['krylov_solvers'] = {'monitor_convergence': False,
    #                                   'report': False,
    #                                   'relative_tolerance': 1e-10,
    #                                   'absolute_tolerance': 1e-10}
    NS_expressions.update(dict(
        constrained_domain=PBC2(NS_parameters["Ly"])
    ))

def mark_subdomains(subdomains, dim, Lx, Ly, rad, res, **NS_namespace):
    fname = get_fname(Lx, Ly, rad, res, "dat")
    obst = np.loadtxt(fname)
    x = obst[:, :dim]
    r = obst[:, dim]
    wall = Walls(x, r)
    wall.mark(subdomains, 1)
    left = Left(Lx)
    left.mark(subdomains, 2)
    right = Right(Lx)
    right.mark(subdomains, 3)
    return dict()

def contact_angles(theta, **NS_namespace):
    return [(theta, 1)]

# Specify boundary conditions
def create_bcs(V, Q, W, subdomains, injected_phase, **NS_namespace):
    bc_ux_wall = DirichletBC(V, 0, subdomains, 1)
    bc_uy_wall = DirichletBC(V, 0, subdomains, 1)
    #bc_ux_top = DirichletBC(V, 0, subdomains, 2)
    #bc_uy_top = DirichletBC(V, u0, subdomains, 2)
    #bc_ux_btm = DirichletBC(V, 0, subdomains, 3)
    #bc_uy_btm = DirichletBC(V, u0, subdomains, 3)
    bc_p_left = DirichletBC(Q, 0, subdomains, 2)
    bc_p_right = DirichletBC(Q, 0, subdomains, 3)
    bc_phig_left = DirichletBC(W.sub(0), -1 * injected_phase, subdomains, 2)
    bc_phig_right = DirichletBC(W.sub(0), 1 * injected_phase, subdomains, 3)
    return dict(u0=[bc_ux_wall],
                u1=[bc_uy_wall],
                p=[bc_p_left, bc_p_right],
                phig=[bc_phig_left, bc_phig_right])


def average_pressure_gradient(F0, **NS_namespace):
    # average pressure gradient
    return Constant(tuple(F0))


def acceleration(g0, **NS_namespace):
    # (gravitational) acceleration
    return Constant(tuple(g0))


def initialize(q_, q_1, q_2, x_1, x_2, bcs, epsilon, VV, Ly, Dp0, sigma, rad, restart_folder, mesh, **NS_namespace):
    if restart_folder is None:
        R = abs(sigma/Dp0)
        eta = np.sign(Dp0)*np.sqrt(R**2 + rad**2 - (Ly/2)**2)

        phi_init = interpolate(Expression(
            #"tanh((sqrt(pow(x[0], 2)+pow(x[1], 2))-0.45)/(sqrt(2)*epsilon))",
            #"tanh((x[1]-0.25*Ly)/(sqrt(2)*epsilon))-tanh((x[1]+0.25*Ly)/(sqrt(2)*epsilon))+1",
            #"-tanh((x[0]-x0)/(sqrt(2)*epsilon))",
            "eta * x[0] > 0 ? tanh((sqrt(pow(x[0]+eta, 2) + pow(x[1], 2))-R)/(sqrt(2)*epsilon)) : -1",
            epsilon=epsilon, eta=eta, R=R, degree=2), VV['phig'].sub(0).collapse())
        assign(q_['phig'].sub(0), phi_init)

        q_1['phig'].vector()[:] = q_['phig'].vector()
        q_2['phig'].vector()[:] = q_['phig'].vector()
        for ui in x_1:
            [bc.apply(x_1[ui]) for bc in bcs[ui]]
        for ui in x_2:
            [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(tstep, t, q_, p_, mesh, u_, newfolder, velocity_degree, pressure_degree, AssignedVectorFunction, 
                   F0, g0, mu, rho, sigma, M, theta, epsilon, rad, res, dt, Lx, Ly, **NS_namespace):
    volume = assemble(Constant(1.) * dx(domain=mesh))
    statsfolder = path.join(newfolder, "Stats")
    timestampsfolder = path.join(newfolder, "Timestamps")

    # Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree)
    uv = AssignedVectorFunction(u_, name="u")

    if MPI.rank(MPI.comm_world) == 0 and not path.exists(statsfolder):
        makedirs(statsfolder)

    if MPI.rank(MPI.comm_world) == 0 and not path.exists(timestampsfolder):
        makedirs(timestampsfolder)

    if MPI.rank(MPI.comm_world) == 0:
        with open(path.join(timestampsfolder, "params.dat"), "a+") as ofile:
            keys = ["F0", "g0", "mu", "rho", "sigma", "M", "theta", "epsilon", "rad", "res", "dt", "Lx", "Ly"]
            for key in keys:
                ofile.write("{}={}\n".format(key, eval(key)))
        with open(path.join(timestampsfolder, "dolfin_params.dat"), "a+") as ofile:
            ofile.write("velocity_space=P{}\n".format(velocity_degree))
            ofile.write("pressure_space=P{}\n".format(pressure_degree))
            ofile.write("phase_field_space=P{}\n".format(velocity_degree))
            ofile.write("timestamps=timestamps.dat\n")
            ofile.write("mesh=mesh.h5\n")
            ofile.write("periodic_x=true\n")
            ofile.write("periodic_y=false\n")
            ofile.write("periodic_z=false\n")
            ofile.write("rho=1.0\n") # ? 
    with HDF5File(mesh.mpi_comm(),
                  path.join(timestampsfolder, "mesh.h5"), "w") as h5f:
        h5f.write(mesh, "mesh")
    write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)

    return dict(uv=uv, statsfolder=statsfolder, timestampsfolder=timestampsfolder, volume=volume)

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


def temporal_hook(q_, tstep, t, dx, u_, p_, phi_, rho_,
                  sigma, epsilon, volume, statsfolder, timestampsfolder,
                  stat_interval, timestamps_interval,
                  uv, mesh, **NS_namespace):
    info_red("tstep = {}".format(tstep))
    if tstep % stat_interval == 0:
        sigma_bar = sigma * 3./(2*np.sqrt(2))
        u0m = assemble(q_['u0'] * dx) / volume
        u1m = assemble(q_['u1'] * dx) / volume
        phim = assemble(phi_ * dx) / volume
        E_kin = 0.5*assemble(rho_ * (u_[0]**2 + u_[1]**2) * dx) / volume
        E_int = 0.5 * sigma_bar * epsilon * assemble((phi_.dx(0)**2 + phi_.dx(1)**2) * dx) / volume
        E_pot = 0.25 * sigma_bar / epsilon * assemble((1-phi_**2)**2 * dx) / volume
        # Do not forget boundary term in E_int !
        if MPI.rank(MPI.comm_world) == 0:
            with open(statsfolder + "/tdata.dat", "a") as tfile:
                tfile.write("{:d} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n".format(
                    tstep, t, u0m, u1m, phim, E_kin, E_int, E_pot))
    if tstep % timestamps_interval == 0:
        write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)


def theend_hook(u_, p_, testing, **NS_namespace):
    u_norm = norm(u_[0].vector())
    if MPI.rank(MPI.comm_world) == 0 and testing:
        print("Velocity norm = {0:2.6e}".format(u_norm))
