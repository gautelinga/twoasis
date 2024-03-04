__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..TPfracStep import *
import numpy as np
from os import makedirs, getcwd
import pickle
import h5py

comm = MPI.comm_world
rank = comm.Get_rank()
size = comm.Get_size()


def get_mesh_extrema(meshfile):
    if rank == 0:
        with h5py.File(meshfile, "r") as h5f:
            x = h5f["mesh/coordinates"][:]
        x_min = x.min(axis=0)
        x_max = x.max(axis=0)
    else:
        x_min = np.zeros(3)
        x_max = np.zeros(3)
    comm.Bcast(x_min, root=0)
    comm.Bcast(x_max, root=0)
    return x_min, x_max

class GenSubDomain(SubDomain):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max
        SubDomain.__init__(self)

class PBC2xy(GenSubDomain):
    def inside(self, x, on_boundary):
        return bool( (near(x[0], self.x_min[0]) or near(x[1], self.x_min[1]))
            and not (near(x[0], self.x_max[0]) or near(x[1], self.x_max[1]))
            and on_boundary)

    def map(self, x, y):
        L = self.x_max - self.x_min
        if near(x[0], self.x_max[0]) and near(x[1], self.x_max[1]):
            y[0] = x[0] - L[0]
            y[1] = x[1] - L[1]
            y[2] = x[2]
        elif near(x[0], self.x_max[0]):
            y[0] = x[0] - L[0]
            y[1] = x[1]
            y[2] = x[2]
        elif near(x[1], self.x_max[1]):
            y[0] = x[0]
            y[1] = x[1] - L[1]
            y[2] = x[2]
        else:
            y[0] = -1000
            y[1] = -1000
            y[2] = -1000

class Walls(SubDomain):
    def __init__(self, xyzr_):
        self.pos_ = xyzr_[:, :3]
        self.r_ = xyzr_[:, 3]
        SubDomain.__init__(self)

    def inside(self, x, on_bnd):
        if on_bnd:
            x_ = np.outer(x, np.ones(len(self.pos_))).T
            dr_ = np.linalg.norm(x_ - self.pos_, axis=1)
            return any(dr_ < self.r_ * (1 + 1e-1) ) #+ DOLFIN_EPS_LARGE)
        return False
    
class Boun(SubDomain):
    def inside(self, x, on_bnd):
        return on_bnd
    
class SideWalls(GenSubDomain):
    def inside(self, x, on_bnd):
        return on_bnd and (x[0] < self.x_min[0] + DOLFIN_EPS_LARGE or
                           x[1] < self.x_min[1] + DOLFIN_EPS_LARGE or
                           x[2] < self.x_min[2] + DOLFIN_EPS_LARGE or
                           x[0] > self.x_max[0] - DOLFIN_EPS_LARGE or
                           x[1] > self.x_max[1] - DOLFIN_EPS_LARGE or
                           x[2] > self.x_max[2] - DOLFIN_EPS_LARGE)

class Bottom(GenSubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[2], self.x_min[2])
    
class Top(GenSubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[2], self.x_max[2])

def inlet(x, on_bnd):
    return on_bnd and near(x[0], 0)

def get_fname(L, R, reps, res, ext):
    meshparams = dict(L=L, R=R, reps=reps, res=res, ext=ext)
    fname = "meshes/porous3d_rcp01.{ext}".format(**meshparams)
    return fname

def ccode_expr(checkerboard):
    expr_fhat = "0.5*(tanh(({xhat}+0.25)/(sqrt(2)*{eps})) - tanh(({xhat}-0.25)/(sqrt(2)*{eps})))"
    expr_xhat = "({N}*x[{d}]/L-({i}+0.5))"
    ccode = ""
    factors = []
    for d in range(len(checkerboard)):
        if checkerboard[d] > 0:
            expr_xhat_d = []
            expr_eps = "{N} / L * epsilon".format(N=checkerboard[d])
            for i in range(checkerboard[d]):
                expr_xhat_di = expr_xhat.format(N=checkerboard[d], d=d, i=i)
                expr_xhat_d.append(expr_xhat_di)
            factors.append("(2*(" + "+".join([expr_fhat.format(xhat=expr_xhat_di, eps=expr_eps) for expr_xhat_di in expr_xhat_d]) + ")-1.0)")
        else:
            factors.append("1")
    ccode = "*".join(factors)
    return ccode

# Create a mesh
def mesh(meshfile, **params):
    mesh = Mesh()
    with HDF5File(mesh.mpi_comm(), meshfile, "r") as h5f:
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
            L=1.0,
            meshfile="meshes/test_4x4x8_out.h5",
            R=0.5, # dummy
            reps=0.0, # dummy
            res=0.01, #dummy for now
            dt=0.1,
            rho=[1, 1],
            mu=[1, 1],
            theta=np.pi/3,
            epsilon=0.05,
            sigma=1.0,
            M=0.0001,
            u0=0.01,
            injected_phase=-1,
            F0=[0., 0., 0],
            g0=[0., 0., 0.],
            velocity_degree=1,
            folder="porous3dflux_results",
            stat_interval=10,
            timestamps_interval=10,
            save_step=10,
            checkpoint=10,
            print_intermediate_info=10,
            use_krylov_solvers=True,
            solver="BDF",  # solver="IPCS",
            bdf_order=1,
            AB_projection_pressure=False,
            max_iter=5,                 # Number of inner pressure velocity iterations on timestep
            max_error=1e-3,               # Tolerance for inner iterations (pressure velocity iterations)
            iters_on_first_timestep=10,  # Number of iterations on first timestep
        )

        #NS_parameters['krylov_solvers'] = {'monitor_convergence': False,
        #                                   'report': False,
        #                                   'relative_tolerance': 1e-10,
        #                                   'absolute_tolerance': 1e-10}

        if "meshfile" in commandline_kwargs:
            NS_parameters["meshfile"] = commandline_kwargs["meshfile"]

    x_min, x_max = get_mesh_extrema(NS_parameters["meshfile"])

    #print(rank, x_min, x_max)

    NS_expressions.update(dict(
        constrained_domain=PBC2xy(x_min, x_max),
        x_min=x_min,
        x_max=x_max
    ))

def mark_subdomains(subdomains, meshfile, x_min, x_max, L, R, reps, res, **NS_namespace):
    xyzr_ = np.loadtxt(meshfile[:-2] + "obst")

    #wall = Walls(xyzr_)
    #wall.mark(subdomains, 1)
    boun = Boun()
    boun.mark(subdomains, 1)

    swalls = SideWalls(x_min, x_max)
    swalls.mark(subdomains, 0)

    btm = Bottom(x_min, x_max)
    btm.mark(subdomains, 2)

    top = Top(x_min, x_max)
    top.mark(subdomains, 3)

    return dict()

def contact_angles(theta, **NS_namespace):
    return [(theta, 1)]

# Specify boundary conditions
def create_bcs(V, W, subdomains, u0, injected_phase, **NS_namespace):
    bc_u_wall = DirichletBC(V, 0, subdomains, 1)
    bc_u_btm = DirichletBC(V, 0, subdomains, 2)
    bc_uz_btm = DirichletBC(V, u0, subdomains, 2)
    bc_u_top = DirichletBC(V, 0, subdomains, 3)
    bc_uz_top = DirichletBC(V, u0, subdomains, 3)
    bc_phig_btm = DirichletBC(W.sub(0), 1 * injected_phase, subdomains, 2)

    return dict(u0=[bc_u_wall, bc_u_btm, bc_u_top],
                u1=[bc_u_wall, bc_u_btm, bc_u_top],
                u2=[bc_u_wall, bc_uz_btm, bc_uz_top],
                p=[],
                phig=[bc_phig_btm])


def average_pressure_gradient(F0, **NS_namespace):
    # average pressure gradient
    return Constant(tuple(F0))


def acceleration(g0, **NS_namespace):
    # (gravitational) acceleration
    return Constant(tuple(g0))


def initialize(q_, q_1, x_1, x_2, bcs, epsilon, VV, x_min, x_max, injected_phase, **NS_namespace):
    frac = 0.1
    z0 = frac*x_max[2]+(1-frac)*x_min[2]
    phig_init = interpolate(Expression(
        "injected_phase*tanh((x[2]-z0)/(sqrt(2)*epsilon))",
        epsilon=epsilon, z0=z0, injected_phase=injected_phase, degree=2), VV['phig'].sub(0).collapse())
    assign(q_['phig'].sub(0), phig_init)
    q_1['phig'].vector()[:] = q_['phig'].vector()
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(tstep, t, q_, p_, mesh, u_, newfolder, velocity_degree, pressure_degree, AssignedVectorFunction, 
                   F0, g0, mu, rho, sigma, M, theta, epsilon, R, reps, res, dt, constrained_domain, **NS_namespace):
    volume = assemble(Constant(1.) * dx(domain=mesh))
    statsfolder = path.join(newfolder, "Stats")
    timestampsfolder = path.join(newfolder, "Timestamps")
    keys = ["F0", "g0", "mu", "rho", "sigma", "M", "theta", "epsilon", "R", "reps", "res", "dt"]

    # Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree)
    uv = AssignedVectorFunction(u_, name="u")
    #phi__ = Function(q_['phig'].function_space().sub(0).collapse(), name="phi")
    phi__ = Function(FunctionSpace(mesh, "CG", velocity_degree, constrained_domain=constrained_domain), name="phi")

    if MPI.rank(MPI.comm_world) == 0 and not path.exists(statsfolder):
        makedirs(statsfolder)

    if MPI.rank(MPI.comm_world) == 0 and not path.exists(timestampsfolder):
        makedirs(timestampsfolder)

    if MPI.rank(MPI.comm_world) == 0:
        with open(path.join(timestampsfolder, "params.dat"), "a+") as ofile:
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
    write_timestamp(tstep, t, mesh, uv, q_, p_, phi__, timestampsfolder)

    return dict(uv=uv, statsfolder=statsfolder, timestampsfolder=timestampsfolder, volume=volume, phi__=phi__)

def write_timestamp(tstep, t, mesh, uv, q_, p_, phi__, timestampsfolder):
    uv()

    h5fname = "up_{}.h5".format(tstep)

    assign(phi__, q_['phig'].sub(0))

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
                  uv, mesh, phi__,
                  **NS_namespace):
    info_red("tstep = {}".format(tstep))
    if tstep % stat_interval == 0:
        sigma_bar = sigma * 3./(2*np.sqrt(2))
        u0m = assemble(q_['u0'] * dx) / volume
        u1m = assemble(q_['u1'] * dx) / volume
        u2m = assemble(q_['u2'] * dx) / volume
        phim = assemble(phi_ * dx) / volume
        E_kin = 0.5 * assemble(rho_ * (u_[0]**2 + u_[1]**2 + u_[2]**2) * dx) / volume
        E_int = 0.5 * sigma_bar * epsilon * assemble((phi_.dx(0)**2 + phi_.dx(1)**2 + phi_.dx(2)**2) * dx) / volume
        E_pot = 0.25 * sigma_bar / epsilon * assemble((1-phi_**2)**2 * dx) / volume
        
        # Do not forget boundary term in E_int !
        if MPI.rank(MPI.comm_world) == 0:
            with open(statsfolder + "/tdata.dat", "a") as tfile:
                tfile.write("{:d} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n".format(
                    tstep, t, u0m, u1m, u2m, phim, E_kin, E_int, E_pot))
    #if tstep % timestamps_interval == 0:
    #    write_timestamp(tstep, t, mesh, uv, q_, p_, phi__, timestampsfolder)
    return dict()

def theend_hook(u_, p_, testing, **NS_namespace):
    pass