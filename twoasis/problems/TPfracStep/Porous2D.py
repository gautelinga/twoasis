__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..TPfracStep import *
from ..DrivenCavity import *
import matplotlib.pyplot as plt
import numpy as np

class PBC(SubDomain):
    def __init__(self, Lx, Ly):
        self.Lx = Lx
        self.Ly = Ly
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two slave edges
        return bool(
            (near(x[0], -self.Lx / 2) or near(x[1], -self.Ly / 2))
            and (not (near(x[0], self.Lx / 2) or near(x[1], self.Ly / 2)))
            and on_boundary)

    def map(self, x, y):
        if near(x[0], self.Lx / 2) and near(x[1], self.Ly / 2):
            y[0] = x[0] - self.Lx
            y[1] = x[1] - self.Ly
        elif near(x[0], self.Lx / 2):
            y[0] = x[0] - self.Lx
            y[1] = x[1]
        else:  # near(x[2], Lz/2.):
            y[0] = x[0]
            y[1] = x[1] - self.Ly

class Walls(SubDomain):
    def __init__(self, pos_, r_):
        self.pos_ = pos_
        self.r_ = r_
        SubDomain.__init__(self)

    def inside(self, x, on_bnd):
        x_ = np.outer(x, np.ones(len(self.pos_))).T
        dr_ = np.linalg.norm(x_ - self.pos_, axis=1)
        return any(dr_ < self.r_ + DOLFIN_EPS_LARGE)

def inlet(x, on_bnd):
    return on_bnd and near(x[0], 0)

def get_fname(Lx, Ly, rad, R, N, res, ext):
    meshparams = dict(Lx=Lx, Ly=Ly, rad=rad, R=R, N=N, res=res, ext=ext)
    fname = "meshes/periodic_porous_Lx{Lx}_Ly{Ly}_r{rad}_R{R}_N{N}_dx{res}.{ext}".format(**meshparams)    
    return fname

# Create a mesh
def mesh(Lx, Ly, rad, R, N, res, **params):
    mesh = Mesh()
    #fname = "meshes/periodic_porous_Lx20_Ly10_rad0.25_N300_dx0.05.h5"
    fname = get_fname(Lx, Ly, rad, R, N, res, "h5")
    with HDF5File(mesh.mpi_comm(), fname, "r") as h5f:
        h5f.read(mesh, "mesh", False)
    return mesh

# Override some problem specific parameters
def problem_parameters(NS_parameters, NS_expressions, commandline_kwargs, **NS_namespace):
    NS_parameters.update(
        T=100.0,
        Lx=4,
        Ly=8,
        rad=0.5,
        N=19,
        res=0.05,
        R=0.55,
        dt=0.05,
        rho=[1, 1],
        mu=[1, 1],
        epsilon=0.05,
        sigma=5.0,
        M=0.0001,
        F0=[0., 10.],
        g0=[0., 0.],
        velocity_degree=1,
        folder="porous2d_results",
        plot_interval=10,
        save_step=10,
        checkpoint=10,
        print_intermediate_info=10,
        use_krylov_solvers=True)

    #scalar_components += ["alfa", "beta"]
    #Schmidt["alfa"] = 1.
    #Schmidt["beta"] = 10.

    #NS_parameters['krylov_solvers'] = {'monitor_convergence': False,
    #                                   'report': False,
    #                                   'relative_tolerance': 1e-10,
    #                                   'absolute_tolerance': 1e-10}
    NS_expressions.update(dict(
        constrained_domain=PBC(NS_parameters["Lx"], NS_parameters["Ly"])
    ))

# Specify boundary conditions
def create_bcs(V, dim, Lx, Ly, rad, R, N, res, **NS_namespace):
    fname = get_fname(Lx, Ly, rad, R, N, res, "dat")
    obst = np.loadtxt(fname)
    x = obst[:, :dim]
    r = obst[:, dim]
    wall = Walls(x, r)
    bc_ux_wall = DirichletBC(V, 0, wall)
    bc_uy_wall = DirichletBC(V, 0, wall)
    return dict(u0=[bc_ux_wall, bc_ux_wall],
                u1=[bc_uy_wall, bc_uy_wall],
                p=[],
                phig=[])


def average_pressure_gradient(F0, **NS_namespace):
    # average pressure gradient
    return Constant(tuple(F0))


def acceleration(g0, **NS_namespace):
    # (gravitational) acceleration
    return Constant(tuple(g0))


def initialize(q_, q_1, x_1, x_2, bcs, epsilon, VV, Ly, **NS_namespace):
    phig_init = interpolate(Expression(
        #"tanh((sqrt(pow(x[0], 2)+pow(x[1], 2))-0.45)/(sqrt(2)*epsilon))",
        "tanh((x[1]-0.25*Ly)/(sqrt(2)*epsilon))-tanh((x[1]+0.25*Ly)/(sqrt(2)*epsilon))+1",
        epsilon=epsilon, Ly=Ly, degree=2), VV['phig'].sub(0).collapse())
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
