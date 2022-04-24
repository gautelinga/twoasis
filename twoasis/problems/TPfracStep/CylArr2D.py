__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..TPfracStep import *
import matplotlib.pyplot as plt
import numpy as np
from os import makedirs

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
    def __init__(self, Lx, Ly, rad):
        self.Lx = Lx
        self.Ly = Ly
        self.rad = rad
        SubDomain.__init__(self)

    def inside(self, x, on_bnd):
        r_ = [
            np.sqrt(x[0]**2 + x[1]**2),
            np.sqrt((x[0] - self.Lx / 2)**2 + (x[1] - self.Ly / 2)**2),
            np.sqrt((x[0] - self.Lx / 2)**2 + (x[1] + self.Ly / 2)**2),
            np.sqrt((x[0] + self.Lx / 2)**2 + (x[1] - self.Ly / 2)**2),
            np.sqrt((x[0] + self.Lx / 2)**2 + (x[1] + self.Ly / 2)**2)
        ]
        return any([r < self.rad + DOLFIN_EPS_LARGE for r in r_])

def inlet(x, on_bnd):
    return on_bnd and near(x[0], 0)


# Create a mesh
def mesh(Lx, Ly, rad, res, **params):
    mesh = Mesh()
    fname = "meshes/cyl_arr_Lx{}_Ly{}_rad{}_dx{}.h5".format(Lx, Ly, rad, res)
    with HDF5File(mesh.mpi_comm(), fname, "r") as h5f:
        h5f.read(mesh, "mesh", False)
    return mesh

# Override some problem specific parameters
def problem_parameters(NS_parameters, NS_expressions, commandline_kwargs, **NS_namespace):
    NS_parameters.update(
        T=1000.0,
        Lx=1.,
        Ly=np.sqrt(3),
        rad=0.33,
        res=0.02,
        dt=0.01,
        rho=[1, 1],
        mu=[1, 1],
        epsilon=0.02,
        sigma=1.0,
        theta=np.pi/4,
        M=0.00001,
        F0=[0., 10.],
        g0=[0., 0.],
        velocity_degree=2,
        folder="cylarr2d_results",
        plot_interval=10,
        stat_interval=10,
        timestamps_interval=10,
        save_step=10,
        checkpoint=10,
        print_intermediate_info=10,
        use_krylov_solvers=True,
        solver="IPCS")

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

def mark_subdomains(subdomains, Lx, Ly, rad, **NS_namespace):
    wall = Walls(Lx, Ly, rad)
    wall.mark(subdomains, 1)
    return dict()

def contact_angles(theta, **NS_namespace):
    return [(theta, 1)]


# Specify boundary conditions
def create_bcs(V, subdomains, **NS_namespace):
    bc_ux_wall = DirichletBC(V, 0, subdomains, 1)
    bc_uy_wall = DirichletBC(V, 0, subdomains, 1)
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


def initialize(q_, q_1, x_1, x_2, bcs, epsilon, VV, Lx, Ly, **NS_namespace):
    phig_init = interpolate(Expression(
        #"tanh((sqrt(pow(x[0], 2)+pow(x[1], 2))-0.45)/(sqrt(2)*epsilon))",
        #"tanh((x[1]-0.25*Ly)/(sqrt(2)*epsilon))-tanh((x[1]+0.25*Ly)/(sqrt(2)*epsilon))+1",
        #"tanh((sqrt(pow(x[0]-x0, 2)+pow(x[1]-y0, 2))-r0)/(sqrt(2)*epsilon))",
        "tanh((x[0]+(alpha-0.25)*Lx)/(sqrt(2)*epsilon))-tanh((x[0]+(alpha+0.25)*Lx)/(sqrt(2)*epsilon))+1",
        epsilon=epsilon, Lx=Lx, Ly=Ly, x0=0.02, y0=0.6, r0=0.2, alpha=0.2,
        degree=2), VV['phig'].sub(0).collapse())

    # Shake things up
    phig_init.vector()[:] = 1e-3*(np.random.rand(len(phig_init.vector()[:]))-0.5)

    assign(q_['phig'].sub(0), phig_init)
    q_1['phig'].vector()[:] = q_['phig'].vector()
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(mesh, u_, newfolder, velocity_degree, pressure_degree, AssignedVectorFunction, 
                   F0, g0, mu, rho, sigma, M, theta, epsilon, rad, res, dt, **NS_namespace):
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
            keys = ["F0", "g0", "mu", "rho", "sigma", "M", "theta", "epsilon", "rad", "res", "dt"]
            for key in keys:
                ofile.write("{}={}\n".format(key, eval(key)))
        with open(path.join(timestampsfolder, "dolfin_params.dat"), "a+") as ofile:
            ofile.write("velocity_space=P{}\n".format(velocity_degree))
            ofile.write("pressure_space=P{}\n".format(pressure_degree))
            ofile.write("timestamps=timestamps.dat\n")
            ofile.write("mesh=mesh.h5\n")
            ofile.write("periodic_x=true\n")
            ofile.write("periodic_y=true\n")
            ofile.write("periodic_z=true\n")
            ofile.write("rho=1.0\n")
    with HDF5File(mesh.mpi_comm(),
                  path.join(timestampsfolder, "mesh.h5"), "w") as h5f:
        h5f.write(mesh, "mesh")

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
                tfile.write("%d %.8f %.8f %.8f %.8f %.8f %.8f %.8f\n" % (tstep, t, u0m, u1m, phim, E_kin, E_int, E_pot))
    if tstep % timestamps_interval == 0:
        uv()

        h5fname = "up_{}.h5".format(tstep)

        with HDF5File(mesh.mpi_comm(), path.join(timestampsfolder, h5fname), "w") as h5f:
            h5f.write(uv, "u")
            h5f.write(p_, "p")

        if MPI.rank(MPI.comm_world) == 0:
            with open(path.join(timestampsfolder, "timestamps.dat"), "a+") as ofile:
                ofile.write("{:.6f} {}\n".format(t, h5fname))


def theend_hook(u_, p_, testing, **NS_namespace):
    if not testing:
        plot(u_, title='Velocity')
        plot(p_, title='Pressure')

    u_norm = norm(u_[0].vector())
    if MPI.rank(MPI.comm_world) == 0 and testing:
        print("Velocity norm = {0:2.6e}".format(u_norm))
