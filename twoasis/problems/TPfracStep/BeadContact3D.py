__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from re import sub
from ..TPfracStep import *
import matplotlib.pyplot as plt
import numpy as np
from os import makedirs

class Bead(SubDomain):
    def __init__(self, R):
        self.R = R
        SubDomain.__init__(self)
    
    def inside(self, x, on_bnd):
        r1 = np.linalg.norm(x)
        return on_bnd and r1 < self.R + DOLFIN_EPS_LARGE

class EpsCyl(SubDomain):
    def __init__(self, reps):
        self.reps = reps
        SubDomain.__init__(self)

    def inside(self, x, on_bnd):
        s1 = np.linalg.norm([x[0], x[1]])
        return on_bnd and s1 < self.reps + DOLFIN_EPS_LARGE

class Walls(SubDomain):
    def __init__(self, H):
        self.H = H
        SubDomain.__init__(self)

    def inside(self, x, on_bnd):
        return on_bnd and (x[2] > self.H/2 - DOLFIN_EPS_LARGE or x[2] < - self.H/2 + DOLFIN_EPS_LARGE)

class PeriodicDomain(SubDomain):
    def __init__(self, Lx, Ly):
        self.Lx = Lx
        self.Ly = Ly
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two slave edges
        return bool((near(x[0], -self.Lx / 2.) or near(x[1], -self.Ly / 2.)) and
                    (not (near(x[0], self.Lx / 2.) or near(x[1], self.Ly / 2.))) and on_boundary)

    def map(self, x, y):
        if near(x[0], self.Lx / 2.) and near(x[1], self.Ly / 2.):
            y[0] = x[0] - self.Lx
            y[1] = x[1] - self.Ly
            y[2] = x[2]
        elif near(x[0], self.Lx / 2.):
            y[0] = x[0] - self.Lx
            y[1] = x[1]
            y[2] = x[2]
        else:  # near(x[2], Lz/2.):
            y[0] = x[0]
            y[1] = x[1] - self.Ly
            y[2] = x[2]

def get_fname(Lx, Ly, H, R, res, ext):
    meshparams = dict(Lx=Lx, Ly=Ly, H=H, R=R, res=res, ext=ext)
    fname = "meshes/beads_Lx{Lx}_Ly{Ly}_H{H}_R{R}_res{res}.{ext}".format(**meshparams)    
    return fname

# Create a mesh
def mesh(Lx, Ly, H, R, res, **params):
    mesh = Mesh()
    fname = get_fname(Lx, Ly, H, R, res, "h5")
    with HDF5File(mesh.mpi_comm(), fname, "r") as h5f:
        h5f.read(mesh, "mesh", False)
    return mesh

# Override some problem specific parameters
def problem_parameters(NS_parameters, NS_expressions, commandline_kwargs, **NS_namespace):
    NS_parameters.update(
        T=100.0,
        Lx=1.5,
        Ly=2.0,
        H=1.0,
        R=0.5,
        r=0.1,
        reps=0.2,
        res=24,
        dt=0.001,
        rho=[1, 1],
        mu=[1, 1],
        theta_bead=np.pi/3,
        theta_wall=np.pi/3,
        epsilon=0.05,
        sigma=1.0,
        M=0.0001,
        g0=[0., 0., 0.],
        velocity_degree=1,
        folder="beadcontact_results",
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
        constrained_domain=PeriodicDomain(NS_parameters["Lx"], NS_parameters["Ly"])
    ))

def mark_subdomains(subdomains, Lx, Ly, H, R, reps, **NS_namespace):
    epscyl = EpsCyl(reps)
    bead = Bead(R)
    wall = Walls(H)
    epscyl.mark(subdomains, 2)
    bead.mark(subdomains, 1)
    wall.mark(subdomains, 3)

def contact_angles(theta_bead, theta_wall, **NS_namespace):
    return [(theta_bead, 1), (theta_wall, 3)]

# Specify boundary conditions
def create_bcs(V, subdomains, **NS_namespace):
    bc0_bead = DirichletBC(V, 0, subdomains, 1)
    bc0_epscyl = DirichletBC(V, 0, subdomains, 2)
    bc0_wall = DirichletBC(V, 0, subdomains, 3)
    return dict(u0=[bc0_bead, bc0_epscyl, bc0_wall],
                u1=[bc0_bead, bc0_epscyl, bc0_wall],
                u2=[bc0_bead, bc0_epscyl, bc0_wall],
                p=[],
                phig=[])


def average_pressure_gradient(**NS_namespace):
    # average pressure gradient
    return Constant((0., 0., 0.))


def acceleration(g0, **NS_namespace):
    # (gravitational) acceleration
    return Constant(tuple(g0))


def initialize(q_, q_1, x_1, x_2, bcs, epsilon, r, VV, Ly, **NS_namespace):
    phig_init = interpolate(Expression(
        "tanh((sqrt(pow(x[1], 2)+pow(x[2], 2))-r)/(sqrt(2)*epsilon))",
        epsilon=epsilon, Ly=Ly, r=r, degree=2), VV['phig'].sub(0).collapse())
    assign(q_['phig'].sub(0), phig_init)
    q_1['phig'].vector()[:] = q_['phig'].vector()
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(mesh, u_, newfolder, velocity_degree, pressure_degree, AssignedVectorFunction, 
                   g0, mu, rho, sigma, M, theta_bead, theta_wall, epsilon, res, dt, 
                   Lx, Ly, H, R, reps,
                   **NS_namespace):
    volume = assemble(Constant(1.) * dx(domain=mesh))
    statsfolder = path.join(newfolder, "Stats")
    timestampsfolder = path.join(newfolder, "Timestamps")

    # Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree)
    uv = AssignedVectorFunction(u_, name="u")

    if MPI.rank(MPI.comm_world) == 0 and not path.exists(statsfolder):
        makedirs(statsfolder)

    if MPI.rank(MPI.comm_world) == 0 and not path.exists(timestampsfolder):
        makedirs(timestampsfolder)

    """
    if MPI.rank(MPI.comm_world) == 0:
        with open(path.join(timestampsfolder, "params.dat"), "a+") as ofile:
            keys = ["g0", "mu", "rho", "sigma", "M", "theta_bead", "theta_wall", "epsilon", "Lx", "Ly", "H", "R", "reps", "res", "dt"]
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
    """

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
    """
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
    """

def theend_hook(u_, p_, testing, **NS_namespace):
    if not testing:
        plot(u_, title='Velocity')
        plot(p_, title='Pressure')

    u_norm = norm(u_[0].vector())
    if MPI.rank(MPI.comm_world) == 0 and testing:
        print("Velocity norm = {0:2.6e}".format(u_norm))
