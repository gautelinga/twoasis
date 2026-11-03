"""A bead between two parallel plates that it touches, in a cell periodic in x and y;
the second phase starts as a thin cylinder along x and meets bead and plates at set
contact angles."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2022-05-06"
__copyright__ = ("Copyright (C) 2022 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"

import numpy as np
from dolfin import (
    assign, DirichletBC, DOLFIN_EPS_LARGE, Expression, interpolate, SubDomain)

from twoasis.problems import info_red
from twoasis.problems.TPfracStep.shared import (
    centred, load_mesh, PeriodicBox, setup_stats, standard_stats, write_tdata)

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


def get_fname(Lx, Ly, H, R, res, ext):
    meshparams = dict(Lx=Lx, Ly=Ly, H=H, R=R, res=res, ext=ext)
    fname = "meshes/beads_Lx{Lx}_Ly{Ly}_H{H}_R{R}_res{res}.{ext}".format(**meshparams)    
    return fname

# Create a mesh
def mesh(Lx, Ly, H, R, res, **params):
    return load_mesh(get_fname(Lx, Ly, H, R, res, "h5"))

# Override some problem specific parameters
def problem_parameters(NS_parameters, commandline_kwargs, **NS_namespace):
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


def constrained_domain(Lx, Ly, **params):
    return PeriodicBox({0: centred(Lx), 1: centred(Ly)})


def mark_subdomains(subdomains, Lx, Ly, H, R, reps, **NS_namespace):
    epscyl = EpsCyl(reps)
    bead = Bead(R)
    wall = Walls(H)
    epscyl.mark(subdomains, 2)
    bead.mark(subdomains, 1)
    wall.mark(subdomains, 3)
    return dict()

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


def pre_solve_hook(mesh, newfolder, **NS_namespace):
    return setup_stats(mesh, newfolder)


def temporal_hook(q_, tstep, t, u_, phi_, rho_, sigma, epsilon, volume, statsfolder,
                  stat_interval, mesh, **NS_namespace):
    """Write to Stats/tdata.dat: tstep, t, u0m, u1m, u2m, phim and the energies per
    volume (kinetic, interface gradient and double-well parts)."""
    info_red("tstep = {}".format(tstep))
    if tstep % stat_interval == 0:
        write_tdata(statsfolder, tstep, t,
                    standard_stats(q_, u_, phi_, rho_, sigma, epsilon, volume, mesh))
