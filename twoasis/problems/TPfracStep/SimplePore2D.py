"""A curved meniscus moving through a single pore throat between two open ends held at
the same pressure, periodic across the throat (mesh from meshes/lattice_mesh.py)."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2023-06-05"
__copyright__ = ("Copyright (C) 2023 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"

from os import path

import numpy as np
from dolfin import (
    assemble, assign, Constant, DirichletBC, dx, Expression, interpolate, MPI)

from twoasis.problems import info_red
from twoasis.problems.TPfracStep.Porous2D import Walls
from twoasis.problems.TPfracStep.shared import (
    centred, Faces, interface_energy, load_mesh, load_restart_parameters,
    makedirs_on_root, PeriodicBox, write_timestamp, write_timestamps_info)


def get_fname(Lx, Ly, rad, res, ext):
    meshparams = dict(Lx=Lx, Ly=Ly, rad=rad, res=res, ext=ext)
    fname = "meshes/lattice_Lx{Lx}_Ly{Ly}_r{rad}_dx{res}.{ext}".format(**meshparams)    
    return fname

# Create a mesh
def mesh(Lx, Ly, rad, res, **params):
    return load_mesh(get_fname(Lx, Ly, rad, res, "h5"))

# Override some problem specific parameters
def problem_parameters(NS_parameters, commandline_kwargs, **NS_namespace):
    if not load_restart_parameters(NS_parameters, commandline_kwargs):
        NS_parameters.update(
            T=100.0,
            Lx=4.0,
            Ly=2.0,
            rad=0.5,
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


def constrained_domain(Ly, **params):
    return PeriodicBox({1: centred(Ly)})


def mark_subdomains(subdomains, dim, Lx, Ly, rad, res, **NS_namespace):
    fname = get_fname(Lx, Ly, rad, res, "dat")
    obst = np.loadtxt(fname)
    x = obst[:, :dim]
    r = obst[:, dim]
    wall = Walls(x, r)
    wall.mark(subdomains, 1)
    left = Faces(0, -Lx/2)
    left.mark(subdomains, 2)
    right = Faces(0, Lx/2)
    right.mark(subdomains, 3)
    return dict()


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


def pre_solve_hook(tstep, t, q_, p_, mesh, u_, newfolder, velocity_degree, pressure_degree,
                   AssignedVectorFunction, constrained_domain, **NS_namespace):
    volume = assemble(Constant(1.) * dx(domain=mesh))
    statsfolder = path.join(newfolder, "Stats")
    timestampsfolder = path.join(newfolder, "Timestamps")
    uv = AssignedVectorFunction(u_, name="u")
    makedirs_on_root(statsfolder, timestampsfolder)
    write_timestamps_info(timestampsfolder, mesh,
                          {key: NS_namespace[key] for key in
                           ["F0", "g0", "mu", "rho", "sigma", "M", "theta", "epsilon", "rad", "res", "dt", "Lx", "Ly"]},
                          velocity_degree, pressure_degree, constrained_domain)
    write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)
    return dict(uv=uv, statsfolder=statsfolder, timestampsfolder=timestampsfolder, volume=volume)


def temporal_hook(q_, tstep, t, u_, p_, phi_, rho_, sigma, epsilon, volume, statsfolder,
                  timestampsfolder, stat_interval, timestamps_interval, uv, mesh, **NS_namespace):
    """Write to Stats/tdata.dat: tstep, t, u0m, u1m, phim and the energies per
    volume (kinetic, interface gradient and double-well parts); and u, p and phi
    to Timestamps/ every timestamps_interval steps."""
    info_red("tstep = {}".format(tstep))
    if tstep % stat_interval == 0:
        dX = dx(domain=mesh)
        um = [assemble(q_['u{}'.format(i)] * dX) / volume for i in range(2)]
        phim = assemble(phi_ * dX) / volume
        E_kin = 0.5 * assemble(rho_ * (u_[0]**2 + u_[1]**2) * dX) / volume
        E_int, E_pot = (E / volume for E in interface_energy(phi_, sigma, epsilon, dX))
        if MPI.rank(MPI.comm_world) == 0:
            with open(path.join(statsfolder, "tdata.dat"), "a") as tfile:
                entries = um + [phim, E_kin, E_int, E_pot]
                tfile.write(("{:d} {:.8f} " + " ".join("{:.8f}" for _ in entries) + "\n").format(
                    tstep, t, *entries))
    if tstep % timestamps_interval == 0:
        write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)
