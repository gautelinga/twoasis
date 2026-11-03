"""Displacement in a two-dimensional porous medium of disks (mesh from meshes/): one
phase enters at a prescribed velocity at the bottom and pushes the other out through
the top, starting from a flat interface or a saved state."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2022-11-20"
__copyright__ = ("Copyright (C) 2022 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"

from os import path

import numpy as np
from dolfin import assign, DirichletBC, Expression, interpolate

from twoasis.common.io import load_timeseries
from twoasis.problems import info_blue, info_red
from twoasis.problems.TPfracStep import Porous2D
from twoasis.problems.TPfracStep.Porous2D import get_fname, Walls
from twoasis.problems.TPfracStep.shared import (
    centred, Faces, makedirs_on_root, PeriodicBox, read_phase_field, setup_stats,
    standard_stats, write_tdata, write_timestamp, write_timestamps_info)

# The mesh and its file name are those of Porous2D.
mesh = Porous2D.mesh

def problem_parameters(NS_parameters, **NS_namespace):
    NS_parameters.update(
        T=100.0,
        Lx=10,
        Ly=20,
        rad=0.5,
        N=88,
        res=0.05,
        R=0.6,
        dt=0.02,  # 0.1 diverges
        rho=[1, 1],
        mu=[10, 1],
        u0=0.001,
        y0=-8,
        theta=np.pi/3,
        epsilon=0.05,
        sigma=5.0,
        M=0.0001,
        F0=[0., 0.],
        g0=[0., 0.],
        velocity_degree=1,
        folder="porous2d_flux_results",
        plot_interval=10,
        stat_interval=10,
        timestamps_interval=10,
        save_step=100,
        checkpoint=10000,
        print_intermediate_info=10,
        use_krylov_solvers=True,
        solver="BDF",
        bdf_order=1,
        AB_projection_pressure=False,
        max_iter=10,                  # Number of inner pressure velocity iterations on timestep
        max_error=1e-3,               # Tolerance for inner iterations (pressure velocity iterations)
        iters_on_first_timestep=20,  # Number of iterations on first timestep
        initial_state=None,
        injected_phase=-1,
    )


def constrained_domain(Lx, **params):
    return PeriodicBox({0: centred(Lx)})


def mark_subdomains(subdomains, dim, Lx, Ly, rad, R, N, res, **NS_namespace):
    fname = get_fname(Lx, Ly, rad, R, N, res, "dat")
    obst = np.loadtxt(fname)
    x = obst[:, :dim]
    r = obst[:, dim]
    wall = Walls(x, r)
    wall.mark(subdomains, 1)
    top = Faces(1, Ly/2)
    top.mark(subdomains, 2)
    btm = Faces(1, -Ly/2)
    btm.mark(subdomains, 3)
    return dict()


# Specify boundary conditions
def create_bcs(V, Q, u0, W, subdomains, injected_phase, **NS_namespace):
    bc_ux_wall = DirichletBC(V, 0, subdomains, 1)
    bc_uy_wall = DirichletBC(V, 0, subdomains, 1)
    bc_ux_top = DirichletBC(V, 0, subdomains, 2)
    bc_uy_top = DirichletBC(V, u0, subdomains, 2)
    bc_ux_btm = DirichletBC(V, 0, subdomains, 3)
    bc_uy_btm = DirichletBC(V, u0, subdomains, 3)
    #bc_p_top = DirichletBC(Q, 0, subdomains, 2)
    #bc_p_btm = DirichletBC(Q, 0, subdomains, 3)
    #bc_phig_top = DirichletBC(W.sub(0), -1 * injected_phase, subdomains, 2)
    bc_phig_btm = DirichletBC(W.sub(0), 1 * injected_phase, subdomains, 3)
    return dict(u0=[bc_ux_wall, bc_ux_btm, bc_ux_top],
                u1=[bc_uy_wall, bc_uy_btm, bc_uy_top],
                p=[],
                phig=[bc_phig_btm])


def initialize(q_, q_1, q_2, x_1, x_2, bcs, epsilon, VV, Ly, y0, initial_state, mesh,
               constrained_domain, **NS_namespace):
    if initial_state is None:
        phi_init = interpolate(Expression(
            #"tanh((sqrt(pow(x[0], 2)+pow(x[1], 2))-0.45)/(sqrt(2)*epsilon))",
            #"tanh((x[1]-0.25*Ly)/(sqrt(2)*epsilon))-tanh((x[1]+0.25*Ly)/(sqrt(2)*epsilon))+1",
            "tanh((x[1]-y0)/(sqrt(2)*epsilon))",
            epsilon=epsilon, Ly=Ly, y0=y0, degree=2), VV['phig'].sub(0).collapse())
        assign(q_['phig'].sub(0), phi_init)
    elif initial_state.replace("/", "").endswith("Timeseries"):
        # The last state in the Timeseries/ folder of an earlier run.
        info_blue("initializing from: " + initial_state)
        load_timeseries(initial_state, q_, VV, constrained_domain)
        for key in q_.keys():
            q_1[key].vector()[:] = q_[key].vector()
            q_2[key].vector()[:] = q_[key].vector()
    else:
        info_blue("initializing from: " + initial_state)
        phi_init = read_phase_field(initial_state, mesh, q_)
        assign(q_['phig'].sub(0), phi_init)

    q_1['phig'].vector()[:] = q_['phig'].vector()
    q_2['phig'].vector()[:] = q_['phig'].vector()
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(tstep, t, q_, p_, mesh, u_, newfolder, velocity_degree, pressure_degree,
                   phase_field_degree, AssignedVectorFunction, constrained_domain, **NS_namespace):
    stats = setup_stats(mesh, newfolder)
    timestampsfolder = path.join(newfolder, "Timestamps")
    uv = AssignedVectorFunction(u_, name="u")
    makedirs_on_root(timestampsfolder)
    write_timestamps_info(timestampsfolder, mesh,
                          {key: NS_namespace[key] for key in
                           ["F0", "g0", "mu", "rho", "sigma", "M", "theta", "epsilon", "rad", "res", "dt", "Lx", "Ly"]},
                          velocity_degree, pressure_degree, phase_field_degree,
                          constrained_domain)
    write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)
    return dict(uv=uv, timestampsfolder=timestampsfolder, **stats)


def temporal_hook(q_, tstep, t, u_, p_, phi_, rho_, sigma, epsilon, volume, statsfolder,
                  timestampsfolder, stat_interval, timestamps_interval, uv, mesh, **NS_namespace):
    """Write to Stats/tdata.dat: tstep, t, u0m, u1m, phim and the energies per
    volume (kinetic, interface gradient and double-well parts); and u, p and phi
    to Timestamps/ every timestamps_interval steps."""
    info_red("tstep = {}".format(tstep))
    if tstep % stat_interval == 0:
        write_tdata(statsfolder, tstep, t,
                    standard_stats(q_, u_, phi_, rho_, sigma, epsilon, volume, mesh))
    if tstep % timestamps_interval == 0:
        write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)
