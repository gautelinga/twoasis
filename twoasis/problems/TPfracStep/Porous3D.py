"""Two-phase flow through a periodic three-dimensional porous medium, driven by a mean
pressure gradient and starting from a checkerboard of the two phases. The medium
(meshes/porous3d_rcp01) is a random close packing of spheres of radius 0.1 in the unit
cube, made with yade_rcp.py (run with yade) and meshed with pore_mesh.py, both in
rcp_beads/ of meshtools (https://github.com/gautelinga/meshtools)."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2022-11-25"
__copyright__ = ("Copyright (C) 2022 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"

from os import path

import numpy as np
from dolfin import DirichletBC, Expression

from twoasis.common.utilities import set_subfunction
from twoasis.problems import info_red
from twoasis.problems.TPfracStep.shared import (
    load_mesh, makedirs_on_root, mark_walls, PeriodicBox, setup_stats, standard_stats, write_tdata,
    write_timestamp, write_timestamps_info)




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
def mesh(L, R, reps, res, **params):
    return load_mesh(get_fname(L, R, reps, res, "h5"), periodic=(True, True, True))

# Override some problem specific parameters
def problem_parameters(NS_parameters, commandline_kwargs, **NS_namespace):
    NS_parameters.update(
        T=100.0,
        L=1.0,
        R=0.1, # dummy
        reps=0.02,
        res=0.01, #dummy for now
        dt=0.1,
        rho=[1, 1],
        mu=[1, 1],
        theta=np.pi/3,
        epsilon=0.01,
        sigma=0.1,
        checkerboard=[4, 4, 4],
        M=0.00002,
        F0=[0., 0., 10],
        g0=[0., 0., 0.],
        velocity_degree=1,
        folder="porous3d_results",
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


def constrained_domain(L, **params):
    return PeriodicBox({i: (0, L) for i in range(3)})


def mark_subdomains(subdomains, mesh, **NS_namespace):
    """Mark the obstacles' walls 1."""
    mark_walls(subdomains, mesh)


# Specify boundary conditions
def create_bcs(V, subdomains, **NS_namespace):
    bc_u_wall = DirichletBC(V, 0, subdomains, 1)
    return dict(u0=[bc_u_wall],
                u1=[bc_u_wall],
                u2=[bc_u_wall],
                p=[],
                phig=[])


def initialize(q_, q_1, x_1, x_2, bcs, epsilon, VV, L, checkerboard, **NS_namespace):
    ccode = ccode_expr(checkerboard)
    set_subfunction(q_['phig'], 0, Expression(
        #"tanh((sqrt(pow(x[0], 2)+pow(x[1], 2))-0.45)/(sqrt(2)*epsilon))",
        #"tanh((x[2]-0.75*L)/(sqrt(2)*epsilon))-tanh((x[2]-0.25*L)/(sqrt(2)*epsilon))+1",
        ccode,
        epsilon=epsilon, L=L, degree=2))
    q_1['phig'].vector()[:] = q_['phig'].vector()
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(tstep, t, q_, p_, mesh, u_, newfolder, velocity_degree, pressure_degree,
                   phase_field_degree, AssignedVectorFunction, constrained_domain, subdomains, **NS_namespace):
    stats = setup_stats(mesh, newfolder)
    timestampsfolder = path.join(newfolder, "Timestamps")
    uv = AssignedVectorFunction(u_, name="u")
    makedirs_on_root(timestampsfolder)
    write_timestamps_info(timestampsfolder, mesh,
                          {key: NS_namespace[key] for key in
                           ["F0", "g0", "mu", "rho", "sigma", "M", "theta", "epsilon", "R", "reps", "res", "dt"]},
                          velocity_degree, pressure_degree, phase_field_degree,
                          constrained_domain, subdomains)
    write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)
    return dict(uv=uv, timestampsfolder=timestampsfolder, **stats)


def temporal_hook(q_, tstep, t, u_, p_, phi_, rho_, sigma, epsilon, volume, statsfolder,
                  timestampsfolder, stat_interval, timestamps_interval, uv, mesh, **NS_namespace):
    """Write to Stats/tdata.dat: tstep, t, u0m, u1m, u2m, phim and the energies per
    volume (kinetic, interface gradient and double-well parts); and u, p and phi
    to Timestamps/ every timestamps_interval steps."""
    info_red("tstep = {}".format(tstep))
    if tstep % stat_interval == 0:
        write_tdata(statsfolder, tstep, t,
                    standard_stats(q_, u_, phi_, rho_, sigma, epsilon, volume, mesh))
    if tstep % timestamps_interval == 0:
        write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)
