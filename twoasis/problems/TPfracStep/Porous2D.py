"""Two-phase flow through a periodic two-dimensional porous medium of disks (mesh from
meshes/, made with periodic_porous_2d in Bernaise), driven by a mean pressure gradient
and starting from a band, a checkerboard or a saved state.

Used in G. Linga et al., Dynamic multiphase flow triggers chaotic mixing in porous
media, PNAS 123, e2612884123 (2026), doi:10.1073/pnas.2612884123."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2022-03-18"
__copyright__ = ("Copyright (C) 2022 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"

from os import path

import numpy as np
from dolfin import (
    assemble, Constant, DirichletBC, dx, Expression, Function, FunctionAssigner, HDF5File,
    interpolate, VectorFunctionSpace)

from twoasis.common.utilities import set_subfunction, subspace
from twoasis.problems import info_blue, info_red
from twoasis.problems.TPfracStep.shared import (
    centred, load_mesh, makedirs_on_root, mark_walls, PeriodicBox, setup_stats, standard_stats,
    write_tdata, write_timestamp, write_timestamps_info)




def get_fname(Lx, Ly, rad, R, N, res, ext):
    meshparams = dict(Lx=Lx, Ly=Ly, rad=rad, R=R, N=N, res=res, ext=ext)
    fname = "meshes/periodic_porous_Lx{Lx}_Ly{Ly}_r{rad}_R{R}_N{N}_dx{res}.{ext}".format(**meshparams)    
    return fname

def initialize_checkerboard_sat(phi_init, phi_target, checkerboard, L, epsilon, tol=1e-8):
    if abs(phi_target) > 1 - tol:
        phi_init.vector()[:] = np.sign(phi_target)
        return
    
    L = np.array(L)
    x0 = -L/2 # np.array([-Lx/2, -Ly/2])
    dims = range(len(L))
    Dx = L / np.array(checkerboard) / 2

    x_ = [interpolate(Expression(f"x[{dim}]", degree=1), phi_init.function_space()).vector()[:] for dim in dims]
    i_ = [np.floor((x_[dim]-x0[dim])/Dx[dim]) for dim in dims]
    xi_ = [x0[dim] + (i_[dim] + 0.5) * Dx[dim] for dim in dims]
    dxi_ = [abs(x_[dim] - xi_[dim]) for dim in dims]
    ir_ = sum([np.round((x_[dim]-x0[dim])/Dx[dim]) for dim in dims])
    psi_ = np.minimum.reduce(dxi_) * (-1)**ir_

    c = 0.
    vol = assemble(Constant(1.) * dx(domain=phi_init.function_space().mesh()))
    phi_mean = 0.
    it = 0
    # Newton iterations
    while abs(phi_mean - phi_target) > tol:
        info_blue(f"it={it}, c={c}, psi_mean={phi_mean}")
        phi_init.vector()[:] = np.tanh( (psi_ + c ) / (np.sqrt(2) * epsilon))
        phi_mean = assemble(phi_init * dx) / vol
        phi2_mean = assemble(phi_init**2 * dx) / vol
        c += np.sqrt(2) * epsilon * (phi_target - phi_mean) / (1 - phi2_mean)
        it += 1

    phi_init.vector()[:] = np.tanh((psi_ + c)/(np.sqrt(2) * epsilon))

# Create a mesh
def mesh(Lx, Ly, rad, R, N, res, **params):
    return load_mesh(get_fname(Lx, Ly, rad, R, N, res, "h5"))

# Override some problem specific parameters
def problem_parameters(NS_parameters, **NS_namespace):
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
        theta=np.pi/3,
        epsilon=0.05,
        sigma=5.0,
        M=0.0001,
        F0=[0., -10.],
        g0=[0., 0.],
        init_state="",
        checkerboard=[12, 18],
        phi_target=0.0,
        velocity_degree=1,
        folder="porous2d_results",
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


def mark_subdomains(subdomains, mesh, **NS_namespace):
    """Mark the obstacles' walls 1."""
    mark_walls(subdomains, mesh)


# Specify boundary conditions
def create_bcs(V, subdomains, **NS_namespace):
    bc_u_wall = DirichletBC(V, 0, subdomains, 1)
    return dict(u0=[bc_u_wall],
                u1=[bc_u_wall],
                p=[],
                phig=[])


def initialize(q_, q_1, q_2, x_1, x_2, bcs, epsilon, VV, Lx, Ly, init_state, checkerboard, mesh, V, Q,
               phi_target, **NS_namespace):
    if init_state == "":
        if checkerboard is None:
            phi_init = Expression(
                #"tanh((sqrt(pow(x[0], 2)+pow(x[1], 2))-0.45)/(sqrt(2)*epsilon))",
                "tanh((x[1]-0.25*Ly)/(sqrt(2)*epsilon))-tanh((x[1]+0.25*Ly)/(sqrt(2)*epsilon))+1",
                epsilon=epsilon, Ly=Ly, degree=2)
        else:
            phi_init = Function(subspace(VV['phig'], 0), name="phi")
            initialize_checkerboard_sat(phi_init, phi_target, checkerboard, [Lx, Ly], epsilon)
        set_subfunction(q_['phig'], 0, phi_init)
        q_1['phig'].vector()[:] = q_['phig'].vector()
    else:
        if not path.exists(init_state):
            raise FileNotFoundError("initial state {} does not exist".format(init_state))
        phi_init = Function(subspace(VV['phig'], 0))
        Vv = VectorFunctionSpace(mesh, V.ufl_element().family(), V.ufl_element().degree(),
                                constrained_domain=V.dofmap().constrained_domain)
        u_init = Function(Vv)
        with HDF5File(mesh.mpi_comm(), init_state, "r") as h5f:
            h5f.read(phi_init, "phi")
            h5f.read(u_init, "u")
            h5f.read(q_['p'], "p")
        set_subfunction(q_['phig'], 0, phi_init)
        # Both velocity components in one assignment.
        FunctionAssigner([q_['u0'].function_space(), q_['u1'].function_space()], Vv).assign(
            [q_['u0'], q_['u1']], u_init)
        for ui in q_:
            q_1[ui].vector()[:] = q_[ui].vector()[:]
            q_2[ui].vector()[:] = q_[ui].vector()[:]

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
                           ["F0", "g0", "mu", "rho", "sigma", "M", "theta", "epsilon", "rad", "res", "dt"]},
                          velocity_degree, pressure_degree, phase_field_degree,
                          constrained_domain, subdomains)
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
