"""Displacement in a three-dimensional porous medium: one phase enters at a prescribed
velocity at the bottom and leaves through the top, starting from a flat interface at a
given height. The medium is a random close packing of spheres, made and meshed with
the scripts in rcp_beads/ of meshtools (https://github.com/gautelinga/meshtools):
yade_rcp.py (run with yade) for the packing, and msp_mesh.py, which meshes its pore
space with Chris Knight's mesh-sphere-packing
(https://github.com/chrisk314/mesh-sphere-packing), changed in one line to make the
mesh uniform in the bulk, and writes the mesh (.h5) and the spheres (.obst).

Used in G. Linga et al., Dynamic multiphase flow triggers chaotic mixing in porous
media, PNAS 123, e2612884123 (2026), doi:10.1073/pnas.2612884123."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2024-02-28"
__copyright__ = ("Copyright (C) 2024 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"

from os import path

import numpy as np
from dolfin import assemble, Constant, DirichletBC, Expression, Function, interpolate

from twoasis.common.utilities import set_subfunction
from twoasis.problems import info_red
from twoasis.problems.TPfracStep.shared import (
    Faces, load_mesh, makedirs_on_root, mark_walls, mesh_extrema, PeriodicBox, setup_stats,
    standard_stats, write_tdata, write_timestamp, write_timestamps_info)





def mesh(meshfile, **params):
    return load_mesh(meshfile, periodic=(True, True, False))

# Override some problem specific parameters
def problem_parameters(NS_parameters, **NS_namespace):
    NS_parameters.update(
        T=100.0,
        meshfile="meshes/test_4x4x8_out.h5",  # with the obstacles in test_4x4x8_out.obst
        dt=0.1,
        rho=[1, 1],
        mu=[1, 1],
        theta=np.pi/2,
        epsilon=0.05,
        sigma=1.0,
        M=0.0001,
        u0=0.01,
        injected_phase=-1,
        zfrac=0.1,
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


def constrained_domain(mesh, **params):
    x_min, x_max = mesh_extrema(mesh)
    return PeriodicBox({0: (x_min[0], x_max[0]), 1: (x_min[1], x_max[1])})


def mark_subdomains(subdomains, mesh, **NS_namespace):
    """Mark the obstacles 1, the bottom 2 and the top 3, and add the mesh's
    bounding box x_min, x_max to the namespace."""
    x_min, x_max = mesh_extrema(mesh)
    mark_walls(subdomains, mesh)
    Faces(2, x_min[2]).mark(subdomains, 2)
    Faces(2, x_max[2]).mark(subdomains, 3)
    return dict(x_min=x_min, x_max=x_max)

def contact_angles(theta, V, meshfile, sigma, x_min, x_max, **NS_namespace):
    obst = np.loadtxt(meshfile[:-2] + "obst", ndmin=2)
    s = Function(V, name="costheta")
    xx = np.vstack([interpolate(Expression(f"x[{d}]", degree=0), V).vector()[:] for d in range(3)]).T

    frac = 0.15
    cutoff = 0.99
    factor = 1.3

    s_ = np.zeros_like(s.vector()[:])
    for i, xr in enumerate(obst):
        #print(i)
        x, r = xr[:3], xr[3]
        reps = np.linalg.norm(xx - np.outer(np.ones(len(xx)), x), axis=1)/r - cutoff
        ds = 1-reps/frac
        s_[ds > 0] += ds[ds > 0]

    rmean = obst[:, 3].mean()

    # top
    reps = (x_max[2] - xx[:, 2])/rmean + 1-cutoff
    ds = 1-reps/frac
    s_[ds > 0] += ds[ds > 0]

    # bottom
    reps = (xx[:, 2] - x_min[2])/rmean + 1-cutoff
    ds = 1-reps/frac
    s_[ds > 0] += ds[ds > 0]

    s_[:] -= 1.0
    s_[s_ < 0] = 0.0
    s_[:] *= factor
    s_[s_ > 1] = 1.0

    s.vector()[:] = 1-s_
    s.vector()[:] *= np.cos(theta)

    #return [(theta, 1)]
    return [(s, 1)]

# Specify boundary conditions
def create_bcs(V, W, subdomains, u0, injected_phase, zfrac, **NS_namespace):
    bc_u_wall = DirichletBC(V, 0, subdomains, 1)
    bc_u_btm = DirichletBC(V, 0, subdomains, 2)
    bc_uz_btm = DirichletBC(V, u0, subdomains, 2)
    bc_u_top = DirichletBC(V, 0, subdomains, 3)
    bc_uz_top = DirichletBC(V, u0, subdomains, 3)
    bc_phig_btm = DirichletBC(W.sub(0), injected_phase, subdomains, 2)    
    ejected_phase = -injected_phase if zfrac > 0.0 and zfrac < 1.0 else injected_phase
    bc_phig_top = DirichletBC(W.sub(0), ejected_phase, subdomains, 3)    
    return dict(u0=[bc_u_wall, bc_u_btm, bc_u_top],
                u1=[bc_u_wall, bc_u_btm, bc_u_top],
                u2=[bc_u_wall, bc_uz_btm, bc_uz_top],
                p=[],
                phig=[bc_phig_btm, bc_phig_top])


def initialize(q_, q_1, q_2, x_1, x_2, bcs, epsilon, VV, x_min, x_max, injected_phase, zfrac, **NS_namespace):
    #zfrac = 0.1
    z0 = zfrac*x_max[2]+(1-zfrac)*x_min[2]
    set_subfunction(q_['phig'], 0, Expression(
        "-injected_phase*tanh((x[2]-z0)/(sqrt(2)*epsilon))",
        epsilon=epsilon, z0=z0, injected_phase=injected_phase, degree=2))
    q_1['phig'].vector()[:] = q_['phig'].vector()
    q_2['phig'].vector()[:] = q_['phig'].vector()
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(tstep, t, q_, p_, mesh, u_, newfolder, velocity_degree, pressure_degree,
                   phase_field_degree, AssignedVectorFunction, constrained_domain, subdomains, ds, **NS_namespace):
    stats = setup_stats(mesh, newfolder)
    area = [assemble(Constant(1.) * ds(i)) for i in (2, 3)]  # bottom, top
    timestampsfolder = path.join(newfolder, "Timestamps")
    uv = AssignedVectorFunction(u_, name="u")
    makedirs_on_root(timestampsfolder)
    write_timestamps_info(timestampsfolder, mesh,
                          {key: NS_namespace[key] for key in
                           ["F0", "g0", "mu", "rho", "sigma", "M", "theta", "epsilon", "dt"]},
                          velocity_degree, pressure_degree, phase_field_degree,
                          constrained_domain, subdomains)
    write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)
    return dict(uv=uv, timestampsfolder=timestampsfolder, area=area, **stats)


def temporal_hook(q_, tstep, t, u_, p_, phi_, rho_, sigma, epsilon, volume, area, statsfolder,
                  stat_interval, ds, angles, mesh, **NS_namespace):
    """Write to Stats/tdata.dat: tstep, t, u0m, u1m, u2m, phim, the energies per
    volume (kinetic, interface gradient and double-well parts), the wall energy
    E_surf (sigma*cos(theta)*(phi^3 - 3*phi + 2)/4 on the obstacles: the energy whose
    derivative the solver's wall term is, zero for phi = 1) and the mean pressure
    drop dp from bottom to top."""
    info_red("tstep = {}".format(tstep))
    if tstep % stat_interval == 0:
        entries = standard_stats(q_, u_, phi_, rho_, sigma, epsilon, volume, mesh)
        costheta, mark = angles[0]
        E_surf = 0.25 * sigma * assemble(costheta * (phi_**3 - 3 * phi_ + 2) * ds(mark))
        dp = assemble(p_ * ds(2)) / area[0] - assemble(p_ * ds(3)) / area[1]
        write_tdata(statsfolder, tstep, t, entries + [E_surf, dp])
