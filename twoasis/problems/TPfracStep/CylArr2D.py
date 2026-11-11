"""Single-phase flow through a periodic hexagonal array of cylinders (mesh from
Bernaise, see get_fname), driven by a force F0 along y (minus the mean pressure
gradient; the flow goes towards -y). The phase field is uniform (phi = 1), so the chemical potential and the
capillary force vanish and this is plain Navier-Stokes flow; the velocity and pressure
written to Timestamps/ serve e.g. particle tracking."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2022-03-22"
__copyright__ = ("Copyright (C) 2022 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"

import numpy as np
from dolfin import assemble, Constant, DirichletBC, dx

from twoasis.common.utilities import set_subfunction
from twoasis.problems import info_red
from twoasis.problems.TPfracStep.shared import (
    centred, load_mesh, mark_walls, PeriodicBox, start_timestamps, write_tdata,
    write_timestamp)


def get_fname(Lx, rad, res, ext):
    """The mesh (.h5) and the cylinders' centres and radii (.dat), made with
    Bernaise: python3 generate_mesh.py mesh=cyl_arr_2d Lx=1.0 rad=0.33 dx=0.02,
    run in its utilities/ folder. The cell is Lx wide and sqrt(3)*Lx high."""
    return "meshes/cyl_arr_Lx{}_rad{}_dx{}.{}".format(Lx, rad, res, ext)


def mesh(Lx, rad, res, **params):
    return load_mesh(get_fname(Lx, rad, res, "h5"))


# Override some problem specific parameters
def problem_parameters(NS_parameters, commandline_kwargs, **NS_namespace):
    NS_parameters.update(
        T=1000.0,
        Lx=1.,
        rad=0.33,
        res=0.02,
        dt=0.01,
        rho=[1, 1],
        mu=[1, 1],
        F0=[0., -10.],
        g0=[0., 0.],
        velocity_degree=2,
        phase_field_degree=1,  # phi = 1 is represented exactly at any degree
        folder="cylarr2d_results",
        plot_interval=10,
        stat_interval=10,
        timestamps_interval=10,
        save_step=10,
        checkpoint=10,
        print_intermediate_info=10,
        solver="IPCS")


def constrained_domain(Lx, **params):
    return PeriodicBox({0: centred(Lx), 1: centred(np.sqrt(3) * Lx)})


def mark_subdomains(subdomains, mesh, **NS_namespace):
    """Mark the cylinders' walls 1."""
    mark_walls(subdomains, mesh)


def create_bcs(V, subdomains, **NS_namespace):
    bc_u_wall = DirichletBC(V, 0, subdomains, 1)
    return dict(u0=[bc_u_wall],
                u1=[bc_u_wall],
                p=[],
                phig=[])


def initialize(q_, q_1, VV, **NS_namespace):
    set_subfunction(q_['phig'], 0, Constant(1.))
    q_1['phig'].vector()[:] = q_['phig'].vector()


def pre_solve_hook(**NS_namespace):
    return start_timestamps(["F0", "g0", "mu", "rho", "rad", "res", "dt"], **NS_namespace)


def temporal_hook(q_, tstep, t, u_, p_, rho_, volume, statsfolder, timestampsfolder,
                  stat_interval, timestamps_interval, uv, mesh, **NS_namespace):
    """Write to Stats/tdata.dat: tstep, t, the mean velocity (u0m, u1m) and the
    kinetic energy per volume; and u, p and phi to Timestamps/ every
    timestamps_interval steps."""
    info_red("tstep = {}".format(tstep))
    if tstep % stat_interval == 0:
        dX = dx(domain=mesh)
        u0m = assemble(q_['u0'] * dX) / volume
        u1m = assemble(q_['u1'] * dX) / volume
        E_kin = 0.5 * assemble(rho_ * (u_[0]**2 + u_[1]**2) * dX) / volume
        write_tdata(statsfolder, tstep, t, [u0m, u1m, E_kin])
    if tstep % timestamps_interval == 0:
        write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)
