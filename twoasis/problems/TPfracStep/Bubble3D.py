"""A bubble rising in a box, the 3D benchmark of Adelsberger et al. (2014), test case 1
(Re = 35, Eo = 10: the densities, viscosities and surface tension of Hysing's 2D test
case 1): the box [0, Lx] x [0, Ly] x [0, Lz] with no-slip walls at bottom and top,
the bubble rising along z, and periodic sides, or with periodic=False free-slip side
walls as in the benchmark.

Stats/benchmark.dat holds the benchmark's quantities (shared.hysing_quantities): the
bubble's volume, centroid height z_c, rise velocity and sphericity. Their reference
values: largest rise velocity 0.355 near t = 0.9, sphericity 0.955 at t = 3.

The phase field and the chemical potential are P1, as the pressure (the velocity P2);
res = 1/16 is the coarsest mesh that resolves the interface in 3D.

J. Adelsberger, P. Esser, M. Griebel, S. Gross, M. Klitz and A. Rüttgers, 3D incompressible
two-phase flow benchmark computations for rising droplets, Proc. WCCM XI (2014)."""
__author__ = "Gaute Linga <gaute.linga@ntnu.no>"
__date__ = "2026-11-12"
__copyright__ = "Copyright (C) 2026 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from os import path

import numpy as np
from twoasis.backend import BoxMesh, DirichletBC, dx, Expression, MPI, Point

from twoasis.common.utilities import set_subfunction
from twoasis.problems import info_red
from twoasis.problems.TPfracStep.shared import (
    append_row, Faces, gravitational_energy, hysing_quantities, PeriodicBox, phase_moments,
    setup_stats, standard_stats, write_tdata)

WALLS, X_SIDES, Y_SIDES = 2, 4, 5  # markers: bottom and top, the x and y side walls


def mesh(Lx, Ly, Lz, res, **params):
    return BoxMesh(Point(0., 0., 0.), Point(Lx, Ly, Lz), int(round(Lx / res)),
                   int(round(Ly / res)), int(round(Lz / res)))


def constrained_domain(Lx, Ly, periodic=True, **params):
    return PeriodicBox({0: (0, Lx), 1: (0, Ly)}) if periodic else None


def problem_parameters(NS_parameters, **NS_namespace):
    NS_parameters.update(
        T=3.0,
        Lx=1.0,
        Ly=1.0,
        Lz=2.0,
        res=1. / 16,
        R=0.25,
        cz=0.5,          # the bubble's centre height at t = 0
        dt=0.002,
        rho=[1000., 100.],
        mu=[10., 1.],
        theta=np.pi / 2,
        epsilon=0.02,
        sigma=24.5,
        M=0.00002,
        F0=[0., 0., 0.],
        g0=[0., 0., -0.98],
        velocity_degree=2,
        periodic=True,   # periodic sides; False: free-slip side walls, as in the benchmark
        folder="bubble3d_results",
        plot_interval=10,
        stat_interval=10,
        timestamps_interval=10,
        save_step=10,
        checkpoint=10,
        print_intermediate_info=10,
        solver="BDF",
        max_iter=100,
        max_error=3e-9,
        iters_on_first_timestep=200,
    )


def mark_subdomains(subdomains, Lx, Ly, Lz, periodic=True, **NS_namespace):
    Faces(2, 0, Lz).mark(subdomains, WALLS)
    if not periodic:
        Faces(0, 0, Lx).mark(subdomains, X_SIDES)
        Faces(1, 0, Ly).mark(subdomains, Y_SIDES)


def contact_angles(theta, **NS_namespace):
    return [(theta, WALLS)]


def create_bcs(V, subdomains, periodic=True, **NS_namespace):
    walls = DirichletBC(V, 0, subdomains, WALLS)
    # Free slip on the sides: no flow through them, the tangential velocity free.
    x_sides = [] if periodic else [DirichletBC(V, 0, subdomains, X_SIDES)]
    y_sides = [] if periodic else [DirichletBC(V, 0, subdomains, Y_SIDES)]
    return dict(u0=[walls] + x_sides, u1=[walls] + y_sides, u2=[walls], p=[], phig=[])


def initialize(q_, q_1, q_2, x_1, x_2, bcs, epsilon, Lx, Ly, R, cz, **NS_namespace):
    set_subfunction(q_['phig'], 0, Expression(
        "tanh((sqrt(pow(x[0]-Lx/2, 2) + pow(x[1]-Ly/2, 2) + pow(x[2]-cz, 2)) - R)"
        "/(sqrt(2)*epsilon))", epsilon=epsilon, Lx=Lx, Ly=Ly, cz=cz, R=R, degree=2))
    q_1['phig'].vector()[:] = q_['phig'].vector()
    q_2['phig'].vector()[:] = q_['phig'].vector()
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(mesh, newfolder, **NS_namespace):
    return setup_stats(mesh, newfolder)


def temporal_hook(q_, tstep, t, u_, phi_, rho_, sigma, epsilon, volume, g0, statsfolder,
                  stat_interval, mesh, **NS_namespace):
    """Write to Stats/tdata.dat: tstep, t, the mean velocity (u0m, u1m, u2m) and phase
    field (phim), the energies per volume (kinetic, interface gradient and
    double-well parts, gravitational), and the bubble's (phi = -1) volume, height and
    rise velocity; and to Stats/benchmark.dat: tstep, t and the benchmark's bubble
    volume, centroid height, rise velocity and sphericity."""
    info_red("tstep = {}".format(tstep))
    if tstep % stat_interval == 0:
        entries = standard_stats(q_, u_, phi_, rho_, sigma, epsilon, volume, mesh)
        entries.append(gravitational_energy(rho_, g0, volume, mesh))
        vol_c, (_, _, z_cm), (_, _, V_c) = phase_moments(q_, mesh, dx(domain=mesh))
        write_tdata(statsfolder, tstep, t, entries + [vol_c, z_cm, V_c])
        values = hysing_quantities(q_, phi_, mesh, dx(domain=mesh))
        if MPI.rank(MPI.comm_world) == 0:
            append_row(path.join(statsfolder, "benchmark.dat"), tstep,
                       ("{:d} {:.8f} " + " ".join("{:.10e}" for _ in values) + "\n").format(
                           tstep, t, *values))
