"""Laplace pressure of a bubble at rest.

A circular bubble (phi = -1) of radius R sits in a square of side L with
no-slip walls. The run stops at steady state and writes it to
Stats/steady.dat:  t dP sigma/R dP/(sigma/R)-1 max|u| R_eq dP/(sigma/R_eq)-1
(Stats/tdata.dat holds the time series).

- dP is the jump in the physical pressure p + phi*g between the bubble interior
  and the far field; p itself is constant at equilibrium.
- R_eq is the radius the bubble settles at: the bulk phase field shifts
  slightly from +-1, so the bubble shrinks (first order in epsilon/R, more for
  small bubbles in a large box). dP matches sigma/R_eq to 0.1%.
- max|u| at steady state reflects the solver tolerances only: at rest, g is
  constant and the force -phi*grad(g) vanishes. Moving interfaces are untested.
- Steady state: dP and max|u| change by less than steady_tol over
  steady_window. M sets only how fast this is reached.
- Time step: dt < sqrt(rho*h**3/(2*pi*sigma)) (explicit surface tension); a very
  viscous bubble needs dt ~ 1/(mu_in - mu_out).

The same test (a bubble in the unit square, rho = mu = sigma = 1, dP against
sigma/R) is the static bubble of A. Lovrić, W. G. Dettmer and D. Perić, Low order
finite element methods for the Navier-Stokes-Cahn-Hilliard equations,
arXiv:1911.06718 (2019).
"""
__author__ = "Gaute Linga <gaute.linga@ntnu.no>"
__date__ = "2026-11-02"
__copyright__ = ("Copyright (C) 2026 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"

from os import path

import numpy as np
from dolfin import (
    assemble, conditional, DirichletBC, dx, Expression, gt, lt, MPI, SpatialCoordinate, sqrt,
    SubDomain)

from twoasis.common.utilities import set_subfunction
from twoasis.problems import info_green
from twoasis.problems.TPfracStep.shared import append_row, rectangle_mesh, setup_stats


def mesh(L, res, **params):
    return rectangle_mesh(L, L, res)


def problem_parameters(NS_parameters, **NS_namespace):
    NS_parameters.update(
        T=2.0,           # upper limit; the run stops at steady state
        dt=5e-4,         # below the capillary limit, ~8e-4 for res = 1/64
        L=1.0,
        R=0.25,
        res=1./64,
        rho=[1., 1.],
        mu=[1., 1.],
        sigma=1.0,
        epsilon=0.02,
        M=0.1,           # large, so the interface relaxes quickly
        steady_tol=1e-4,
        steady_window=0.1,
        theta=np.pi/2,
        F0=[0., 0.],
        g0=[0., 0.],
        velocity_degree=2,
        solver="BDF",
        folder="laplacebubble2d_results",
        plot_interval=10,
        stat_interval=10,
        timestamps_interval=10,
        save_step=10,
        checkpoint=10,
        print_intermediate_info=10,
        max_iter=100,
        max_error=1e-6,
        iters_on_first_timestep=200,
    )


class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


def mark_subdomains(subdomains, **NS_namespace):
    Walls().mark(subdomains, 1)
    return dict()


def contact_angles(**NS_namespace):
    return []  # the bubble does not touch the walls


def create_bcs(V, subdomains, **NS_namespace):
    bc_u = DirichletBC(V, 0, subdomains, 1)
    return dict(u0=[bc_u], u1=[bc_u], p=[], phig=[])


def initialize(q_, q_1, q_2, x_1, x_2, bcs, epsilon, VV, L, R, **NS_namespace):
    set_subfunction(q_['phig'], 0, Expression(
        "tanh((sqrt(pow(x[0]-L/2, 2)+pow(x[1]-L/2, 2))-R)/(sqrt(2)*epsilon))",
        epsilon=epsilon, L=L, R=R, degree=2))
    q_1['phig'].vector()[:] = q_['phig'].vector()
    q_2['phig'].vector()[:] = q_['phig'].vector()


def pre_solve_hook(mesh, newfolder, L, R, epsilon, **NS_namespace):
    # Regions where the phases are pure, away from the diffuse interface.
    X = SpatialCoordinate(mesh)
    r = sqrt((X[0] - L/2)**2 + (X[1] - L/2)**2)
    inside = conditional(lt(r, R - 5*epsilon), 1., 0.)
    outside = conditional(gt(r, R + 5*epsilon), 1., 0.)
    area_in = assemble(inside * dx(domain=mesh))
    area_out = assemble(outside * dx(domain=mesh))
    return dict(inside=inside, outside=outside, area_in=area_in, area_out=area_out,
                laplace_history=[], **setup_stats(mesh, newfolder))


def temporal_hook(q_, tstep, t, dx, u_, p_, phi_, g_, rho_, sigma, R, statsfolder,
                  stat_interval, inside, outside, area_in, area_out, volume,
                  laplace_history, steady_tol, steady_window, **NS_namespace):
    if tstep % stat_interval == 0:
        P = p_ + phi_ * g_
        P_in = assemble(P * inside * dx) / area_in
        P_out = assemble(P * outside * dx) / area_out
        speed2 = q_['u0'].vector().get_local()**2 + q_['u1'].vector().get_local()**2
        u_max = np.sqrt(MPI.max(MPI.comm_world, speed2.max() if speed2.size else 0.))
        E_kin = 0.5 * assemble(rho_ * (u_[0]**2 + u_[1]**2) * dx) / volume
        phi_mean = assemble(phi_ * dx) / volume
        bubble_area = assemble(conditional(lt(phi_, 0.), 1., 0.)
                               * dx(metadata={"quadrature_degree": 8}))
        R_eq = np.sqrt(bubble_area / np.pi)
        phi_in = assemble(phi_ * inside * dx) / area_in
        phi_out = assemble(phi_ * outside * dx) / area_out
        if MPI.rank(MPI.comm_world) == 0:
            entries = [P_in - P_out, sigma / R, P_in, P_out, u_max, E_kin, phi_mean,
                       R_eq, phi_in, phi_out]
            append_row(path.join(statsfolder, "tdata.dat"), tstep,
                       ("{:d} {:.8f} " + " ".join("{:.10e}" for _ in entries) + "\n").format(
                           tstep, t, *entries))

        dP = P_in - P_out
        laplace_history.append((t, dP, u_max))
        if is_steady(laplace_history, steady_tol, steady_window):
            error = dP / (sigma / R) - 1
            laplace_error = dP / (sigma / R_eq) - 1
            info_green("Steady state at t = {:.4f}: dP = {:.6f}, sigma/R = {:.6f} (error {:.3%}), "
                       "R_eq = {:.5f} (Laplace error {:.3%}), max|u| = {:.3e}".format(
                           t, dP, sigma / R, error, R_eq, laplace_error, u_max))
            if MPI.rank(MPI.comm_world) == 0:
                with open(path.join(statsfolder, "steady.dat"), "w") as sfile:
                    sfile.write("{:.8f} {:.10e} {:.10e} {:.10e} {:.10e} {:.10e} {:.10e}\n".format(
                        t, dP, sigma / R, error, u_max, R_eq, laplace_error))
            return dict(stop=True)


def is_steady(history, tol, window):
    """Whether dP and max|u| changed by less than tol (relative) over the last
    window time units. history holds (t, dP, u_max) samples, oldest first."""
    t, dP, u_max = history[-1]
    earlier = [h for h in history if t - h[0] >= window - 1e-12]
    if not earlier:
        return False
    _, dP_0, u_max_0 = earlier[-1]
    return (abs(dP - dP_0) <= tol * abs(dP)
            and abs(u_max - u_max_0) <= tol * max(u_max, 1e-300))
