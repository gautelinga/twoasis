"""A decaying Taylor-Green vortex in a periodic square, compared with its exact
(manufactured) solution. phi0 > 0 adds a decaying phase field of amplitude phi0,
kept exact by a source term; with the default phi0 = 0 the flow is single-phase."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2023-08-10"
__copyright__ = ("Copyright (C) 2023 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"

from os import path

import numpy as np
from dolfin import (
    assemble, dx, Expression, interpolate, MPI, Point, RectangleMesh)

from twoasis.common.utilities import set_subfunction, subspace
from twoasis.problems import info_red
from twoasis.problems.TPfracStep.shared import (
    centred, interface_energy, makedirs_on_root, PeriodicBox, write_tdata)

ccode = dict(
    u0 = "u0*exp(-8*pow(M_PI, 2)*mu*t/(pow(L, 2)*rho))*sin(2*x[1]*M_PI/L)*cos(2*x[0]*M_PI/L)",
    u1 = "-u0*exp(-8*pow(M_PI, 2)*mu*t/(pow(L, 2)*rho))*sin(2*x[0]*M_PI/L)*cos(2*x[1]*M_PI/L)",
    p = "-(0.25*pow(L, 2)*epsilon*rho*pow(u0, 2)*(cos(4*x[0]*M_PI/L) + cos(4*x[1]*M_PI/L))*exp(16*pow(M_PI, 2)*M*beta*sigma*t/(pow(L, 2)*epsilon)) + (1.0/256.0)*pow(phi0, 2)*sigma*(pow(L, 2)*pow(phi0, 2)*(192*pow(cos(2*x[0]*M_PI/L), 4)*pow(cos(2*x[1]*M_PI/L), 4) - 27) - 32*(pow(L, 2) - 8*pow(M_PI, 2)*pow(epsilon, 2))*(4*pow(cos(2*x[0]*M_PI/L), 2)*pow(cos(2*x[1]*M_PI/L), 2) - 1)*exp(8*pow(M_PI, 2)*M*beta*sigma*t/(pow(L, 2)*epsilon)))*exp(16*pow(M_PI, 2)*mu*t/(pow(L, 2)*rho)))*exp(-16*pow(M_PI, 2)*t*(M*beta*rho*sigma + epsilon*mu)/(pow(L, 2)*epsilon*rho))/(pow(L, 2)*epsilon)",
    phi = "phi0*exp(-4*pow(M_PI, 2)*M*beta*sigma*t/(pow(L, 2)*epsilon))*cos(2*x[0]*M_PI/L)*cos(2*x[1]*M_PI/L)",
    q = "4*pow(M_PI, 2)*M*phi0*sigma*(-pow(L, 2)*beta*exp(8*pow(M_PI, 2)*M*beta*sigma*t/(pow(L, 2)*epsilon)) - 6*pow(L, 2)*pow(phi0, 2)*pow(sin(2*x[0]*M_PI/L), 2)*pow(cos(2*x[1]*M_PI/L), 2) - 6*pow(L, 2)*pow(phi0, 2)*pow(sin(2*x[1]*M_PI/L), 2)*pow(cos(2*x[0]*M_PI/L), 2) + 6*pow(L, 2)*pow(phi0, 2)*pow(cos(2*x[0]*M_PI/L), 2)*pow(cos(2*x[1]*M_PI/L), 2) - 2*pow(L, 2)*exp(8*pow(M_PI, 2)*M*beta*sigma*t/(pow(L, 2)*epsilon)) + 16*pow(M_PI, 2)*pow(epsilon, 2)*exp(8*pow(M_PI, 2)*M*beta*sigma*t/(pow(L, 2)*epsilon)))*exp(-12*pow(M_PI, 2)*M*beta*sigma*t/(pow(L, 2)*epsilon))*cos(2*x[0]*M_PI/L)*cos(2*x[1]*M_PI/L)/(pow(L, 4)*epsilon)",
)

def compute_errors(t, exprs, q_):
    """The L2 errors of u0, u1, p and phi against the exact solution at t (the
    Expressions are of degree 6). Assembled directly, which builds no new space."""
    for expr in exprs.values():
        expr.t = t
    fields = dict(u0=q_['u0'], u1=q_['u1'], p=q_['p'], phi=q_['phig'].sub(0))
    return {key: np.sqrt(assemble((u - exprs[key])**2 * dx)) for key, u in fields.items()}

def dump_stats(sigma, epsilon, rho_, u_, phi_, statsfolder, tstep, t, q_, exprs, mesh, **namespace):
    E_kin = 0.5*assemble(rho_ * (u_[0]**2 + u_[1]**2) * dx(domain=mesh))
    E_int = sum(interface_energy(phi_, sigma, epsilon, dx(domain=mesh)))

    errs = compute_errors(t, exprs, q_)

    write_tdata(statsfolder, tstep, t,
                [E_kin, E_int, errs["u0"], errs["u1"], errs["p"], errs["phi"]])


# Create a mesh
def mesh(L, N, **params):
    mesh = RectangleMesh(Point(-L/2, -L/2), Point(L/2, L/2), N, N)
    return mesh

# Override some problem specific parameters
def problem_parameters(NS_parameters, **NS_namespace):
    NS_parameters.update(
        T=1.0,
        L=2.0,
        N=80,
        dt=0.001,
        rho=[1.0, 1.0],
        mu=[0.01, 0.01],
        u0=1.0,
        phi0=0.0,
        epsilon=0.2,
        sigma=1.0,
        M=0.1,
        beta0=2.0,
        F0=[0., 0.],
        g0=[0., 0.],
        velocity_degree=1,
        folder="taylorgreen2d_results",
        plot_interval=10,
        stat_interval=10,
        timestamps_interval=10,
        save_step=10,
        checkpoint=10,
        print_intermediate_info=10,
        use_krylov_solvers=True,
        solver="BDF",
        max_iter=10,
        max_error=1e-6,
        iters_on_first_timestep=20,
        AB_projection_pressure=False,)


def constrained_domain(L, **params):
    return PeriodicBox({0: centred(L), 1: centred(L)})

def mark_subdomains(**NS_namespace):
    return dict()


# Specify boundary conditions
def create_bcs(V, subdomains, **NS_namespace):
    return dict(u0=[], u1=[], p=[], phig=[])


def phase_field_source(exprs, **NS_namespace):
    return exprs["q"]

def initialize(q_, q_1, q_2, VV, t, dt, u0, phi0, rho, mu, sigma, M, epsilon, beta0, L, **NS_namespace):
    rho0 = np.mean(rho)
    mu0 = np.mean(mu)
    sigma_bar = sigma * 3./(2*np.sqrt(2)) # subtlety!

    exprs = dict([(key, Expression(code, u0=u0, phi0=phi0, rho=rho0, mu=mu0, sigma=sigma_bar, M=M, epsilon=epsilon, beta=beta0, L=L, t=0, degree=6)) for key, code in ccode.items()]) 
    function_spaces = dict(
        u0=VV['u0'],
        u1=VV['u0'],
        p=VV['p'],
        phi=subspace(VV['phig'], 0))
    
    # Initialize time derivatives for 2nd order accuracy
    for expr in exprs.values():
        expr.t = t-dt
    q_init_1 = dict([(key, interpolate(exprs[key], space)) for key, space in function_spaces.items()])
    for expr in exprs.values():
        expr.t = t
    q_init_ = dict([(key, interpolate(exprs[key], space)) for key, space in function_spaces.items()])

    for key in ['u0', 'u1', 'p']:
        q_[key].vector()[:] = q_init_[key].vector()
        q_1[key].vector()[:] = q_init_[key].vector()
        q_2[key].vector()[:] = q_init_1[key].vector()
    set_subfunction(q_['phig'], 0, q_init_['phi'])
    set_subfunction(q_1['phig'], 0, q_init_['phi'])
    set_subfunction(q_2['phig'], 0, q_init_1['phi'])
    
    return dict(exprs=exprs)
    
def pre_solve_hook(tstep, t, q_, mesh, u_, newfolder, rho_, phi_, sigma, epsilon, exprs,
                   **NS_namespace):
    statsfolder = path.join(newfolder, "Stats")
    makedirs_on_root(statsfolder)
    dump_stats(**vars())
    return dict(statsfolder=statsfolder)


def temporal_hook(q_, tstep, t, u_, phi_, rho_, sigma, epsilon, statsfolder, stat_interval,
                  exprs, mesh, **NS_namespace):
    """Write to Stats/tdata.dat: tstep, t, the kinetic and interface energies,
    and the L2 errors of u0, u1, p and phi."""
    info_red("tstep = {}".format(tstep))
    if tstep % stat_interval == 0:
        dump_stats(**vars())

def theend_hook(t, q_, u_, p_, testing, exprs, **NS_namespace):
    errs = compute_errors(t, exprs, q_)

    if MPI.rank(MPI.comm_world) == 0: # and testing:
        #print("Velocity norm = {0:2.6e}".format(u_norm))
        print(errs)
