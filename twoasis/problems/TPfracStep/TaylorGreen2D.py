__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2023-08-10"
__copyright__ = "Copyright (C) 2023 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..TPfracStep import *
import matplotlib.pyplot as plt
import numpy as np
from os import makedirs

ccode = dict(
    u0 = "u0*exp(-8*pow(M_PI, 2)*mu*t/(pow(L, 2)*rho))*sin(2*x[1]*M_PI/L)*cos(2*x[0]*M_PI/L)",
    u1 = "-u0*exp(-8*pow(M_PI, 2)*mu*t/(pow(L, 2)*rho))*sin(2*x[0]*M_PI/L)*cos(2*x[1]*M_PI/L)",
    p = "-(0.25*pow(L, 2)*epsilon*rho*pow(u0, 2)*(cos(4*x[0]*M_PI/L) + cos(4*x[1]*M_PI/L))*exp(16*pow(M_PI, 2)*M*beta*sigma*t/(pow(L, 2)*epsilon)) + (1.0/256.0)*pow(phi0, 2)*sigma*(pow(L, 2)*pow(phi0, 2)*(192*pow(cos(2*x[0]*M_PI/L), 4)*pow(cos(2*x[1]*M_PI/L), 4) - 27) - 32*(pow(L, 2) - 8*pow(M_PI, 2)*pow(epsilon, 2))*(4*pow(cos(2*x[0]*M_PI/L), 2)*pow(cos(2*x[1]*M_PI/L), 2) - 1)*exp(8*pow(M_PI, 2)*M*beta*sigma*t/(pow(L, 2)*epsilon)))*exp(16*pow(M_PI, 2)*mu*t/(pow(L, 2)*rho)))*exp(-16*pow(M_PI, 2)*t*(M*beta*rho*sigma + epsilon*mu)/(pow(L, 2)*epsilon*rho))/(pow(L, 2)*epsilon)",
    phi = "phi0*exp(-4*pow(M_PI, 2)*M*beta*sigma*t/(pow(L, 2)*epsilon))*cos(2*x[0]*M_PI/L)*cos(2*x[1]*M_PI/L)",
    q = "4*pow(M_PI, 2)*M*phi0*sigma*(-pow(L, 2)*beta*exp(8*pow(M_PI, 2)*M*beta*sigma*t/(pow(L, 2)*epsilon)) - 6*pow(L, 2)*pow(phi0, 2)*pow(sin(2*x[0]*M_PI/L), 2)*pow(cos(2*x[1]*M_PI/L), 2) - 6*pow(L, 2)*pow(phi0, 2)*pow(sin(2*x[1]*M_PI/L), 2)*pow(cos(2*x[0]*M_PI/L), 2) + 6*pow(L, 2)*pow(phi0, 2)*pow(cos(2*x[0]*M_PI/L), 2)*pow(cos(2*x[1]*M_PI/L), 2) - 2*pow(L, 2)*exp(8*pow(M_PI, 2)*M*beta*sigma*t/(pow(L, 2)*epsilon)) + 16*pow(M_PI, 2)*pow(epsilon, 2)*exp(8*pow(M_PI, 2)*M*beta*sigma*t/(pow(L, 2)*epsilon)))*exp(-12*pow(M_PI, 2)*M*beta*sigma*t/(pow(L, 2)*epsilon))*cos(2*x[0]*M_PI/L)*cos(2*x[1]*M_PI/L)/(pow(L, 4)*epsilon)",
)

def error_norm(expr, u, norm_type='l2'):
    S = u.function_space()
    is_subspace = len(S.component())
    ue_ = interpolate(expr, S.collapse() if is_subspace else S)
    if is_subspace:
        u_ = Function(S.collapse())
        assign(u_, u)
        ue_.vector()[:] -= u_.vector()
    else:
        ue_.vector()[:] -= u.vector()
    return norm(ue_, norm_type=norm_type)

def compute_errors(t, exprs, q_):
    for expr in exprs.values():
        expr.t = t
    errs = dict()
    for key in ['u0', 'u1', 'p']:
        #ue_ = interpolate(exprs[key], q_[key].function_space())
        errs[key] = errornorm(exprs[key], q_[key])
        #errs[key] = errornorm(ue_, q_[key])
    #ue_ = interpolate(exprs['phi'], q_['phig'].sub(0).function_space().collapse())
    errs['phi'] = errornorm(exprs['phi'], q_['phig'].sub(0))
    #errs['phi'] = errornorm(ue_, q_['phig'].sub(0))
    return errs

def dump_stats(sigma_bar, epsilon, rho_, u_, phi_, statsfolder, tstep, t, q_, exprs, **namespace):
    E_kin = 0.5*assemble(rho_ * (u_[0]**2 + u_[1]**2) * dx)
    E_int = sigma_bar * (
        0.5 * epsilon * assemble((phi_.dx(0)**2 + phi_.dx(1)**2) * dx) 
        + 0.25 / epsilon * assemble((1-phi_**2)**2 * dx))
    
    errs = compute_errors(t, exprs, q_)

    if MPI.rank(MPI.comm_world) == 0:
        #print("{0:2.6e}".format(E_kin))
        items = [tstep, t, E_kin, E_int, errs["u0"], errs["u1"], errs["p"], errs["phi"]]
        with open(statsfolder + "/tdata.dat", "a") as tfile:
            tfile.write("{:d} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n".format(*items))

class PBC(SubDomain):
    def __init__(self, Lx, Ly):
        self.Lx = Lx
        self.Ly = Ly
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two slave edges
        return bool(
            (near(x[0], -self.Lx / 2) or near(x[1], -self.Ly / 2))
            and (not (near(x[0], self.Lx / 2) or near(x[1], self.Ly / 2)))
            and on_boundary)

    def map(self, x, y):
        if near(x[0], self.Lx / 2) and near(x[1], self.Ly / 2):
            y[0] = x[0] - self.Lx
            y[1] = x[1] - self.Ly
        elif near(x[0], self.Lx / 2):
            y[0] = x[0] - self.Lx
            y[1] = x[1]
        else:  # near(x[2], Lz/2.):
            y[0] = x[0]
            y[1] = x[1] - self.Ly

# Create a mesh
def mesh(L, N, **params):
    mesh = RectangleMesh(Point(-L/2, -L/2), Point(L/2, L/2), N, N)
    return mesh

# Override some problem specific parameters
def problem_parameters(NS_parameters, NS_expressions, commandline_kwargs, **NS_namespace):
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

    NS_expressions.update(dict(
        constrained_domain=PBC(NS_parameters["L"], NS_parameters["L"])
    ))

def mark_subdomains(**NS_namespace):
    return dict()

def contact_angles(**NS_namespace):
    return []

# Specify boundary conditions
def create_bcs(V, subdomains, **NS_namespace):
    return dict(u0=[], u1=[], p=[], phig=[])

def average_pressure_gradient(F0, **NS_namespace):
    # average pressure gradient
    return Constant(tuple(F0))

def acceleration(g0, **NS_namespace):
    # (gravitational) acceleration
    return Constant(tuple(g0))

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
        phi=VV['phig'].sub(0).collapse())
    
    # Initialize time derivatives for 2nd order accuracy
    for expr in exprs.values():
        expr.t = t-dt
    q_init_1 = dict([(key, interpolate(exprs[key], space)) for key, space in function_spaces.items()])
    for expr in exprs.values():
        expr.t = t
    q_init_ = dict([(key, interpolate(exprs[key], space)) for key, space in function_spaces.items()])

    #u0_ = interpolate(expr["u0"], VV['u0'])
    #u1_ = interpolate(expr["u1"], VV['u0'])
    #p_ = interpolate(expr["p"], VV['p'])
    #phig_ = interpolate(expr["phi"], VV['phig'].sub(0).collapse())
    
    #q_['u0'].vector()[:] = u0_.vector()
    #q_['u1'].vector()[:] = u1_.vector()
    #q_['p'].vector()[:] = p_.vector()
    
    for key in ['u0', 'u1', 'p']:
        q_[key].vector()[:] = q_init_[key].vector()
        q_1[key].vector()[:] = q_init_[key].vector()
        q_2[key].vector()[:] = q_init_1[key].vector()
    assign(q_['phig'].sub(0), q_init_['phi'])
    assign(q_1['phig'].sub(0), q_init_['phi'])
    assign(q_2['phig'].sub(0), q_init_1['phi'])
    
    return dict(exprs=exprs)
    
def pre_solve_hook(tstep, t, q_, p_, mesh, u_, newfolder, rho_, phi_, sigma_bar, velocity_degree, pressure_degree, AssignedVectorFunction, 
                   F0, g0, u0, phi0, mu, rho, sigma, M, epsilon, exprs, **NS_namespace):
    statsfolder = path.join(newfolder, "Stats")
    timestampsfolder = path.join(newfolder, "Timestamps")

    # Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree)
    uv = AssignedVectorFunction(u_, name="u")

    if MPI.rank(MPI.comm_world) == 0 and not path.exists(statsfolder):
        makedirs(statsfolder)

    if MPI.rank(MPI.comm_world) == 0 and not path.exists(timestampsfolder):
        makedirs(timestampsfolder)

    dump_stats(**vars())

    with HDF5File(mesh.mpi_comm(),
                  path.join(timestampsfolder, "mesh.h5"), "w") as h5f:
        h5f.write(mesh, "mesh")
    #write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)

    return dict(uv=uv, statsfolder=statsfolder, timestampsfolder=timestampsfolder)

def temporal_hook(q_, tstep, t, dx, u_, p_, phi_, rho_,
                  sigma_bar, epsilon, statsfolder, timestampsfolder,
                  plot_interval, stat_interval, timestamps_interval, exprs,
                  uv, mesh,
                  **NS_namespace):
    info_red("tstep = {}".format(tstep))

    if tstep % stat_interval == 0:
        dump_stats(**vars())
        pass

    if tstep % timestamps_interval == 0:
        #write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)
        pass
    
    return dict()

def theend_hook(t, q_, u_, p_, testing, exprs, **NS_namespace):
    errs = compute_errors(t, exprs, q_)

    if MPI.rank(MPI.comm_world) == 0: # and testing:
        #print("Velocity norm = {0:2.6e}".format(u_norm))
        print(errs)
