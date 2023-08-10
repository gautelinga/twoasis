__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2023-08-10"
__copyright__ = "Copyright (C) 2023 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..TPfracStep import *
import matplotlib.pyplot as plt
import numpy as np
from os import makedirs

ccode = dict(
    u0 = "u0*exp(-2*mu*t/rho)*sin(x[1])*cos(x[0])",
    u1 = "-u0*exp(-2*mu*t/rho)*sin(x[0])*cos(x[1])",
    p = ("-0.25*rho*pow(u0, 2)*exp(-4*mu*t/rho)*(cos(2*x[0]) + cos(2*x[1]))"
         "-epsilon*pow(phi0, 2)*sigma*exp(-2*M*beta*sigma*t/epsilon)*pow(cos(x[0]), 2)*pow(cos(x[1]), 2)"
         "+(1.0/2.0)*pow(phi0, 2)*sigma*exp(-2*M*beta*sigma*t/epsilon)*pow(cos(x[0]), 2)*pow(cos(x[1]), 2)/epsilon"
         "-3.0/4.0*pow(phi0, 4)*sigma*exp(-4*M*beta*sigma*t/epsilon)*pow(cos(x[0]), 4)*pow(cos(x[1]), 4)/epsilon"),
    phi = "phi0*exp(-M*beta*sigma*t/epsilon)*cos(x[0])*cos(x[1])",
    q = "M*phi0*sigma*(-beta*exp(2*M*beta*sigma*t/epsilon) + 4*pow(epsilon, 2)*exp(2*M*beta*sigma*t/epsilon) - 6*pow(phi0, 2)*pow(sin(x[0]), 2)*pow(cos(x[1]), 2) - 6*pow(phi0, 2)*pow(sin(x[1]), 2)*pow(cos(x[0]), 2) + 6*pow(phi0, 2)*pow(cos(x[0]), 2)*pow(cos(x[1]), 2) - 2*exp(2*M*beta*sigma*t/epsilon))*exp(-3*M*beta*sigma*t/epsilon)*cos(x[0])*cos(x[1])/epsilon",
)

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
        L=2*np.pi,
        N=100,
        dt=0.01,
        rho=[1, 1],
        mu=[1, 1],
        u0=1.0,
        phi0=0.0,
        epsilon=0.2,
        sigma=1.0,
        M=0.1,
        beta=2.0,
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
        solver="BDF")

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

def phase_field_source(t, u0, phi0, rho, mu, sigma, M, epsilon, beta, exprs, **NS_namespace):
    return exprs["q"]

def initialize(q_, q_1, q_2, VV, t, dt, u0, phi0, rho, mu, sigma, M, epsilon, beta, **NS_namespace):
    rho0 = np.mean(rho)
    mu0 = np.mean(mu)
    sigma_bar = sigma * 3./(2*np.sqrt(2)) # subtlety!

    exprs = dict([(key, Expression(code, u0=u0, phi0=phi0, rho=rho0, mu=mu0, sigma=sigma_bar, M=M, epsilon=epsilon, beta=beta, t=0, degree=6)) for key, code in ccode.items()]) 
    function_spaces = dict(
        u0=VV['u0'],
        u1=VV['u0'],
        p=VV['p'],
        phi=VV['phig'].sub(0).collapse())
    
    # Initialize time derivatives for 2nd order accuracy
    for expr in exprs.values():
        expr.t = t-2*dt
    q_init_2 = dict([(key, interpolate(exprs[key], space)) for key, space in function_spaces.items()])
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
        q_1[key].vector()[:] = q_init_1[key].vector()
        q_2[key].vector()[:] = q_init_2[key].vector()
    assign(q_['phig'].sub(0), q_init_['phi'])
    assign(q_1['phig'].sub(0), q_init_1['phi'])
    assign(q_2['phig'].sub(0), q_init_2['phi'])
    
    return dict(exprs=exprs)
    
def pre_solve_hook(tstep, t, q_, p_, mesh, u_, newfolder, rho_, phi_, sigma_bar, velocity_degree, pressure_degree, AssignedVectorFunction, 
                   F0, g0, u0, phi0, mu, rho, sigma, M, epsilon, beta, dt, **NS_namespace):
    statsfolder = path.join(newfolder, "Stats")
    timestampsfolder = path.join(newfolder, "Timestamps")

    # Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree)
    uv = AssignedVectorFunction(u_, name="u")

    if MPI.rank(MPI.comm_world) == 0 and not path.exists(statsfolder):
        makedirs(statsfolder)

    if MPI.rank(MPI.comm_world) == 0 and not path.exists(timestampsfolder):
        makedirs(timestampsfolder)

    E_kin = 0.5*assemble(rho_ * (u_[0]**2 + u_[1]**2) * dx)
    E_int = sigma_bar * (
        0.5 * epsilon * assemble((phi_.dx(0)**2 + phi_.dx(1)**2) * dx) 
        + 0.25 / epsilon * assemble((1-phi_**2)**2 * dx))
    if MPI.rank(MPI.comm_world) == 0:
        print("{0:2.6e}".format(E_kin))
        with open(statsfolder + "/tdata.dat", "a") as tfile:
            tfile.write("{:d} {:.8f} {:.8f} {:.8f}\n".format(
                tstep, t, E_kin, E_int))

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
        E_kin = 0.5*assemble(rho_ * (u_[0]**2 + u_[1]**2) * dx)
        E_int = sigma_bar * (
            0.5 * epsilon * assemble((phi_.dx(0)**2 + phi_.dx(1)**2) * dx) 
            + 0.25 / epsilon * assemble((1-phi_**2)**2 * dx))
        if MPI.rank(MPI.comm_world) == 0:
            print("{0:2.6e}".format(E_kin))
            with open(statsfolder + "/tdata.dat", "a") as tfile:
                tfile.write("{:d} {:.8f} {:.8f} {:.8f}\n".format(
                    tstep, t, E_kin, E_int))
        pass

    if tstep % timestamps_interval == 0:
        #write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)
        pass
    
    return dict()


def theend_hook(t, q_, u_, p_, testing, exprs, **NS_namespace):
    norm_type = dict()

    for key, expr in exprs.items():
        expr.t = t
        norm_type[key] = "L2"
    norm_type['p'] = "H10"

    errs = dict()
    for key in ['u0', 'u1', 'p']:
        errs[key] = errornorm(exprs[key], q_[key], norm_type=norm_type[key])
    errs['phi'] = errornorm(exprs['phi'], q_['phig'].sub(0), norm_type="L2")

    if MPI.rank(MPI.comm_world) == 0: # and testing:
        #print("Velocity norm = {0:2.6e}".format(u_norm))
        print(errs)
