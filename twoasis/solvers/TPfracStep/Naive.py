__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-20"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
"""This is a simplest possible naive implementation of the Chorin solver.

The idea is that this solver can be quickly modified and tested for
alternative implementations. In the end it can be used to validate
the implementations of the more complex optimized solvers.

"""
from dolfin import *
from . import *
from . import __all__
import math

__all__ += ["max_iter", "iters_on_first_timestep"]

# Chorin is noniterative
max_iter = 2
iters_on_first_timestep = 10

def setup(u, q_, q_1, uc_comp, u_components, dt, v, U_AB, u_1, rho, mu,
          nu, p_, p_1, dp_, mesh, f, fs, q, p, u_, Schmidt, M, phi, g, phi_, g_, g_1, phi_1, phi_2, xi, eta, sigma, epsilon,
          scalar_components, ct, dim, **NS_namespace):
    """Set up all equations to be solved."""
    # Implicit Crank Nicholson velocity at t - dt/2
    U_CN = dict((ui, 0.5 * (u + q_1[ui])) for ui in uc_comp)
    phi_CN = 0.5 * (phi + phi_1)
    g_CN = 0.5 * (g + g_1)
    #u_cn = 0.5 * (u + u_1)

    sigma_bar = sigma * 3./(2*math.sqrt(2))

    rho_ = weighted_arithmetic_mean(rho, phi_)
    mu_ = weighted_geometric_mean(rho, phi_)

    mom_1 = rho_ * u_1 # - M * drho_ * grad(g_)

    F = {}
    Fu = {}
    for i, ui in enumerate(u_components):
        # Tentative velocity step
        #u_cn = 0.5*(u+q_1[ui])
        #u_cn_ = 0.5*(u_+u_1)

        F[ui] = ((1. / dt) * rho_ * inner(u - q_1[ui], v) * dx
                 + inner(dot(mom_1, nabla_grad(u)), v) * dx
                 + mu_ * inner(grad(u), grad(v)) * dx
                 - inner(inner(grad(mu_), u_1.dx(i)), v) * dx
                 + p_.dx(i) * v * dx
                 - inner(f[i], v) * dx
                 + inner(phi_ * g_.dx(i), v) * dx)

        # Velocity update
        Fu[ui] = (rho_ * inner(u - q_[ui], v) * dx + dt * inner(dp_.dx(i), v) * dx)
    """
    F['u'] = ((1. / dt) * rho_ * inner(u - u_1, v) * dx
            + inner(dot(mom_1, nabla_grad(u)), v) *                                                             
            #+ 2 * mu_ * inner(sym(grad(u)), sym(grad(v))) * dx
            + mu_ * inner(grad(u), grad(v)) * dx
            + sum([inner(inner(grad(mu_), U_AB.dx(i)), v[i]) * dx for i in range(dim)])
            - p_1 * div(v) * dx
            - inner(f, v) * dx
            + inner(phi_ * grad(g_), v) * dx)

    # Velocity update
    Fu['u'] = rho_ * inner(u - u_, v) * dx - dt * inner(p_ - p_1, div(v)) * dx
    """

    # Pressure solve
    Fp = 1./rho_ * inner(grad(q), grad(p - p_)) * dx + (1. / dt) * div(u_) * q * dx
    # min(rho) 

    # Scalar with SUPG
    h = CellDiameter(mesh)
    #vw = v + h * inner(grad(v), U_AB)
    vw = ct + h * inner(grad(ct), U_AB)
    n = FacetNormal(mesh)

    # Phase field solve
    F['phig'] = ((1. / dt) * inner(phi_ - phi_1, xi) * dx
                 #+ inner(dot(grad(phi_CN), U_AB), xi) * dx
                 #- inner(dot(grad(xi), u_), phi_ab) * dx
                 - inner(dot(grad(xi), u_), phi_) * dx
                 + M * inner(grad(g_), grad(xi)) * dx
                 + g_ * eta * dx
                 - sigma_bar * epsilon * dot(grad(phi_), grad(eta)) * dx
                 #- sigma_bar * epsilon * dot(grad(phi_cn), grad(eta)) * dx
                 - sigma_bar / epsilon * potential_linear_derivative(phi_, phi_1) * eta * dx
                 #- sigma_bar / epsilon * potential_derivative_approx(phi_, phi_1, phi_2) * eta * dx
                 )

    for ci in scalar_components:
        F[ci] = ((1. / dt) * inner(u - q_1[ci], vw) * dx
                 + inner(dot(grad(U_CN[ci]), U_AB), vw) * dx
                 + nu / Schmidt[ci] * inner(grad(U_CN[ci]),
                            grad(vw)) * dx - inner(fs[ci], vw) * dx)

    return dict(F=F, Fu=Fu, Fp=Fp)


def velocity_tentative_solve(ui, F, q_, bcs, x_, b_tmp, udiff, **NS_namespace):
    """Linear algebra solve of tentative velocity component."""
    b_tmp[ui][:] = x_[ui]
    A, L = system(F[ui])
    solve(A == L, q_[ui], bcs[ui])
    udiff[0] += norm(b_tmp[ui] - x_[ui])


def pressure_solve(x_, Fp, dp_, p_, bcs, **NS_namespace):
    """Solve pressure equation."""
    dp_.vector()[:] = x_['p']
    solve(lhs(Fp) == rhs(Fp), p_, bcs['p'])
    if bcs['p'] == []:
        normalize(p_.vector())
    dpv = dp_.vector()
    dpv *= -1
    dpv.axpy(1.0, x_['p'])

def velocity_update(u_components, q_, bcs, Fu, **NS_namespace):
    """Update the velocity after finishing pressure velocity iterations."""
    for ui in u_components:
        solve(lhs(Fu[ui]) == rhs(Fu[ui]), q_[ui], bcs[ui])


def scalar_solve(ci, F, q_, bcs, **NS_namespace):
    """Solve scalar equation."""
    solve(lhs(F[ci]) == rhs(F[ci]), q_[ci], bcs[ci])


def phase_field_solve(F, q_, bcs, **NS_namespace):
    """Solve scalar equation."""
    #solve(lhs(F['phig']) == rhs(F['phig']), q_['phig'], bcs['phig'])
    solve(F['phig'] == 0, q_['phig'], bcs['phig'])