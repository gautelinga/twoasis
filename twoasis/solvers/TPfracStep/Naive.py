"""Plain implementation of the first-order scheme of IPCS.py, for validating the
optimized solvers.

Every equation is written as one UFL form and solved with dolfin's solve()
(LU), so the scheme can be read off directly. Each timestep solves:

  1. the Cahn-Hilliard equations for (phi, g), with phi advected by u_1 and the
     double-well potential linearized about phi_1;
  2. the tentative velocity u*, one component at a time;
  3. the pressure Poisson equation (steps 2 and 3 repeat for the inner
     iterations);
  4. the velocity correction u = u* - dt/rho grad(dp), dp being the pressure
     change of the last inner iteration.

The discretization is the one of IPCS.py, so the two agree to the tolerance of
IPCS's Krylov solvers. The density and the viscosity are the same nodal fields
on the velocity space as in BDF.py and IPCS.py (set_mixed_fields), and the
pressure-gradient correction is projected onto the velocity space as in IPCS.py.
"""
__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-20"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
# Modified for two-phase flow by Gaute Linga, 2022-2026.

import math

from dolfin import (
    as_backend_type, assemble, div, dot, dx, Function, grad, inner, lhs, normalize,
    PETScKrylovSolver, rhs, solve, Vector, VectorSpaceBasis)

from twoasis.common.utilities import homogenize
from twoasis.solvers.TPfracStep import __all__ as _solver_all
from twoasis.solvers.TPfracStep import (
    add_contact_angles, PhaseFieldOnVelocitySpace, potential_linear_derivative,
    set_mixed_fields, wall_potential_linear_derivative)

# The functions the main script takes from a solver.
__all__ = list(_solver_all)


def setup(u_components, u, v, p, q, phi, g, xi, eta, q_, q_1, u_, u_1, p_, phi_, g_, phi_1,
          V, Q, x_, bcs, dt, rho, mu, sigma, epsilon, M, angles, ds, normal, acc, gradp_avg,
          fpf, bpf, velocity_update_solver, **NS_namespace):
    """Define the forms of all equations."""
    sigma_bar = sigma * 3. / (2 * math.sqrt(2))

    # Density, its inverse and viscosity on V, set from the phase field
    rho_, rho_inv_, mu_ = Function(V), Function(V), Function(V)
    phi_V = PhaseFieldOnVelocitySpace(q_['phig'].function_space(), V)
    set_mixed_fields(phi_V, rho_, rho_inv_, mu_, q_, rho, mu)

    # Cahn-Hilliard: conservative advection of phi (the boundary term gives the
    # flux at open boundaries), mobility M, chemical potential g with the
    # linearized potential and the wall terms of the contact angles.
    Fpf = ((1. / dt) * (phi - phi_1) * xi * dx
           - phi * dot(u_1, grad(xi)) * dx + dot(u_1, normal) * phi * xi * ds
           + M * inner(grad(g), grad(xi)) * dx
           + g * eta * dx
           - sigma_bar * epsilon * inner(grad(phi), grad(eta)) * dx
           - sigma_bar / epsilon * potential_linear_derivative(phi, phi_1) * eta * dx)
    Fpf = add_contact_angles(Fpf, angles, sigma, wall_potential_linear_derivative(phi, phi_1),
                             eta, ds)
    if bpf:
        Fpf -= fpf * xi * dx

    # Tentative velocity, component i: semi-implicit convection by u_1, viscous
    # stress div(mu grad(u)) plus its transpose part grad(mu).d(u_1)/dx_i
    # (explicit), the pressure at the current inner iteration, the capillary
    # force -phi grad(g), the mean pressure gradient and the body force rho*acc.
    Fu = {ui: ((1. / dt) * rho_ * (u - q_1[ui]) * v * dx
               + rho_ * dot(u_1, grad(u)) * v * dx
               + mu_ * inner(grad(u), grad(v)) * dx
               - inner(grad(mu_), u_1.dx(i)) * v * dx
               + p_.dx(i) * v * dx
               + phi_ * g_.dx(i) * v * dx
               + gradp_avg[i] * v * dx
               - rho_ * acc[i] * v * dx)
          for i, ui in enumerate(u_components)}

    # Pressure: div(1/rho grad(p - p_)) = div(u*)/dt, p_ being the pressure
    # before the solve.
    Fp = rho_inv_ * inner(grad(p - p_), grad(q)) * dx + (1. / dt) * div(u_) * q * dx

    # Without a pressure Dirichlet condition the pressure is defined up to a
    # constant. LU cannot solve the singular system, so use a Krylov solver
    # (with a tight tolerance) on the complement of the constants.
    p_sol = null_space = None
    if not bcs['p']:
        null_vec = Vector(x_['p'])
        Q.dofmap().set(null_vec, 1.0)
        null_vec *= 1.0 / null_vec.norm('l2')
        null_space = VectorSpaceBasis([null_vec])
        p_sol = PETScKrylovSolver('gmres', 'hypre_amg')
        p_sol.parameters.update(dict(relative_tolerance=1e-12, absolute_tolerance=1e-15,
                                     maximum_iterations=1000, error_on_nonconvergence=True))

    # The velocity correction uses d(dp)/dx_i projected onto V. As in IPCS, the
    # projection lumps the mass matrix for P1 if velocity_update_solver says
    # so (then the inverse row sums ML_inv are needed), and otherwise uses the
    # consistent mass matrix with homogeneous Dirichlet conditions.
    ML_inv = None
    if velocity_update_solver.get('method') == 'lumping' and V.ufl_element().degree() == 1:
        ML_inv = assemble(v * dx)
        ML_inv.set_local(1. / ML_inv.get_local())
        ML_inv.apply('insert')

    # Each equation is stored as the pair (bilinear form, linear form).
    return dict(Fpf=(lhs(Fpf), rhs(Fpf)), Fu={ui: (lhs(F), rhs(F)) for ui, F in Fu.items()},
                Fp=(lhs(Fp), rhs(Fp)), rho_=rho_, rho_inv_=rho_inv_, mu_=mu_, phi_V=phi_V,
                sigma_bar=sigma_bar, p_sol=p_sol, null_space=null_space, ML_inv=ML_inv,
                bcs0={ui: homogenize(bcs[ui]) for ui in u_components}, dpdx=Function(V))


def phase_field_solve(Fpf, q_, bcs, t, fpf, bpf, **NS_namespace):
    """Solve the Cahn-Hilliard equations."""
    if bpf and "t" in getattr(fpf, "_user_parameters", {}):
        fpf.t = t
    solve(Fpf[0] == Fpf[1], q_['phig'], bcs['phig'])


def assemble_first_inner_iter(phi_V, rho_, rho_inv_, mu_, q_, rho, mu, **NS_namespace):
    """Update the density and viscosity from the new phase field."""
    set_mixed_fields(phi_V, rho_, rho_inv_, mu_, q_, rho, mu)


def velocity_tentative_solve(ui, Fu, q_, x_, bcs, udiff, **NS_namespace):
    """Solve for one component of the tentative velocity."""
    change = x_[ui].copy()
    solve(Fu[ui][0] == Fu[ui][1], q_[ui], bcs[ui])
    change.axpy(-1., x_[ui])
    udiff[0] += change.norm('l2')


def pressure_solve(Fp, p_, dp_, bcs, p_sol, null_space, **NS_namespace):
    """Solve for the pressure; dp_ is its change."""
    dp_.assign(p_)
    if p_sol is None:
        solve(Fp[0] == Fp[1], p_, bcs['p'])
    else:
        A, b = assemble(Fp[0]), assemble(Fp[1])
        as_backend_type(A).set_nullspace(null_space)
        null_space.orthogonalize(b)
        p_sol.solve(A, p_.vector(), b)
        normalize(p_.vector())
    dpv = dp_.vector()
    dpv.axpy(-1., p_.vector())
    dpv *= -1.


def velocity_update(u_components, u, v, x_, bcs, bcs0, dp_, dpdx, dt, rho_inv_, ML_inv,
                    **NS_namespace):
    """Correct the velocity: u = u* - dt/rho d(dp)/dx_i."""
    for i, ui in enumerate(u_components):
        if ML_inv is not None:
            dpdx.vector().zero()
            dpdx.vector().axpy(1., assemble(dp_.dx(i) * v * dx) * ML_inv)
        else:
            solve(u * v * dx == dp_.dx(i) * v * dx, dpdx, bcs0[ui])
        x_[ui].axpy(-dt, dpdx.vector() * rho_inv_.vector())
        [bc.apply(x_[ui]) for bc in bcs[ui]]
