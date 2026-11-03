"""Fractional-step solver for the two-phase model with first-order incremental
pressure correction (IPCS)."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2022-03-18"
__copyright__ = ("Copyright (C) 2022 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"

import math

from dolfin import (
    as_backend_type, as_vector, assemble, Constant, dot, dx, Function, grad,
    inner, Matrix, nabla_grad, normalize, Timer)

from twoasis.problems import info_blue
from twoasis.solvers.TPfracStep import __all__ as _solver_all
from twoasis.solvers.TPfracStep import (
    assemble_phase_field_system, attach_pressure_nullspace, krylov_get_solvers,
    phase_field_system, PhaseFieldOnVelocitySpace, set_mixed_fields)

# The functions the main script takes from a solver.
__all__ = list(_solver_all)


def setup(u_components, u, v, p, q, bcs,
          scalar_components, V, Q, x_, p_, u_, A_cache, q_,
          rho, mu, phi_, phi_1, u_1, g_, sigma, epsilon, M, phi, xi, g, eta,
          angles, ds, normal,
          velocity_update_solver, assemble_matrix, homogenize,
          GradFunction, PhiGradFunction, DivFunction, TPsource, **NS_namespace):
    """Preassemble matrices."""
    sigma_bar = sigma * 3./(2*math.sqrt(2))

    rho_ = Function(V)
    rho_inv_ = Function(V)
    mu_ = Function(V)
    phi_V = PhaseFieldOnVelocitySpace(q_['phig'].function_space(), V)
    set_mixed_fields(phi_V, rho_, rho_inv_, mu_, q_, rho, mu)

    # Mass matrix without density coefficient
    Md = assemble_matrix(inner(u, v) * dx)
    # Mass matrix with the density, which changes with time
    Mt = (Matrix(Md), rho_ * inner(u, v))

    # Stiffness matrix that changes with time
    Kt = (Matrix(Md), mu_ * inner(grad(u), grad(v)))

    # Pressure Laplacian without density coefficient
    Apd = assemble_matrix(inner(grad(q), grad(p)) * dx, bcs['p'])
    # Pressure Laplacian that changes with time
    Apt = [Matrix(Apd), rho_inv_ * inner(grad(q), grad(p)), None]

    # Allocate coefficient matrix (needs reassembling)
    A = Matrix(Md)
    Ai = None
    if not all([bcs['u0'] == bcs[ui] for ui in u_components[1:]]):
        info_blue("Some velocity Dirichlet BCs are not no-slip.")
        Ai = dict([(ui, Matrix(Md)) for ui in u_components])

    # Allocate Function for holding and computing the velocity divergence on Q
    divu = DivFunction(u_, Q, name='divu', method=velocity_update_solver)

    # Convecting velocity, set every timestep; used for both phase-field and momentum advection.
    u_adv = as_vector([Function(V) for _ in range(len(u_components))])

    # Phase-field matrices and forms. The time-derivative term (1/dt)*Mpf[0] is
    # added to the constant part on the first step.
    pf = phase_field_system(phi, xi, g, eta, phi_1, u_adv, normal, M, sigma, sigma_bar,
                            epsilon, angles, ds, assemble_matrix)

    # Allocate a dictionary of Functions for holding and computing pressure gradients
    gradp = {ui: GradFunction(p_, V, i=i, name='dpd' + ('x', 'y', 'z')[i],
                              bcs=homogenize(bcs[ui]),
                              method=velocity_update_solver)
             for i, ui in enumerate(u_components)}

    phigradg = {ui: PhiGradFunction(q_['phig'], V, i=i, name='phi_dgd' + ('x', 'y', 'z')[i],
                              bcs=homogenize(bcs[ui]),
                              method=velocity_update_solver)
             for i, ui in enumerate(u_components)}

    # Create dictionary to be returned into global NS namespace
    d = dict(A=A, Ai=Ai, Md=Md, Apt=Apt, Mt=Mt, Kt=Kt, divu=divu, gradp=gradp,
             phigradg=phigradg, **pf)

    # Convection form
    a_conv = inner(v, rho_ * dot(u_adv, nabla_grad(u))) * dx

    TPT = TPsource(mu_, u_adv, V, name="N")

    # Shortcuts when both phases have the same properties. With equal viscosities
    # mu_ is uniform: grad(mu) = 0, so the TPT source term vanishes, and the
    # stiffness matrix Kt never changes. With equal densities rho_ is uniform, so
    # the mass matrix Mt and the pressure Poisson matrix never change. With both,
    # the coefficient fields never change.
    matched_mu = mu[0] == mu[1]
    matched_rho = rho[0] == rho[1]
    matched_properties = matched_mu and matched_rho
    if matched_mu:
        info_blue("Matched viscosities: skipping the TPT source term, which is zero, "
                  "and assembling the stiffness matrix once.")
        assemble(Kt[1] * dx, tensor=Kt[0])
    if matched_rho:
        info_blue("Matched densities: assembling the mass and pressure Poisson matrices once.")
        assemble(Mt[1] * dx, tensor=Mt[0])
        assemble(Apt[1] * dx, tensor=Apt[0])
        [bc.apply(Apt[0]) for bc in bcs['p']]
    d.update(matched_mu=matched_mu, matched_rho=matched_rho,
             matched_properties=matched_properties)

    if bcs['p'] == []:
        Apt[2] = attach_pressure_nullspace(Apt[0], x_, Q)

    d.update(u_adv=u_adv, a_conv=a_conv, TPT=TPT, rho_=rho_, mu_=mu_, rho_inv_=rho_inv_, phi_V=phi_V, sigma_bar=sigma_bar)
    return d

get_solvers = krylov_get_solvers


def assemble_first_inner_iter(A, Ai, a_conv, dt, Mt, Md, scalar_components,
                              Kt, u_adv, u_components,
                              b_tmp, bg0, b0, x_1, bcs, TPT, rho, mu,
                              rho_, mu_, rho_inv_, phi_V, q_, acc, gradp_avg, bgp0,
                              phigradg, Apt, matched_mu, matched_rho, matched_properties,
                              **NS_namespace):
    """Called on the first inner iteration of each timestep.

    Assembles everything that stays fixed during the inner iterations: the
    coefficient fields, the tentative-velocity matrix and the parts of its
    right-hand side that do not depend on the pressure, and the pressure matrix.
    """
    t0 = Timer("Assemble first inner iter")

    # With matched properties the coefficient fields are uniform and were set in setup.
    if not matched_properties:
        set_mixed_fields(phi_V, rho_, rho_inv_, mu_, q_, rho, mu)

    # assemble mass for rhs and lhs
    if not matched_rho:
        assemble(Mt[1] * dx, tensor=Mt[0])

    # A uniform acceleration gives the right-hand side acc[i]*assemble(rho_*v*dx),
    # which equals acc[i]*Md*rho_.
    acc_is_uniform = isinstance(acc, Constant)
    if acc_is_uniform:
        acc_vals = acc.values()
        b_rho = Md * rho_.vector()

    for i, ui in enumerate(u_components):
        # start with (gravitational) acceleration
        if acc_is_uniform:
            b_tmp[ui].zero()
            b_tmp[ui].axpy(acc_vals[i], b_rho)
        else:
            assemble(rho_ * bg0[ui] * dx, tensor=b_tmp[ui])
        # add average pressure gradient
        if not isinstance(gradp_avg, Constant):  # may change during the run
            assemble(bgp0[ui] * dx, tensor=b0[ui])
        b_tmp[ui].axpy(-1., b0[ui])

        # Add transient, convection and diffusion
        b_tmp[ui].axpy(1. / dt, Mt[0] * x_1[ui])
        if not matched_mu:
            TPT.assemble_rhs(i)
            b_tmp[ui].axpy(1., TPT.vector())

        # Capillary force. The phase field is solved before the inner
        # iterations, so this term is fixed within the timestep.
        phigradg[ui].assemble_rhs()
        b_tmp[ui].axpy(-1., phigradg[ui].rhs)

    # assemble semi-implicit convection, diffusion etc. for lhs
    assemble(a_conv, tensor=A)
    A.axpy(1. / dt, Mt[0], True)
    if not matched_mu:
        assemble(Kt[1] * dx, tensor=Kt[0])
    A.axpy(1.0, Kt[0], True)

    if Ai:
        for ui in u_components:
            Ai[ui].zero()
            Ai[ui].axpy(1.0, A, True)
            [bc.apply(Ai[ui]) for bc in bcs[ui]]
    else:
        [bc.apply(A) for bc in bcs['u0']] # assumes only no-slip!

    # The pressure matrix depends on the density, which is fixed within the timestep.
    if not matched_rho:
        t1 = Timer("Pressure assemble")
        assemble(Apt[1] * dx, tensor=Apt[0])
        if bcs['p'] == []:
            Aa = as_backend_type(Apt[0])
            Aa.set_nullspace(Apt[2])
        else:
            [bc.apply(Apt[0]) for bc in bcs['p']]
        t1.stop()

def velocity_tentative_assemble(ui, b, b_tmp, p_, gradp, **NS_namespace):
    """Add the pressure gradient to the rhs of the tentative velocity system."""
    b[ui].zero()
    b[ui].axpy(1., b_tmp[ui])
    gradp[ui].assemble_rhs(p_)
    b[ui].axpy(-1., gradp[ui].rhs)


def pressure_assemble(b, x_, dt, Apt, divu, **NS_namespace):
    """Assemble the rhs of the pressure equation. The matrix is assembled in
    assemble_first_inner_iter."""
    divu.assemble_rhs()  # Computes div(u_)*q*dx
    b['p'][:] = divu.rhs
    b['p'] *= (-1. / dt)
    b['p'].axpy(1., Apt[0] * x_['p'])


def pressure_solve(dp_, x_, Apt, b, p_sol, bcs, inner_iter, tstep, matched_rho,
                   **NS_namespace):
    """Solve pressure equation."""
    # Rebuilding the AMG preconditioner is expensive, so reuse it while the matrix
    # is unchanged: within a timestep, or for the whole run if the matrix is constant.
    if matched_rho:
        p_sol.set_reuse_preconditioner(tstep > 1 or inner_iter > 1)
    else:
        p_sol.set_reuse_preconditioner(inner_iter > 1)
    [bc.apply(b['p']) for bc in bcs['p']]
    dp_.vector().zero()
    dp_.vector().axpy(1., x_['p'])
    # With no pressure boundary condition the pressure is defined up to a
    # constant: make the right-hand side consistent (orthogonal to the null space).
    if Apt[2] is not None:
        Apt[2].orthogonalize(b['p'])

    t1 = Timer("Pressure Linear Algebra Solve")
    p_sol.solve(Apt[0], x_['p'], b['p'])
    t1.stop()
    # LUSolver uses normalize directly for normalization of pressure
    if bcs['p'] == []:
        normalize(x_['p'])

    dpv = dp_.vector()
    dpv.axpy(-1., x_['p'])
    dpv *= -1.


def velocity_update(u_components, bcs, gradp, dp_, dt, x_, rho, rho_inv_, matched_rho,
                    **NS_namespace):
    """Update the velocity after regular pressure velocity iterations."""
    for ui in u_components:
        gradp[ui](dp_)
        if matched_rho:
            x_[ui].axpy(-dt / rho[0], gradp[ui].vector())
        else:
            x_[ui].axpy(-dt, gradp[ui].vector() * rho_inv_.vector())
        [bc.apply(x_[ui]) for bc in bcs[ui]]

def scalar_assemble(a_scalar, a_conv, Ta, dt, M, scalar_components, Schmidt_T, KT,
                    nu, nut_, nunn_, Schmidt, b, K, x_1, b0, les_model, nn_model, **NS_namespace):
    """Assemble scalar equation."""
    pass


def scalar_solve(ci, scalar_components, Ta, b, x_, bcs, c_sol,
                 nu, Schmidt, K, **NS_namespace):
    """Solve scalar equation."""
    pass

def phase_field_assemble(dt, Apf, Mpf, Lpft, a_pf_full, Apf_const, Apf_alpha, b, x_1,
                         u_components, u_adv, t, fpf, bpf, **NS_namespace):
    """Assemble phase field equation (backward Euler in time)."""

    # Update u_adv used as convecting velocity
    for i, ui in enumerate(u_components):
        u_adv[i].vector().zero()
        u_adv[i].vector().axpy(1.0, x_1[ui])

    t1 = Timer("Phase field assemble")
    assemble_phase_field_system(1. / dt, Apf, a_pf_full, Apf_const, Apf_alpha, Mpf,
                                Lpft, b['phig'], bpf, fpf, t)
    # Add mass
    b['phig'].axpy(1. / dt, Mpf[0] * x_1['phig'])
    t1.stop()

def phase_field_solve(pf_sol, Apf, x_, b, q_, bcs, **NS_namespace):
    """Solve scalar equation."""
    t1 = Timer("Phase field solve")
    [bc.apply(Apf, b['phig']) for bc in bcs['phig']]
    pf_sol.solve(Apf, x_['phig'], b['phig'])
    t1.stop()
