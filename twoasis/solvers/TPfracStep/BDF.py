"""Fractional-step solver for the two-phase model with BDF time stepping (first or
second order) and incremental pressure correction."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2022-09-16"
__copyright__ = ("Copyright (C) 2022 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"

import math

import numpy as np
from dolfin import (
    as_vector, Constant, DOLFIN_EPS_LARGE, dot, dx, Function, grad, inner, nabla_grad,
    Timer)

from femtools.fast_assembly import assemble
from twoasis.problems import info_blue
from twoasis.solvers.TPfracStep import __all__ as _solver_all
from twoasis.solvers.TPfracStep import (
    assemble_phase_field_system, assemble_pressure_matrix,
    incremental_pressure_assemble as pressure_assemble,
    incremental_pressure_solve as pressure_solve,
    incremental_velocity_update as velocity_update, krylov_get_solvers,
    phase_field_system, PhaseFieldOnVelocitySpace, pressure_matrix, projection_solver,
    rotational_fields, rotational_pressure_update as pressure_update, set_mixed_fields,
    velocity_matrices)

# The functions the main script takes from a solver.
__all__ = list(_solver_all)


def has_previous_level(x_1, x_2, fields):
    """Whether x_2 holds a usable solution at t - 2*dt for the given fields.

    It does if it differs from x_1 (the solution at t - dt) and is not zero, or
    if the fields are zero at both levels.
    """
    norm_1 = sum(x_1[ui].norm('l2') for ui in fields)
    norm_2 = sum(x_2[ui].norm('l2') for ui in fields)
    diff = sum((x_1[ui] - x_2[ui]).norm('l2') for ui in fields)
    tol = DOLFIN_EPS_LARGE * (1. + norm_1)
    if norm_1 <= tol and norm_2 <= tol:
        return True
    return diff > tol and norm_2 > tol


def setup(u_components, u, v, p, q, bcs, #dt,
          scalar_components, V, Q, x_, x_1, x_2, p_, u_, q_,
          rho, mu, phi_, phi_1, u_1, g_, sigma, epsilon, M, phi, xi, g, eta,
          angles, ds, bdf_order, constant_poisson_coefficient, normal,
          initial_step_1st_order, tstep, rotational,
          velocity_update_solver, assemble_matrix, homogenize, acc, gradp_avg,
          GradFunction, PhiGradRHS, DivRHS, TPsource, krylov_solvers, **NS_namespace):
    """Preassemble matrices."""
    sigma_bar = sigma * 3./(2*math.sqrt(2))

    rho_ = Function(V)
    rho_inv_ = Function(V)
    mu_ = Function(V)
    phi_V = PhaseFieldOnVelocitySpace(q_['phig'].function_space(), V)

    # If things need to be precomputed
    set_mixed_fields(phi_V, rho_, rho_inv_, mu_, q_, rho, mu)

    # Mass matrix without density coefficient
    Md = assemble_matrix(inner(u, v) * dx)

    # Viscous term
    a_visc = mu_ * inner(grad(u), grad(v * rho_inv_)) * dx

    # Allocate coefficient matrix (needs reassembling)
    A, Ai = velocity_matrices(Md, bcs, u_components)

    # Allocate Function for holding and computing the velocity divergence on Q
    divu = DivRHS(u_, Q, low_memory=velocity_update_solver.get('low_memory_version', False))

    # Setup for solving convection
    u_adv = as_vector([Function(V) for _ in range(len(u_components))])
    #rho_ * u_1 - M * drho_ * grad(g_)
    a_conv = inner(v, dot(u_adv, nabla_grad(u))) * dx

    # Phase-field matrices and forms. phase_field_assemble adds the time-derivative
    # term alpha[0]*Mpf[0] to the constant part whenever alpha[0] changes: on the
    # first step, and when switching from BDF1 to BDF2.
    pf = phase_field_system(phi, xi, g, eta, phi_1, u_adv, normal, M, sigma, sigma_bar,
                            epsilon, angles, ds, assemble_matrix)

    # Allocate a dictionary of Functions for holding and computing pressure gradients
    gradp = {ui: GradFunction(p_, V, i=i, name='dpd' + ('x', 'y', 'z')[i],
                              bcs=homogenize(bcs[ui]), method=velocity_update_solver,
                              solver=projection_solver(velocity_update_solver,
                                                       krylov_solvers))
             for i, ui in enumerate(u_components)}

    phigradg = {ui: PhiGradRHS(q_['phig'], V, i)
             for i, ui in enumerate(u_components)}

    # Decide whether the first step of this run is first order (BDF1).
    #   True  -> BDF1 on the first step of a new run.
    #   False -> BDF2 from the first step.
    #   None  -> BDF2 only if both the velocity and the phase field have a usable
    #            solution at t - 2*dt.
    # initialize() or the checkpoint provides that level in x_2. It counts as
    # usable if it differs from the level at t - dt and is not left at zero, or if
    # the field is zero at both levels. Problems that copy x_1 into x_2, or leave
    # x_2 at zero, therefore start with BDF1.
    if initial_step_1st_order is None:
        bdf1_first_step = not (has_previous_level(x_1, x_2, u_components)
                               and has_previous_level(x_1, x_2, ['phig']))
    else:
        bdf1_first_step = bool(initial_step_1st_order) and tstep == 0
    if bdf_order == 2:
        info_blue(f"BDF1 on first timestep: {bdf1_first_step}")
    # Timestep number that uses BDF1 (None if none does).
    bdf1_tstep = tstep + 1 if bdf1_first_step else None
    alpha = np.zeros(3)
    beta = np.zeros(2)

    # Create dictionary to be returned into global NS namespace
    d = dict(A=A, Md=Md, Ai=Ai, divu=divu, gradp=gradp, phigradg=phigradg,
             alpha=alpha, beta=beta, bdf1_tstep=bdf1_tstep,
             **rotational_fields(rotational, q, Q, V, mu_), **pf)

    TPT = TPsource(mu_, u_adv, V, name="N")

    # A body force that is a Constant has a right-hand side equal to a scaled
    # copy of assemble(v*dx), so it does not need to be assembled every timestep.
    Fv = assemble(v * dx)
    d.update(Fv=Fv)

    # Shortcuts when both phases have the same properties. With equal
    # viscosities grad(mu) = 0, so the TPT source term vanishes. With equal
    # densities the pressure Poisson coefficient is constant. With both, the
    # viscous matrix K_fixed never changes.
    matched_mu = mu[0] == mu[1]
    matched_rho = rho[0] == rho[1]
    if matched_mu:
        info_blue("Matched viscosities: skipping the TPT source term, which is zero.")
    if matched_rho and not constant_poisson_coefficient:
        info_blue("Matched densities: using a constant-coefficient pressure Poisson equation.")
        constant_poisson_coefficient = True
    matched_properties = matched_mu and matched_rho
    if matched_properties:
        info_blue("Matched properties: assembling the viscous matrix once.")
    d.update(matched_mu=matched_mu, matched_properties=matched_properties,
             constant_poisson_coefficient=constant_poisson_coefficient,
             K_fixed=assemble(a_visc) if matched_properties else None,
             **pressure_matrix(constant_poisson_coefficient, rho, rho_inv_, p, q, Q, bcs))

    d.update(u_adv=u_adv, a_conv=a_conv, a_conv_visc=a_conv + a_visc, TPT=TPT, rho_=rho_, mu_=mu_, rho_inv_=rho_inv_, phi_V=phi_V, sigma_bar=sigma_bar)
    return d

get_solvers = krylov_get_solvers

def assemble_first_inner_iter(A, Ai, a_conv, a_conv_visc, dt, Md, scalar_components, bdf_order,
                              K_fixed, u_adv, u_components, alpha, beta, AB_projection_pressure,
                              b_tmp, bg0, b0, x_, x_1, x_2, bcs, TPT, rho, mu,
                              rho_, mu_, rho_inv_, phi_V, q_, acc, gradp_avg, bgp0,
                              Fv, matched_mu, matched_properties, phigradg,
                              Ap, a_p, p_null, constant_poisson_coefficient,
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

    acc_is_uniform = isinstance(acc, Constant)
    gradp_is_uniform = isinstance(gradp_avg, Constant)
    if acc_is_uniform:
        acc_vals = acc.values()
    if gradp_is_uniform:
        gradp_vals = gradp_avg.values()
        b_gp = Md * rho_inv_.vector()  # equals assemble(v*rho_inv_*dx)

    for i, ui in enumerate(u_components):
        # zero out rhs
        b_tmp[ui].zero()
        # start with (gravitational) acceleration
        if acc_is_uniform:
            b_tmp[ui].axpy(acc_vals[i], Fv)
        else:
            assemble(bg0[ui] * dx, tensor=b_tmp[ui])
        # add average pressure gradient
        if gradp_is_uniform:
            b_tmp[ui].axpy(-gradp_vals[i], b_gp)
        else:
            b0[ui].zero()
            assemble(bgp0[ui] * rho_inv_ * dx, tensor=b0[ui])
            b_tmp[ui].axpy(-1., b0[ui])
        b_tmp[ui].axpy(-alpha[1], Md * x_1[ui])
        b_tmp[ui].axpy(-alpha[2], Md * x_2[ui])

        if not matched_mu:
            TPT.assemble_rhs(i)
            b_tmp[ui].axpy(1., TPT.vector() * rho_inv_.vector())

        # Capillary force. The phase field is solved before the inner
        # iterations, so this term is fixed within the timestep.
        phigradg[ui].assemble_rhs()
        b_tmp[ui].axpy(-1., phigradg[ui].rhs * rho_inv_.vector())

    # assemble semi-implicit convection, diffusion etc. for lhs
    if matched_properties:
        assemble(a_conv, tensor=A)
        A.axpy(1.0, K_fixed, True)
    else:
        assemble(a_conv_visc, tensor=A)
    A.axpy(alpha[0], Md, True)

    if Ai:
        for ui in u_components:
            Ai[ui].zero()
            Ai[ui].axpy(1.0, A, True)
            [bc.apply(Ai[ui]) for bc in bcs[ui]]
    else:
        [bc.apply(A) for bc in bcs['u0']]  # the same rows for every component

    # The pressure matrix depends on the density, which is fixed within the timestep.
    assemble_pressure_matrix(Ap, a_p, p_null, bcs, constant_poisson_coefficient)

    x_['p'].zero()
    if AB_projection_pressure: # and t < (T - tstep * DOLFIN_EPS) and not stop:
        #x_['p'].axpy(0.5, dp_.vector())
        x_['p'].axpy(beta[0], x_1['p'])
        x_['p'].axpy(beta[1], x_2['p'])
    else:
        x_['p'].axpy(1., x_1['p'])

def velocity_tentative_assemble(ui, b, b_tmp, p_, gradp, rho_inv_, **NS_namespace):
    """Add the pressure gradient to the rhs of the tentative velocity system."""
    b[ui].zero()
    b[ui].axpy(1., b_tmp[ui])
    gradp[ui].assemble_rhs(p_)
    b[ui].axpy(-1., gradp[ui].rhs * rho_inv_.vector())


def scalar_assemble(a_scalar, a_conv, Ta, dt, M, scalar_components, Schmidt_T, KT,
                    nu, nut_, nunn_, Schmidt, b, K, x_1, b0, les_model, nn_model, **NS_namespace):
    """Assemble scalar equation."""
    pass


def scalar_solve(ci, scalar_components, Ta, b, x_, bcs, c_sol,
                 nu, Schmidt, K, **NS_namespace):
    """Solve the scalar equations: none, scalars are not supported."""
    pass

def phase_field_assemble(t, tstep, dt, Apf, Mpf, Lpft, a_pf_full, Apf_const, Apf_alpha,
                         b, x_1, x_2, u_adv, u_components, alpha,
                         beta, bdf_order, bdf1_tstep, fpf, bpf, NS_parameters,
                         **NS_namespace):
    """Assemble phase field equation."""
    # BDF2 needs the solution at t - 2*dt. On the step where setup found none
    # (bdf1_tstep), take a BDF1 step instead, since BDF2 with a wrong t - 2*dt
    # level does not conserve mass.
    # The coefficients for a step dt after a step dt_1 (BDF2 with a variable
    # step). dt_1 is kept in NS_parameters, so that it is also saved with the
    # checkpoint: a restart with another dt continues with the right ratio.
    dt_1 = NS_parameters.get("dt_1") or dt
    NS_parameters["dt_1"] = dt
    if bdf_order == 2 and tstep != bdf1_tstep:
        w_n = dt / dt_1

        alpha[0] = (1. + 2.*w_n)/(1. + w_n) / dt  # 3.0 / (2. * dt)
        alpha[1] = -(1. + w_n) / dt               # - 4.0 / (2 * dt)
        alpha[2] = w_n**2 / (1. + w_n) / dt       # 1. / (2 * dt)

        beta[0] = 1.0 + w_n  # 2.0
        beta[1] = -w_n       # -1.0
    else: # bdf_order == 1:
        alpha[0] = 1. / dt
        alpha[1] = - 1. / dt
        alpha[2] = 0.

        beta[0] = 1.0
        beta[1] = 0.0


    # Update u_adv used as convecting velocity
    for i, ui in enumerate(u_components):
        u_adv[i].vector().zero()
        u_adv[i].vector().axpy(beta[0], x_1[ui])
        u_adv[i].vector().axpy(beta[1], x_2[ui])

    t1 = Timer("Phase field assemble")
    assemble_phase_field_system(alpha[0], Apf, a_pf_full, Apf_const, Apf_alpha, Mpf,
                                Lpft, b['phig'], bpf, fpf, t)

    # Add mass
    b['phig'].axpy(-alpha[1], Mpf[0] * x_1['phig'])
    b['phig'].axpy(-alpha[2], Mpf[0] * x_2['phig'])

    t1.stop()
