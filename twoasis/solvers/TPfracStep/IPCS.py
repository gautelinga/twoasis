"""Fractional-step solver for the two-phase model with first-order incremental
pressure correction (IPCS)."""
__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2022-03-18"
__copyright__ = ("Copyright (C) 2022 " + __author__ + "; based on Oasis, "
                 "Copyright (C) 2013 Mikael Mortensen <mikaem@math.uio.no>")
__license__ = "GNU Lesser GPL version 3 or any later version"

import math

import numpy as np
from dolfin import as_vector, Constant, dot, dx, Function, grad, inner, Matrix, nabla_grad, Timer

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


def setup(u_components, u, v, p, q, bcs,
          scalar_components, V, Q, x_, p_, u_, q_,
          rho, mu, phi_, phi_1, u_1, g_, sigma, epsilon, M, phi, xi, g, eta,
          angles, ds, normal, rotational,
          velocity_update_solver, assemble_matrix, homogenize,
          GradFunction, PhiGradRHS, DivRHS, TPsource, krylov_solvers, **NS_namespace):
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
    Mt = (Matrix(Md), rho_ * inner(u, v) * dx)

    # Viscous term
    a_visc = mu_ * inner(grad(u), grad(v)) * dx

    # Allocate coefficient matrix (needs reassembling)
    A, Ai = velocity_matrices(Md, bcs, u_components)

    # Allocate Function for holding and computing the velocity divergence on Q
    divu = DivRHS(u_, Q, low_memory=velocity_update_solver.get('low_memory_version', False))

    # Convecting velocity, set every timestep; used for both phase-field and momentum advection.
    u_adv = as_vector([Function(V) for _ in range(len(u_components))])

    # Phase-field matrices and forms. The time-derivative term (1/dt)*Mpf[0] is
    # added to the constant part on the first step.
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

    # Create dictionary to be returned into global NS namespace
    # A Constant mean pressure gradient has a right-hand side equal to a scaled
    # copy of assemble(v*dx); its value is read every timestep, so a hook may
    # change it.
    Fv = assemble(v * dx)
    # alpha[0] = 1/dt, the coefficient of the new velocity in the time derivative,
    # set by phase_field_assemble.
    d = dict(A=A, Ai=Ai, Md=Md, Mt=Mt, divu=divu, gradp=gradp, phigradg=phigradg, Fv=Fv,
             alpha=np.zeros(1), **rotational_fields(rotational, q, Q, V, mu_), **pf)

    # Convection form
    a_conv = inner(v, rho_ * dot(u_adv, nabla_grad(u))) * dx

    TPT = TPsource(mu_, u_adv, V, name="N")

    # Shortcuts when both phases have the same properties. With equal viscosities
    # mu_ is uniform: grad(mu) = 0, so the TPT source term vanishes, and the
    # viscous matrix K_fixed never changes. With equal densities rho_ is uniform, so
    # the mass matrix Mt and the pressure Poisson matrix never change. With both,
    # the coefficient fields never change.
    matched_mu = mu[0] == mu[1]
    matched_rho = rho[0] == rho[1]
    matched_properties = matched_mu and matched_rho
    if matched_mu:
        info_blue("Matched viscosities: skipping the TPT source term, which is zero, "
                  "and assembling the viscous matrix once.")
    if matched_rho:
        info_blue("Matched densities: assembling the mass and pressure Poisson matrices once.")
        assemble(Mt[1], tensor=Mt[0])
    # The pressure equation's coefficient 1/rho is constant only with equal densities.
    d.update(matched_mu=matched_mu, matched_rho=matched_rho,
             matched_properties=matched_properties, constant_poisson_coefficient=matched_rho,
             K_fixed=assemble(a_visc) if matched_mu else None,
             **pressure_matrix(matched_rho, rho, rho_inv_, p, q, Q, bcs))

    d.update(u_adv=u_adv, a_conv=a_conv, a_conv_visc=a_conv + a_visc, TPT=TPT, rho_=rho_, mu_=mu_, rho_inv_=rho_inv_, phi_V=phi_V, sigma_bar=sigma_bar)
    return d

get_solvers = krylov_get_solvers


def assemble_first_inner_iter(A, Ai, a_conv, a_conv_visc, dt, Mt, Md, scalar_components,
                              K_fixed, u_adv, u_components,
                              b_tmp, bg0, b0, x_1, bcs, TPT, rho, mu,
                              rho_, mu_, rho_inv_, phi_V, q_, acc, gradp_avg, bgp0,
                              phigradg, Ap, a_p, p_null, constant_poisson_coefficient,
                              matched_mu, matched_rho, matched_properties,
                              Fv, **NS_namespace):
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
        assemble(Mt[1], tensor=Mt[0])

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
        # add the mean pressure gradient (a hook may change it during the run)
        if isinstance(gradp_avg, Constant):
            b_tmp[ui].axpy(-gradp_avg.values()[i], Fv)
        else:
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
    if matched_mu:
        assemble(a_conv, tensor=A)
        A.axpy(1.0, K_fixed, True)
    else:
        assemble(a_conv_visc, tensor=A)
    A.axpy(1. / dt, Mt[0], True)

    if Ai:
        for ui in u_components:
            Ai[ui].zero()
            Ai[ui].axpy(1.0, A, True)
            [bc.apply(Ai[ui]) for bc in bcs[ui]]
    else:
        [bc.apply(A) for bc in bcs['u0']]  # the same rows for every component

    # The pressure matrix depends on the density, which is fixed within the timestep.
    assemble_pressure_matrix(Ap, a_p, p_null, bcs, constant_poisson_coefficient)

def velocity_tentative_assemble(ui, b, b_tmp, p_, gradp, **NS_namespace):
    """Add the pressure gradient to the rhs of the tentative velocity system."""
    b[ui].zero()
    b[ui].axpy(1., b_tmp[ui])
    gradp[ui].assemble_rhs(p_)
    b[ui].axpy(-1., gradp[ui].rhs)


def scalar_assemble(a_scalar, a_conv, Ta, dt, M, scalar_components, Schmidt_T, KT,
                    nu, nut_, nunn_, Schmidt, b, K, x_1, b0, les_model, nn_model, **NS_namespace):
    """Assemble scalar equation."""
    pass


def scalar_solve(ci, scalar_components, Ta, b, x_, bcs, c_sol,
                 nu, Schmidt, K, **NS_namespace):
    """Solve the scalar equations: none, scalars are not supported."""
    pass

def phase_field_assemble(dt, Apf, Mpf, Lpft, a_pf_full, Apf_const, Apf_alpha, b, x_1,
                         u_components, u_adv, t, fpf, bpf, alpha, **NS_namespace):
    """Assemble phase field equation (backward Euler in time)."""
    alpha[0] = 1. / dt

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
