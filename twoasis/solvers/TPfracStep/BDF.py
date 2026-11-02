from dolfin import *
from ..TPfracStep import *
from ..TPfracStep import __all__
import math
import numpy as np
from twoasis.problems import info_blue

#__all__ += ["max_iter", "iters_on_first_timestep"]

#max_iter = 1 #0                 # Number of inner pressure velocity iterations on timestep
#iters_on_first_timestep = 10

def setup(u_components, u, v, p, q, bcs, #dt,
          scalar_components, V, Q, x_, x_1, x_2, p_, u_, A_cache, q_,
          rho, mu, phi_, phi_1, phi_2, u_1, g_, sigma, epsilon, M, phi, xi, g, eta,
          angles, ds, bdf_order, constant_poisson_coefficient, normal,
          initial_step_1st_order, tstep,
          velocity_update_solver, assemble_matrix, homogenize, acc, gradp_avg,
          GradFunction, PhiGradFunction, DivFunction, TPsource, **NS_namespace):
    """Preassemble matrices."""
    sigma_bar = sigma * 3./(2*math.sqrt(2))

    rho_ = Function(V)
    rho_inv_ = Function(V)
    mu_ = Function(V)
    c__ = Function(V)

    # If things need to be precomputed
    set_mixed_fields(c__, rho_, rho_inv_, mu_, q_, rho, mu)

    # Mass matrix without density coefficient
    Md = assemble_matrix(inner(u, v) * dx)

    # Stiffness matrix that changes with time
    Kt = (Matrix(Md), mu_ * inner(grad(u), grad(v * rho_inv_)))

    # Pressure Laplacian without density coefficient
    Apd = assemble_matrix(1./min(rho) * inner(grad(q), grad(p)) * dx, bcs['p'])

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

    # Setup for solving convection
    u_adv = as_vector([Function(V) for _ in range(len(u_components))])
    #rho_ * u_1 - M * drho_ * grad(g_)
    a_conv = inner(v, dot(u_adv, nabla_grad(u))) * dx

    # Tangent linearization of the potential about phi_1: the implicit
    # f''(phi_1)-term makes the replacement error O(dt^2), matching BDF2.
    fprime = potential_linear_derivative(phi, phi_1)
    fwprime = wall_potential_linear_derivative(phi, phi_1)

    Fpft = -sigma_bar / epsilon * fprime * eta * dx
    for theta, mark in angles:
        if isinstance(theta, Function):
            # theta is actually cos(theta)!
            Fpft += sigma * theta * fwprime * eta * ds(mark)
        else:
            Fpft += sigma * math.cos(theta) * fwprime * eta * ds(mark)
    apft, Lpft = lhs(Fpft), rhs(Fpft)

    Mpf = (assemble_matrix(inner(phi, xi) * dx), assemble_matrix(g * eta * dx))
    a_pf = -inner(dot(grad(xi), u_adv), phi) * dx + dot(u_adv, normal) * xi * phi * ds
    Kpf = (assemble_matrix(inner(grad(g), grad(xi)) * dx), assemble_matrix(inner(grad(phi), grad(eta)) * dx))
    Fpft = (apft, Lpft)
    Apf = Matrix(Mpf[0])

    # Time-varying part of the phase-field operator, assembled in ONE sweep per
    # step: advection + linearized potential (incl. wall terms).
    a_pf_full = a_pf + apft

    # Constant part, combined once: mobility diffusion + g-mass + capillary.
    # The time-derivative mass alpha[0]*Mpf[0] is baked in by phase_field_assemble
    # when alpha[0] changes (startup and the BDF1->BDF2 switch).
    Apf_const = Matrix(Mpf[0])
    Apf_const.zero()
    Apf_const.axpy(M, Kpf[0], True)
    Apf_const.axpy(1., Mpf[1], True)
    Apf_const.axpy(-sigma_bar * epsilon, Kpf[1], True)
    Apf_alpha = np.zeros(1)

    # Allocate a dictionary of Functions for holding and computing pressure gradients
    gradp = {ui: GradFunction(p_, V, i=i, name='dpd' + ('x', 'y', 'z')[i],
                              bcs=homogenize(bcs[ui]),
                              method=velocity_update_solver)
             for i, ui in enumerate(u_components)}

    phigradg = {ui: PhiGradFunction(q_['phig'], V, i=i, name='phi_dgd' + ('x', 'y', 'z')[i],
                              bcs=homogenize(bcs[ui]),
                              method=velocity_update_solver)
             for i, ui in enumerate(u_components)}

    # Decide whether the first step of this run uses BDF1 (see phase_field_assemble).
    #   None  -> auto: BDF2 only if initialize/restart provided a distinct t-2*dt
    #            velocity level (x_2 != x_1), as in TaylorGreen2D's analytic start,
    #            and the phase field's t-2*dt level is not a mere copy of a nonzero
    #            t-dt level. Cases that get BDF1: problems with x_2 = x_1; problems
    #            that leave x_2['phig'] at zero (velocity zero at both levels); and
    #            restarts from old checkpoints without '/previous' for phig, where
    #            q_2 = q_1 is the fallback.
    #   True  -> BDF1 on the first step of a fresh run (tstep == 0).
    #   False -> BDF2 from the first step.
    if initial_step_1st_order is None:
        def levels_differ(fields):
            diff = sum((x_1[ui] - x_2[ui]).norm('l2') for ui in fields)
            scale = sum(x_1[ui].norm('l2') for ui in fields)
            return diff > DOLFIN_EPS_LARGE * (1. + scale)
        phig_copied = (not levels_differ(['phig'])
                       and x_1['phig'].norm('l2') > DOLFIN_EPS_LARGE)
        bdf1_first_step = not levels_differ(u_components) or phig_copied
    else:
        bdf1_first_step = bool(initial_step_1st_order) and tstep == 0
    if bdf_order == 2:
        info_blue(f"BDF1 on first timestep: {bdf1_first_step}")
    bdf1_first_step = np.array([bdf1_first_step], dtype=bool)
    alpha = np.zeros(3)
    beta = np.zeros(2)

    # Create dictionary to be returned into global NS namespace
    d = dict(A=A, Md=Md, Ai=Ai, Apd=Apd, Apt=Apt, Kt=Kt, divu=divu, gradp=gradp, phigradg=phigradg,
             Apf=Apf, Mpf=Mpf, Kpf=Kpf, Fpft=Fpft, a_pf=a_pf, alpha=alpha, beta=beta,
             bdf1_first_step=bdf1_first_step,
             a_pf_full=a_pf_full, Apf_const=Apf_const, Apf_alpha=Apf_alpha)

    TPT = TPsource(mu_, u_adv, V, name="N")

    # Spatially uniform (Constant) body forces: their rhs vectors are scaled
    # copies of assemble(v*dx) resp. Md*rho_inv_, so skip per-step assembly.
    # Values are re-read every step, so mutated Constants stay correct.
    Fv = assemble(v * dx)
    acc_is_uniform = isinstance(acc, Constant)
    gradp_is_uniform = isinstance(gradp_avg, Constant)
    d.update(Fv=Fv, acc_is_uniform=acc_is_uniform, gradp_is_uniform=gradp_is_uniform)

    # Matched-property fast paths (exact, auto-detected): matched mu makes
    # grad(mu)=0 so the TPT source vanishes identically; matched rho makes the
    # Poisson coefficient constant; with both, Kt never changes.
    matched_mu = mu[0] == mu[1]
    if matched_mu:
        info_blue("Matched viscosities: skipping the identically zero TPT source.")
    if rho[0] == rho[1] and not constant_poisson_coefficient:
        info_blue("Matched densities: enabling constant-coefficient pressure Poisson.")
        constant_poisson_coefficient = True
    matched_properties = matched_mu and rho[0] == rho[1]
    if matched_properties:
        info_blue("Matched properties: assembling the stiffness matrix once.")
        assemble(Kt[1] * dx, tensor=Kt[0])
    d.update(matched_mu=matched_mu, matched_properties=matched_properties,
             constant_poisson_coefficient=constant_poisson_coefficient)

    if bcs['p'] == []:
        Apt[2] = attach_pressure_nullspace(Apt[0], x_, Q)
        attach_pressure_nullspace(Apd, x_, Q)

    d.update(u_adv=u_adv, a_conv=a_conv, TPT=TPT, rho_=rho_, mu_=mu_, rho_inv_=rho_inv_, c__=c__, sigma_bar=sigma_bar)
    return d

def get_solvers(krylov_solvers, q_, velocity_krylov_solver, velocity_update_solver,
                pressure_krylov_solver, phase_field_krylov_solver,
                phase_field_fieldsplit, **NS_namespace):
    """Return linear solvers: (tentative velocity, velocity update), pressure,
    phase field, scalar (None)."""
    u_sol = (krylov_solver("velocity_krylov_solver", velocity_krylov_solver, krylov_solvers),
             krylov_solver("velocity_update_solver", velocity_update_solver, krylov_solvers))
    p_sol = krylov_solver("pressure_krylov_solver", pressure_krylov_solver, krylov_solvers)
    if phase_field_fieldsplit:
        pf_sol = phase_field_fieldsplit_solver(q_['phig'].function_space(),
                                               phase_field_krylov_solver, krylov_solvers)
    else:
        pf_sol = krylov_solver("phase_field_krylov_solver", phase_field_krylov_solver,
                               krylov_solvers)
    return [u_sol, p_sol, pf_sol, None]


def set_mixed_fields(c__, rho_, rho_inv_, mu_, q_, rho, mu):
    assign(c__, q_['phig'].sub(0))
    cv = c__.vector()[:]
    cv += 1
    cv *= 0.5
    cv[cv < 0.0] = 0.0
    cv[cv > 1.0] = 1.0

    rho_.vector()[:] = rho[0]*cv + rho[1]*(1-cv)
    rho_inv_.vector()[:] = 1. / rho_.vector()[:]

    mu_.vector()[:] = mu[0]**cv[:] * mu[1]**(1-cv[:])
    # mu_.vector()[:] = 1 / (cv[:]/mu[0] + (1-cv[:]) / mu[1])


def assemble_first_inner_iter(A, Ai, a_conv, dt, Md, scalar_components, bdf_order,
                              Kt, u_adv, u_components, alpha, beta, AB_projection_pressure,
                              b_tmp, bg0, b0, x_, x_1, x_2, bcs, TPT, rho, mu,
                              rho_, mu_, rho_inv_, c__, q_, acc, gradp_avg, bgp0,
                              Fv, acc_is_uniform, gradp_is_uniform,
                              matched_mu, matched_properties,
                              **NS_namespace):
    """Called on first inner iteration of velocity/pressure system.

    Assemble convection matrix, compute rhs of tentative velocity and
    reset coefficient matrix for solve.

    """
    #info_blue("Assembling first inner iter")

    t0 = Timer("Assemble first inner iter")

    # Uniform coefficient fields were set once in setup.
    if not matched_properties:
        set_mixed_fields(c__, rho_, rho_inv_, mu_, q_, rho, mu)

    if acc_is_uniform:
        acc_vals = acc.values()
    if gradp_is_uniform:
        gradp_vals = gradp_avg.values()
        b_gp = Md * rho_inv_.vector()  # == assemble(v*rho_inv_*dx), shared by all components

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

        if not matched_mu:  # grad(mu) = 0 -> TPT source is identically zero
            TPT.assemble_rhs(i)
            b_tmp[ui].axpy(1., TPT.vector() * rho_inv_.vector())

    # assemble semi-implicit convection, diffusion etc. for lhs
    assemble(a_conv, tensor=A)
    A.axpy(alpha[0], Md, True)

    if not matched_properties:  # else Kt was assembled once in setup
        assemble(Kt[1] * dx, tensor=Kt[0])
    A.axpy(1.0, Kt[0], True)

    if Ai:
        for ui in u_components:
            Ai[ui].zero()
            Ai[ui].axpy(1.0, A, True)
            [bc.apply(Ai[ui]) for bc in bcs[ui]]
    else:
        [bc.apply(A) for bc in bcs['u0']] # assumes only no-slip!

    x_['p'].zero()
    if AB_projection_pressure: # and t < (T - tstep * DOLFIN_EPS) and not stop:
        #x_['p'].axpy(0.5, dp_.vector())
        x_['p'].axpy(beta[0], x_1['p'])
        x_['p'].axpy(beta[1], x_2['p'])
    else:
        x_['p'].axpy(1., x_1['p'])

def attach_pressure_nullspace(Ap, x_, Q):
    """Create null space basis object and attach to Krylov solver."""
    null_vec = Vector(x_['p'])
    Q.dofmap().set(null_vec, 1.0)
    null_vec *= 1.0 / null_vec.norm('l2')
    Aa = as_backend_type(Ap)
    null_space = VectorSpaceBasis([null_vec])
    Aa.set_nullspace(null_space)
    Aa.null_space = null_space
    return null_space

def velocity_tentative_assemble(ui, q_, b, b_tmp, p_, gradp, phigradg, rho_inv_,
                                inner_iter, **NS_namespace):
    """Add pressure gradient to rhs of tentative velocity system."""
    #info_blue(f"Assembling {ui}")

    b[ui].zero()
    b[ui].axpy(1., b_tmp[ui])
    gradp[ui].assemble_rhs(p_)
    b[ui].axpy(-1., gradp[ui].rhs * rho_inv_.vector())
    # phig_ only changes once per timestep (solved before the inner loop), so
    # the capillary rhs is frozen within a timestep.
    if inner_iter == 1:
        phigradg[ui].assemble_rhs()
    b[ui].axpy(-1., phigradg[ui].rhs * rho_inv_.vector())

def velocity_tentative_solve(ui, A, Ai, bcs, x_, x_2, u_sol, b, udiff,
                             use_krylov_solvers, u_components, inner_iter, **NS_namespace):
    """Linear algebra solve of tentative velocity component."""
    #info_blue(f"Solving {ui}")

    # Reuse the preconditioner within a timestep when all components share one
    # matrix A (Ai is None). With per-component Ai the shared KSP alternates
    # operators, so reuse would use the wrong hierarchy -> skip.
    if Ai is None:
        u_sol[0].set_reuse_preconditioner(not (inner_iter == 1 and ui == u_components[0]))

    [bc.apply(b[ui]) for bc in bcs[ui]]
    # x_2 only used on inner_iter 1, so use here as work vector
    x_2[ui].zero()
    x_2[ui].axpy(1., x_[ui])
    t1 = Timer("Tentative Linear Algebra Solve")
    if Ai:
        u_sol[0].solve(Ai[ui], x_[ui], b[ui])
    else:
        u_sol[0].solve(A, x_[ui], b[ui])
    t1.stop()
    udiff[0] += norm(x_2[ui] - x_[ui])


def pressure_assemble(b, x_, dx, dt, Apt, Apd, divu, bcs, alpha, constant_poisson_coefficient,
                      inner_iter, **NS_namespace):
    """Assemble rhs of pressure equation."""
    #info_blue("Assembling P")

    # rho_inv_ only changes in assemble_first_inner_iter, so the Laplacian is
    # frozen within a timestep; reassemble on the first inner iteration only.
    if not constant_poisson_coefficient and inner_iter == 1:
        t1 = Timer("Pressure assemble")
        assemble(Apt[1] * dx, tensor=Apt[0])
        if bcs['p'] == []:
            Aa = as_backend_type(Apt[0])
            Aa.set_nullspace(Apt[2])
        else:
            [bc.apply(Apt[0]) for bc in bcs['p']]
        t1.stop()

    divu.assemble_rhs()  # Computes div(u_)*q*dx
    b['p'][:] = divu.rhs
    b['p'] *= -alpha[0]
    if not constant_poisson_coefficient:
        b['p'].axpy(1., Apt[0] * x_['p'])
    else:
        b['p'].axpy(1., Apd * x_['p'])

def pressure_solve(dp_, x_, Apt, Apd, b, p_sol, bcs, constant_poisson_coefficient,
                   inner_iter, tstep, **NS_namespace):
    """Solve pressure equation."""
    #info_blue("Solving P")

    # Reuse the AMG hierarchy across inner iterations (operator frozen within a
    # timestep); build once per run when constant_poisson_coefficient.
    if constant_poisson_coefficient:
        p_sol.set_reuse_preconditioner(tstep > 1 or inner_iter > 1)
    else:
        p_sol.set_reuse_preconditioner(inner_iter > 1)

    [bc.apply(b['p']) for bc in bcs['p']]
    dp_.vector().zero()
    dp_.vector().axpy(1., x_['p'])

    t1 = Timer("Pressure Linear Algebra Solve")
    if not constant_poisson_coefficient:
        # KrylovSolvers use nullspace for normalization of pressure
        if hasattr(Apt[0], 'null_space'):
            p_sol.null_space.orthogonalize(b['p'])
        p_sol.solve(Apt[0], x_['p'], b['p'])
    else:
        if hasattr(Apd, 'null_space'):
            p_sol.null_space.orthogonalize(b['p'])
        p_sol.solve(Apd, x_['p'], b['p'])
    t1.stop()

    # LUSolver uses normalize directly for normalization of pressure
    if bcs['p'] == []:
        normalize(x_['p'])

    dpv = dp_.vector()
    dpv.axpy(-1., x_['p'])
    dpv *= -1.


def velocity_update(u_components, bcs, gradp, dp_, dt, x_, rho_inv_, rho, alpha, constant_poisson_coefficient, **NS_namespace):
    """Update the velocity after regular pressure velocity iterations."""
    factor = 1./alpha[0]
    for ui in u_components:
        gradp[ui](dp_)
        if not constant_poisson_coefficient:
            x_[ui].axpy(- factor, gradp[ui].vector() * rho_inv_.vector())
        else:
            x_[ui].axpy(- factor, gradp[ui].vector() / min(rho))
        [bc.apply(x_[ui]) for bc in bcs[ui]]

def scalar_assemble(a_scalar, a_conv, Ta, dt, M, scalar_components, Schmidt_T, KT,
                    nu, nut_, nunn_, Schmidt, b, K, x_1, b0, les_model, nn_model, **NS_namespace):
    """Assemble scalar equation."""
    pass


def scalar_solve(ci, scalar_components, Ta, b, x_, bcs, c_sol,
                 nu, Schmidt, K, **NS_namespace):
    """Solve scalar equation."""
    pass

def phase_field_assemble(t, tstep, dt, Apf, Mpf, Fpft, a_pf_full, Apf_const, Apf_alpha,
                         b, x_1, x_2, u_adv, u_components, alpha,
                         beta, bdf_order, bdf1_first_step, fpf, bpf,
                         **NS_namespace):
    """Assemble phase field equation."""
    #info_blue("Assembling PF")
    # BDF1 on the first step when x_2 is not a valid t-2*dt level (BDF2 there loses
    # mass); decided once in setup from initial_step_1st_order.
    if bdf_order == 2 and not bdf1_first_step[0]:
        w_n = 1. # dt / dt_1

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

    bdf1_first_step[0] = False
    #info_blue(f"{alpha}, {beta}")

    # Update u_adv used as convecting velocity
    for i, ui in enumerate(u_components):
        u_adv[i].vector().zero()
        u_adv[i].vector().axpy(beta[0], x_1[ui])
        u_adv[i].vector().axpy(beta[1], x_2[ui])

    t1 = Timer("Phase field assemble")
    if alpha[0] != Apf_alpha[0]:
        # Bake the time-derivative mass into the constant part; alpha[0] only
        # changes at startup and at the BDF1->BDF2 switch.
        Apf_const.axpy(alpha[0] - Apf_alpha[0], Mpf[0], True)
        Apf_alpha[0] = alpha[0]

    assemble(a_pf_full, tensor=Apf)  # advection + linearized potential
    Apf.axpy(1., Apf_const, True)    # + diffusion, g-mass, capillary, dt-mass

    assemble(Fpft[1], tensor=b['phig'])
    if bpf:
        if "t" in fpf._user_parameters:
            fpf.t = t
        assemble(bpf[0] * dx, tensor=bpf[1])
        b['phig'].axpy(1.0, bpf[1])

    # Add mass
    b['phig'].axpy(-alpha[1], Mpf[0] * x_1['phig'])
    b['phig'].axpy(-alpha[2], Mpf[0] * x_2['phig'])

    t1.stop()


def phase_field_solve(pf_sol, Apf, x_, b, q_, bcs, **NS_namespace):
    """Solve scalar equation."""
    #info_blue("Solving PF")
    t1 = Timer("Phase field solve")
    [bc.apply(Apf, b['phig']) for bc in bcs['phig']]
    pf_sol.solve(Apf, x_['phig'], b['phig'])
    t1.stop()