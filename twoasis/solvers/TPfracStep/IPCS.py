from dolfin import *
from ..TPfracStep import *
from ..TPfracStep import __all__
import math
import numpy as np
from twoasis.problems import info_blue

#__all__ += ["max_iter", "iters_on_first_timestep"]

#max_iter = 1 #0                 # Number of inner pressure velocity iterations on timestep
#iters_on_first_timestep = 10

def setup(u_components, u, v, p, q, bcs, dt,
          scalar_components, V, Q, x_, p_, u_, A_cache, q_,
          rho, mu, phi_, phi_1, u_1, g_, sigma, epsilon, M, phi, xi, g, eta,
          angles, ds,
          velocity_update_solver, assemble_matrix, homogenize,
          GradFunction, PhiGradFunction, DivFunction, TPsource, **NS_namespace):
    """Preassemble matrices."""
    sigma_bar = sigma * 3./(2*math.sqrt(2))

    #rho_ = weighted_arithmetic_mean(rho, phi_)
    #mu_ = weighted_geometric_mean(mu, phi_)
    rho_ = Function(V)
    rho_inv_ = Function(V)
    mu_ = Function(V)
    c__ = Function(V)
    #rho_.vector()[:] = rho[0]
    #mu_.vector()[:] = mu[0]

    # Mass matrix without density coefficient
    Md = assemble_matrix(inner(u, v) * dx)
    Mt = (Matrix(Md), rho_ * inner(u, v) )

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

    # Allocate phase field matrices
    fprime = potential_linear_derivative(phi, phi_1)
    fwprime = wall_potential_linear_derivative(phi, phi_1)

    Fpft = -sigma_bar / epsilon * fprime * eta * dx
    #Fpft = fprime * eta * dx
    for theta, mark in angles:
        Fpft += sigma * math.cos(theta) * fwprime * eta * ds(mark)
    apft, Lpft = lhs(Fpft), rhs(Fpft)

    #FM = assemble_matrix(phi * xi * dx + g * eta * dx)
    #FK = assemble_matrix(M * inner(grad(g), grad(xi)) * dx - sigma_bar * epsilon * dot(grad(phi), grad(eta)) * dx)
    #a_phase_field = inner(dot(grad(xi), u_), phi_) * dx - sigma_bar / epsilon * lhs(fprime) * eta * dx
    #L_phase_field = - sigma_bar/ epsilon * rhs(fprime) * eta * dx
    """
    Fpf = ((1. / dt) * inner(phi_ - phi_1, xi) * dx
                - inner(dot(grad(xi), u_), phi_) * dx
                + M * inner(grad(g_), grad(xi)) * dx
                + g_ * eta * dx
                - sigma_bar * epsilon * dot(grad(phi_), grad(eta)) * dx
                - sigma_bar / epsilon * fprime * eta * dx
                )
    """
    
    Mpf = (assemble_matrix(inner(phi, xi) * dx), assemble_matrix(g * eta * dx))
    a_pf = -inner(dot(grad(xi), u_), phi) * dx
    Kpf = (assemble_matrix(inner(grad(g), grad(xi)) * dx), assemble_matrix(inner(grad(phi), grad(eta)) * dx))
    Fpft = (Matrix(Mpf[0]), apft, Lpft)
    Apf = Matrix(Mpf[0])

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
    d = dict(A=A, Ai=Ai, Apt=Apt, Mt=Mt, Kt=Kt, divu=divu, gradp=gradp, phigradg=phigradg, 
             Apf=Apf, Mpf=Mpf, Kpf=Kpf, Fpft=Fpft, a_pf=a_pf)

    # Setup for solving convection
    u_adv = as_vector([Function(V) for _ in range(len(u_components))])
    #rho_ * u_1 - M * drho_ * grad(g_)
    a_conv = inner(v, rho_ * dot(u_adv, nabla_grad(u))) * dx

    TPT = TPsource(mu_, u_adv, V, name="N")
    #TPT = None

    if bcs['p'] == []:
        attach_pressure_nullspace(Apt, x_, Q)

    d.update(u_adv=u_adv, a_conv=a_conv, TPT=TPT, rho_=rho_, mu_=mu_, rho_inv_=rho_inv_, c__=c__, sigma_bar=sigma_bar)
    return d

def get_solvers(use_krylov_solvers, krylov_solvers, bcs,
                x_, Q, scalar_components, velocity_krylov_solver, velocity_update_solver,
                pressure_krylov_solver, scalar_krylov_solver, phase_field_krylov_solver, **NS_namespace):
    """Return linear solvers.

    We are solving for
       - tentative velocity
       - pressure correction

       and possibly:
       - scalars

    """
    ## tentative velocity solver ##
    u_pred_prec = PETScPreconditioner(velocity_krylov_solver['preconditioner_type'])
    u_pred_sol = PETScKrylovSolver(velocity_krylov_solver['solver_type'], u_pred_prec)
    u_pred_sol.parameters.update(krylov_solvers)

    ## velocity update solver ##
    u_corr_prec = PETScPreconditioner(velocity_update_solver['preconditioner_type'])
    u_corr_sol = PETScKrylovSolver(velocity_update_solver['solver_type'], u_corr_prec)
    u_corr_sol.parameters.update(krylov_solvers)

    u_sol = (u_pred_sol, u_corr_sol)

    ## pressure solver ##
    p_prec = PETScPreconditioner(pressure_krylov_solver['preconditioner_type'])
    p_sol = PETScKrylovSolver(pressure_krylov_solver['solver_type'], p_prec)
    p_sol.parameters.update(krylov_solvers)
    #p_sol.set_reuse_preconditioner(True) TEST

    ## phase field solver ##
    pf_prec = PETScPreconditioner(phase_field_krylov_solver["preconditioner_type"])
    pf_sol = PETScKrylovSolver(phase_field_krylov_solver['solver_type'], pf_prec)
    pf_sol.parameters.update(krylov_solvers)
    #phig_sol.set_reuse_preconditioner(True) TEST!

    sols = [u_sol, p_sol, pf_sol]

    sols.append(None)

    return sols


def assemble_first_inner_iter(A, Ai, a_conv, dt, Mt, scalar_components,
                              Kt, u_adv, u_components,
                              b_tmp, bg0, b0, x_1, bcs, TPT, rho, mu,
                              rho_, mu_, rho_inv_, c__, q_, gradp_avg, bgp0,
                              **NS_namespace):
    """Called on first inner iteration of velocity/pressure system.

    Assemble convection matrix, compute rhs of tentative velocity and
    reset coefficient matrix for solve.

    """
    t0 = Timer("Assemble first inner iter")

    assign(c__, q_['phig'].sub(0))
    cv = c__.vector()[:]
    cv += 1
    cv *= 0.5
    cv[cv < 0.0] = 0.0
    cv[cv > 1.0] = 1.0
    #print("cv", np.min(cv))
    #rho_.vector().zero()
    #rho_.vector().axpy(1., rho[0]*cv + rho[1]*(1-cv))
    # rho_.vector()[:] = rho[0]**cv[:] * rho[1]**(1-cv[:]) !not good
    rho_.vector()[:] = rho[0]*cv + rho[1]*(1-cv)
    #print("rho", np.min(rho_.vector()[:]))
    rho_inv_.vector()[:] = 1. / rho_.vector()[:]
    #chi = 0.5*(1 + np.sin(math.pi/2 * phi_bar))
    # mu_.vector()[:] = mu[0]**cv[:] * mu[1]**(1-cv[:])
    mu_.vector()[:] = 1 / (cv[:]/mu[0] + (1-cv[:]) / mu[1])

    # assemble mass for rhs and lhs
    assemble(Mt[1] * dx, tensor=Mt[0])
    
    for i, ui in enumerate(u_components):
        # zero out rhs
        b_tmp[ui].zero()
        # start with (gravitational) acceleration
        assemble(rho_ * bg0[ui] * dx, tensor=b_tmp[ui])
        # add average pressure gradient
        if isinstance(gradp_avg, Expression):
            b0[ui].zero()
            assemble(bgp0[ui] * dx, tensor=b0[ui])
        b_tmp[ui].axpy(-1., b0[ui])

        # Add transient, convection and diffusion
        b_tmp[ui].axpy(1. / dt, Mt[0] * x_1[ui])
        TPT.assemble_rhs(i)
        b_tmp[ui].axpy(1., TPT.vector())

    # Update u_adv used as convecting velocity
    for i, ui in enumerate(u_components):
        u_adv[i].vector().zero()
        u_adv[i].vector().axpy(1.0, x_1[ui])

    # assemble semi-implicit convection, diffusion etc. for lhs
    A = assemble(a_conv, tensor=A)
    A.axpy(1. / dt, Mt[0], True)
    assemble(Kt[1] * dx, tensor=Kt[0])
    A.axpy(1.0, Kt[0], True)

    if Ai:
        for ui in u_components:
            Ai[ui].zero()
            Ai[ui].axpy(1.0, A, True)
            [bc.apply(Ai[ui]) for bc in bcs[ui]]
    else:
        [bc.apply(A) for bc in bcs['u0']] # assumes only no-slip!

def attach_pressure_nullspace(Apt, x_, Q):
    """Create null space basis object and attach to Krylov solver."""
    null_vec = Vector(x_['p'])
    Q.dofmap().set(null_vec, 1.0)
    null_vec *= 1.0 / null_vec.norm('l2')
    Aa = as_backend_type(Apt[0])
    null_space = VectorSpaceBasis([null_vec])
    Aa.set_nullspace(null_space)
    Aa.null_space = null_space
    Apt[2] = null_space

def velocity_tentative_assemble(ui, q_, b, b_tmp, p_, gradp, phigradg, c__, **NS_namespace):
    """Add pressure gradient to rhs of tentative velocity system."""
    b[ui].zero()
    b[ui].axpy(1., b_tmp[ui])
    gradp[ui].assemble_rhs(p_)
    b[ui].axpy(-1., gradp[ui].rhs)
    phigradg[ui].assemble_rhs()
    b[ui].axpy(-1., phigradg[ui].rhs)

def velocity_tentative_solve(ui, A, Ai, bcs, x_, x_2, u_sol, b, udiff,
                             use_krylov_solvers, **NS_namespace):
    """Linear algebra solve of tentative velocity component."""
    #if use_krylov_solvers:
        #if ui == 'u0':
            #u_sol.parameters['preconditioner']['structure'] = 'same_nonzero_pattern'
        #else:
            #u_sol.parameters['preconditioner']['structure'] = 'same'
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


def pressure_assemble(b, x_, dx, dt, Apt, divu, bcs, **NS_namespace):
    """Assemble rhs of pressure equation."""
    assemble(Apt[1] * dx, tensor=Apt[0])
    if bcs['p'] == []:
        Aa = as_backend_type(Apt[0])
        Aa.set_nullspace(Apt[2])
    divu.assemble_rhs()  # Computes div(u_)*q*dx
    b['p'][:] = divu.rhs
    b['p'] *= (-1. / dt)
    b['p'].axpy(1., Apt[0] * x_['p'])


def pressure_solve(dp_, x_, Apt, b, p_sol, bcs, **NS_namespace):
    """Solve pressure equation."""
    [bc.apply(b['p']) for bc in bcs['p']]
    dp_.vector().zero()
    dp_.vector().axpy(1., x_['p'])
    # KrylovSolvers use nullspace for normalization of pressure
    if hasattr(Apt[0], 'null_space'):
        p_sol.null_space.orthogonalize(b['p'])

    t1 = Timer("Pressure Linear Algebra Solve")
    p_sol.solve(Apt[0], x_['p'], b['p'])
    t1.stop()
    # LUSolver uses normalize directly for normalization of pressure
    if bcs['p'] == []:
        normalize(x_['p'])

    dpv = dp_.vector()
    dpv.axpy(-1., x_['p'])
    dpv *= -1.


def velocity_update(u_components, bcs, gradp, dp_, dt, x_, b_tmp, Mt, A, u_sol, rho_inv_, **NS_namespace):
    """Update the velocity after regular pressure velocity iterations."""
    for ui in u_components:
        gradp[ui](dp_)
        x_[ui].axpy(-dt, gradp[ui].vector() * rho_inv_.vector())
        [bc.apply(x_[ui]) for bc in bcs[ui]]
        """
        gradp[ui](dp_)

        b_tmp[ui].zero()
        b_tmp[ui].axpy(1.0, Mt[0] * x_[ui])
        b_tmp[ui].axpy(-dt, gradp[ui].vector())
        
        A.zero()
        A.axpy(1.0, Mt[0], True)

        [bc.apply(A) for bc in bcs[ui]]
        [bc.apply(b_tmp[ui]) for bc in bcs[ui]]
        u_sol[1].solve(A, x_[ui], b_tmp[ui])
        [bc.apply(x_[ui]) for bc in bcs[ui]]
        """

def scalar_assemble(a_scalar, a_conv, Ta, dt, M, scalar_components, Schmidt_T, KT,
                    nu, nut_, nunn_, Schmidt, b, K, x_1, b0, les_model, nn_model, **NS_namespace):
    """Assemble scalar equation."""
    pass


def scalar_solve(ci, scalar_components, Ta, b, x_, bcs, c_sol,
                 nu, Schmidt, K, **NS_namespace):
    """Solve scalar equation."""
    pass

def phase_field_assemble(dt, M, sigma_bar, epsilon, Apf, Kpf, Mpf, Fpft, a_pf, b, x_1, **NS_namespace):
    """Assemble phase field equation."""

    assemble(a_pf, tensor=Apf)
    assemble(Fpft[1], tensor=Fpft[0])

    Apf.axpy(1./dt, Mpf[0], True)
    Apf.axpy(M, Kpf[0], True)

    Apf.axpy(1., Mpf[1], True)
    Apf.axpy(-sigma_bar * epsilon, Kpf[1], True)
    Apf.axpy(1., Fpft[0], True)

    assemble(Fpft[2], tensor=b['phig'])
    #b['phig'] *= -sigma_bar / epsilon
    # Add mass
    b['phig'].axpy(1. / dt, Mpf[0] * x_1['phig'])

def phase_field_solve(pf_sol, Apf, x_, b, q_, bcs, **NS_namespace):
    """Solve scalar equation."""
    [bc.apply(Apf, b['phig']) for bc in bcs['phig']]
    pf_sol.solve(Apf, x_['phig'], b['phig'])