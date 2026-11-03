__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
# Modified for two-phase flow by Gaute Linga, 2022-2026.

from dolfin import (
    as_backend_type, assemble, Constant, dot, dx, Function, FunctionAssigner, grad,
    inner, lhs, Matrix, MPI, norm, PETScDMCollection, PETScKrylovSolver, PETScOptions,
    PETScPreconditioner, rhs, Timer, Vector, VectorSpaceBasis)
from twoasis.common.utilities import subfunction, subspace
from twoasis.problems import info_blue
from twoasis.solvers import scalar_assemble, scalar_solve, setup
import math
import numpy as np
# Use the UFL that dolfin was built with (ufl_legacy in recent installations).
try:
    import ufl_legacy as ufl
    if not isinstance(Constant(0.), ufl.Coefficient):
        raise ImportError
except ImportError:
    import ufl


"""Define all functions required by fractional step solver."""
__all__ = ["assemble_first_inner_iter", "velocity_tentative_assemble",
           "velocity_tentative_solve", "pressure_assemble",
           "pressure_solve", "velocity_update", "scalar_assemble",
           "scalar_solve", "get_solvers", "setup",
           "print_velocity_pressure_info", "phase_field_solve", "phase_field_assemble",
           "potential_linear_derivative",
           "wall_potential_linear_derivative",
           "weighted_arithmetic_mean", "weighted_geometric_mean",
           "krylov_solver_parameters", "krylov_solver", "krylov_get_solvers",
           "phase_field_fieldsplit_solver"]


def krylov_solver_parameters(name, cfg, krylov_solvers):
    """Parameters for the Krylov solver whose settings are NS_parameters[name] (`cfg`).

    The settings in `krylov_solvers` apply to every solver. The relative tolerance
    and the iteration limit are taken from the first of these that gives a value:
    the solver's own settings, `krylov_solvers`, and the solver's default in
    KRYLOV_TOLERANCES. So krylov_solvers='{"relative_tolerance": 1e-10}' tightens
    every solve, while pressure_krylov_solver='{"relative_tolerance": 1e-10}'
    tightens only the pressure solve.
    """
    from twoasis.problems.TPfracStep import KRYLOV_TOLERANCES
    params = {key: value for key, value in krylov_solvers.items() if value is not None}
    for key, default in KRYLOV_TOLERANCES.get(name, {}).items():
        value = cfg.get(key)
        if value is None:
            value = krylov_solvers.get(key)
        params[key] = default if value is None else value
    return params


def krylov_solver(name, cfg, krylov_solvers):
    """Krylov solver with the solver and preconditioner type given in `cfg`."""
    sol = PETScKrylovSolver(cfg['solver_type'], PETScPreconditioner(cfg['preconditioner_type']))
    sol.parameters.update(krylov_solver_parameters(name, cfg, krylov_solvers))
    return sol


def krylov_get_solvers(krylov_solvers, q_, velocity_krylov_solver, velocity_update_solver,
                       pressure_krylov_solver, phase_field_krylov_solver,
                       phase_field_fieldsplit, **NS_namespace):
    """Linear solvers for the BDF and IPCS schemes, in the order
    (tentative velocity, velocity update), pressure, phase field, scalar.
    Scalars are not supported, so the scalar solver is None."""
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


def phase_field_fieldsplit_solver(W, phase_field_krylov_solver, krylov_solvers):
    """Fieldsplit (Schur) preconditioner for the coupled (phi, g) Cahn-Hilliard KSP."""
    from petsc4py import PETSc
    pf_sol = PETScKrylovSolver(phase_field_krylov_solver['solver_type'])
    pf_sol.parameters.update(krylov_solver_parameters(
        "phase_field_krylov_solver", phase_field_krylov_solver, krylov_solvers))
    ksp = pf_sol.ksp()
    ksp.setOptionsPrefix("phasefield_")
    pc = ksp.getPC()
    pc.setType("fieldsplit")
    is_phi = PETSc.IS().createGeneral(W.sub(0).dofmap().dofs(), comm=PETSc.COMM_WORLD)
    is_g = PETSc.IS().createGeneral(W.sub(1).dofmap().dofs(), comm=PETSc.COMM_WORLD)
    pc.setFieldSplitIS(("phi", is_phi), ("g", is_g))
    # Schur complement on (phi, g). The g-block is a mass matrix, and the Schur
    # complement onto phi is a stiff, biharmonic-like operator that AMG handles
    # well. A multiplicative split works poorly here: both diagonal blocks are
    # well-conditioned mass matrices, while the stiff terms sit off the diagonal.
    PETScOptions.set("phasefield_pc_fieldsplit_type",
                     phase_field_krylov_solver.get('fieldsplit_type', 'schur'))
    PETScOptions.set("phasefield_pc_fieldsplit_schur_fact_type", "full")
    PETScOptions.set("phasefield_pc_fieldsplit_schur_precondition", "selfp")
    for name in ("phi", "g"):
        PETScOptions.set("phasefield_fieldsplit_%s_ksp_type" % name, "preonly")
        PETScOptions.set("phasefield_fieldsplit_%s_pc_type" % name, "hypre")
    ksp.setFromOptions()
    return pf_sol


def get_solvers(**NS_namespace):
    """Return 4 linear solvers.

    We are solving for
       - tentative velocity
       - pressure correction
       - velocity update (unless lumping is switched on)
       - phase field

       and possibly:
       - scalars

    """
    return (None, ) * 4

def assemble_first_inner_iter(**NS_namespace):
    """Called first thing on a new velocity/pressure iteration."""
    pass


def velocity_matrices(Md, bcs, u_components):
    """The tentative velocity's coefficient matrix A (with the boundary rows of
    u0's conditions), and Ai, one matrix per component, only when the components'
    Dirichlet conditions constrain different dofs. Applying a condition to a
    matrix sets the rows of its dofs whatever its values, so components
    constrained at the same dofs (no-slip walls, or a wall moving along x) share
    A. Decided alike on every process."""
    def dofs(ui):
        return set(k for bc in bcs[ui] for k in bc.get_boundary_values())

    first = dofs(u_components[0])
    same = all(dofs(ui) == first for ui in u_components[1:])
    if MPI.min(MPI.comm_world, float(same)) == 1.:
        return Matrix(Md), None
    info_blue("The velocity components have Dirichlet conditions on different dofs: "
              "one matrix each.")
    return Matrix(Md), {ui: Matrix(Md) for ui in u_components}


def velocity_tentative_solve(ui, A, Ai, bcs, x_, x_2, u_sol, b, udiff,
                             u_components, inner_iter, **NS_namespace):
    """Solve for one component of the tentative velocity."""
    # When all components share the matrix A (Ai is None), it only changes at the
    # start of a timestep, so the preconditioner is rebuilt for the first solve of
    # the timestep and reused after that. With one matrix per component (Ai), the
    # solver switches matrices between solves and must rebuild each time.
    if Ai is None:
        u_sol[0].set_reuse_preconditioner(not (inner_iter == 1 and ui == u_components[0]))

    [bc.apply(b[ui]) for bc in bcs[ui]]
    # x_2 is only needed on the first inner iteration, so use it as a work
    # vector for the change in velocity.
    x_2[ui].zero()
    x_2[ui].axpy(1., x_[ui])
    t1 = Timer("Tentative Linear Algebra Solve")
    u_sol[0].solve(Ai[ui] if Ai else A, x_[ui], b[ui])
    t1.stop()
    x_2[ui].axpy(-1., x_[ui])
    udiff[0] += x_2[ui].norm('l2')


def velocity_tentative_assemble(**NS_namespace):
    """Assemble remaining system for tentative velocity component."""
    pass

def pressure_assemble(**NS_namespace):
    """Assemble rhs of pressure equation."""
    pass


def pressure_solve(**NS_namespace):
    """Solve pressure equation."""
    pass


def velocity_update(**NS_namespace):
    """Update the velocity after finishing pressure velocity iterations."""
    pass

def print_velocity_pressure_info(num_iter, print_velocity_pressure_convergence,
                                 inner_iter, udiff, dp_, **NS_namespace):
    if num_iter > 1 and print_velocity_pressure_convergence:
        if inner_iter == 1:
            info_blue('  Inner iterations velocity pressure:')
            info_blue('                 error u  error p')
        info_blue('    Iter = {0:4d}, {1:2.2e} {2:2.2e}'.format(
            inner_iter, udiff[0], norm(dp_.vector())))

def phase_field_assemble(**NS_namespace):
    """Solve scalar equation."""
    pass

def phase_field_solve(**NS_namespace):
    """Solve scalar equation."""
    pass

def potential_linear_derivative(phi, phi0):
    """ Linearised derivative of phase field potential. """
    return phi0**3 - phi0 + (3*phi0**2 - 1.) * (phi - phi0)

def wall_potential_linear_derivative(phi, phi0):
    return 3.*(1. - phi0**2)/4. - 3.*phi0/2.* (phi - phi0)

class PhaseFieldOnVelocitySpace:
    """The phase field phi, taken from the mixed (phi, g) function, as a Function
    c on the velocity space V: copied when phi has the velocity's degree,
    interpolated when it is lower. The FunctionAssigner and the interpolation
    (transfer) matrix are built once, so an update is a copy and a sparse
    matrix-vector product."""

    def __init__(self, W, V):
        self.c = Function(V)
        self.transfer = None
        if W.sub(0).ufl_element().degree() == V.ufl_element().degree():
            self.assigner = FunctionAssigner(V, W.sub(0))
        else:
            self.transfer = PETScDMCollection.create_transfer_matrix(subspace(W, 0), V)

    def __call__(self, phig):
        if self.transfer is None:
            self.assigner.assign(self.c, phig.sub(0))
        else:
            self.transfer.mult(subfunction(phig, 0).vector(), self.c.vector())
        return self.c

def set_mixed_fields(phi_V, rho_, rho_inv_, mu_, q_, rho, mu):
    """Set the density rho_, its inverse and the viscosity mu_ (Functions on the
    velocity space) from the phase field, mapped there by phi_V (a
    PhaseFieldOnVelocitySpace): c = (1 + phi)/2, clipped to [0, 1], weights rho
    linearly and mu geometrically (robust for large viscosity ratios)."""
    cv = phi_V(q_['phig']).vector()[:]
    cv += 1
    cv *= 0.5
    cv[cv < 0.0] = 0.0
    cv[cv > 1.0] = 1.0
    rho_.vector()[:] = rho[0]*cv + rho[1]*(1-cv)
    rho_inv_.vector()[:] = 1. / rho_.vector()[:]
    mu_.vector()[:] = mu[0]**cv * mu[1]**(1-cv)

def add_contact_angles(F, angles, sigma, fwprime, eta, ds):
    """Add the wall terms sigma*cos(theta)*f_w'(phi) of the phase-field equation
    to the form F, for each (theta, marker) in angles. theta is measured through
    the phase phi = +1; a Function in its place holds cos(theta) pointwise."""
    for theta, mark in angles:
        costheta = theta if isinstance(theta, Function) else math.cos(theta)
        F += sigma * costheta * fwprime * eta * ds(mark)
    return F

def attach_pressure_nullspace(Ap, x_, Q):
    """Attach the null space of constant pressures to the pressure matrix Ap
    (for a pressure without Dirichlet conditions) and return it."""
    null_vec = Vector(x_['p'])
    Q.dofmap().set(null_vec, 1.0)
    null_vec *= 1.0 / null_vec.norm('l2')
    null_space = VectorSpaceBasis([null_vec])
    as_backend_type(Ap).set_nullspace(null_space)
    return null_space


def phase_field_system(phi, xi, g, eta, phi_1, u_adv, normal, M, sigma, sigma_bar,
                       epsilon, angles, ds, assemble_matrix):
    """Forms and matrices of the (phi, g) system of the Cahn-Hilliard equation.

    The double-well potential is linearized about phi_1 (tangent line), with an
    error of O(dt^2). The matrix is split in two parts:
      a_pf_full  the part that changes every timestep: advection by u_adv and the
                 linearized potential, including the wall terms. One form.
      Apf_const  the part that does not: mobility diffusion, the mass matrix for g
                 and the capillary term, plus the time-derivative term
                 alpha0*Mpf[0], which assemble_phase_field_system adds (the
                 coefficient it holds is kept in Apf_alpha).
    Lpft is the right-hand side of the linearized potential.
    """
    fprime = potential_linear_derivative(phi, phi_1)
    fwprime = wall_potential_linear_derivative(phi, phi_1)
    Fpft = add_contact_angles(-sigma_bar / epsilon * fprime * eta * dx,
                              angles, sigma, fwprime, eta, ds)
    apft, Lpft = lhs(Fpft), rhs(Fpft)

    Mpf = (assemble_matrix(inner(phi, xi) * dx), assemble_matrix(g * eta * dx))
    Kpf = (assemble_matrix(inner(grad(g), grad(xi)) * dx),
           assemble_matrix(inner(grad(phi), grad(eta)) * dx))
    # Conservative advection of phi: -int phi (u.grad xi) dx + int (u.n) phi xi ds.
    # The boundary term gives the right flux at open boundaries.
    a_pf = -inner(dot(grad(xi), u_adv), phi) * dx + dot(u_adv, normal) * xi * phi * ds

    Apf_const = Matrix(Mpf[0])
    Apf_const.zero()
    Apf_const.axpy(M, Kpf[0], True)
    Apf_const.axpy(1., Mpf[1], True)
    Apf_const.axpy(-sigma_bar * epsilon, Kpf[1], True)
    return dict(Apf=Matrix(Mpf[0]), Mpf=Mpf, Lpft=Lpft, a_pf_full=a_pf + apft,
                Apf_const=Apf_const, Apf_alpha=np.zeros(1))


def assemble_phase_field_system(alpha0, Apf, a_pf_full, Apf_const, Apf_alpha, Mpf,
                                Lpft, b, bpf, fpf, t):
    """Assemble the phase-field matrix Apf, with the time-derivative term
    alpha0*Mpf[0], and the parts of the right-hand side b that do not involve
    earlier timesteps: the linearized potential and the source term bpf (if any).
    The caller adds the time-derivative terms of the earlier timesteps to b."""
    if alpha0 != Apf_alpha[0]:
        # The constant part only changes when the time-derivative coefficient does.
        Apf_const.axpy(alpha0 - Apf_alpha[0], Mpf[0], True)
        Apf_alpha[0] = alpha0

    assemble(a_pf_full, tensor=Apf)  # advection + linearized potential
    Apf.axpy(1., Apf_const, True)    # + diffusion, g-mass, capillary, dt-mass

    assemble(Lpft, tensor=b)
    # Source term (used by manufactured solutions)
    if bpf:
        if "t" in getattr(fpf, "_user_parameters", {}):
            fpf.t = t
        assemble(bpf[0] * dx, tensor=bpf[1])
        b.axpy(1.0, bpf[1])

def weighted_arithmetic_mean(a, phi):
    phi_bar = ufl.min_value(ufl.max_value(phi, -1.0), 1.0)
    #phi_mod = ufl.sin(math.pi/2 * phi_bar)
    #return 0.5 * (a[0] + a[1] + (a[0] - a[1]) * phi_mod)
    return 0.5 * (a[0] + a[1] + (a[0] - a[1]) * phi_bar)

def weighted_geometric_mean(a, phi):
    phi_bar = ufl.min_value(ufl.max_value(phi, -1.0), 1.0)
    chi = 0.5*(1 + ufl.sin(math.pi/2 * phi_bar))
    return a[0]**chi * a[1]**(1-chi)
