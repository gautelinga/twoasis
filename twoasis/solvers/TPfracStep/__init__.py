__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
# Modified for two-phase flow by Gaute Linga, 2022-2026.

from dolfin import (
    as_backend_type, Constant, dot, dx, Function, grad, IndexMap, inner, lhs, Matrix, MPI, norm,
    normalize, PETScKrylovSolver, rhs, TestFunctions, Timer, TrialFunctions, Vector)
from femtools.fast_assembly import assemble
from femtools.fast_bc import apply as apply_bc
from femtools.fast_interpolation import interpolate
from femtools import ksp
from femtools.nullspace import constant_nullspace
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
           "pressure_solve", "velocity_update", "pressure_update", "scalar_assemble",
           "scalar_solve", "get_solvers", "setup",
           "print_velocity_pressure_info", "phase_field_solve", "phase_field_assemble",
           "potential_linear_derivative",
           "wall_potential_linear_derivative",
           "weighted_arithmetic_mean", "weighted_geometric_mean",
           "krylov_solver_parameters", "krylov_solver", "krylov_get_solvers", "projection_solver",
           "phase_field_schur_solver", "phase_field_krylov_solve"]


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


def krylov_solver(name, cfg, krylov_solvers, pc_options=None):
    """Krylov solver for NS_parameters[name] (`cfg`): its solver_type and
    preconditioner_type (PETSc's names, or dolfin's: femtools.ksp), then its
    petsc_options, any further PETSc options, which are set under the prefix
    '<name>_' (e.g. {"sub_pc_type": "icc"} for bjacobi). `pc_options`, when
    given, replace the preconditioner_type's options."""
    options = ksp.solver_options(cfg['solver_type'], cfg['preconditioner_type'],
                                 cfg.get('petsc_options'), pc_options)
    sol = PETScKrylovSolver()
    ksp.configure(sol.ksp(), name + "_", options)
    sol.parameters.update(krylov_solver_parameters(name, cfg, krylov_solvers))
    return sol


def projection_solver(velocity_update_solver, krylov_solvers):
    """A Krylov solver for a velocity-update projection (the velocity_update_solver
    settings), started from zero: it projects the gradient of the pressure
    change, which the last result says nothing about, and as the inner
    iterations converge its right-hand side tends to zero, where a nonzero first
    guess stops PETSc at once (DIVERGED_DTOL)."""
    sol = krylov_solver("velocity_update_solver", velocity_update_solver, krylov_solvers)
    sol.parameters["nonzero_initial_guess"] = False
    return sol


def krylov_get_solvers(krylov_solvers, q_, mesh, bcs, velocity_krylov_solver,
                       velocity_update_solver, pressure_krylov_solver,
                       phase_field_krylov_solver, velocity_tolerance_factor, **NS_namespace):
    """Linear solvers for the BDF and IPCS schemes, in the order tentative
    velocity, pressure, phase field, scalar. Scalars are not supported, so the
    scalar solver is None. The velocity update's solvers belong to the
    projections (GradFunction, made in setup).

    With velocity_tolerance_factor, a tentative-velocity solve stops once its
    residual has fallen by that factor from the one it starts from (a constant
    forcing term of inexact Newton methods, Eisenstat and Walker, SIAM J. Sci.
    Comput. 17 (1996) 16-32). Each pass then changes the velocity by what it
    needs to within that factor, so the passes still converge and their test on
    the change stays valid, while the early passes, which change the velocity
    most, are solved loosely."""
    u_cfg = velocity_krylov_solver
    if velocity_tolerance_factor:
        u_cfg = dict(u_cfg, relative_tolerance=velocity_tolerance_factor, petsc_options=dict(
            {'ksp_converged_use_initial_residual_norm': True}, **u_cfg.get('petsc_options', {})))
    u_sol = krylov_solver("velocity_krylov_solver", u_cfg, krylov_solvers)
    p_cfg = pressure_krylov_solver
    if (mesh.geometry().dim() == 3
            and ksp.PC_NAMES.get(p_cfg['preconditioner_type']) == ksp.BOOMERAMG):
        p_cfg = dict(p_cfg, petsc_options=dict(ksp.BOOMERAMG_3D,
                                               **p_cfg.get('petsc_options', {})))
    p_sol = krylov_solver("pressure_krylov_solver", p_cfg, krylov_solvers)
    if phase_field_krylov_solver['preconditioner_type'] == 'schur':
        pf_sol = phase_field_schur_solver(q_['phig'].function_space(), phase_field_krylov_solver,
                                          krylov_solvers, bcs['phig'])
    else:
        pf_sol = krylov_solver("phase_field_krylov_solver", phase_field_krylov_solver,
                               krylov_solvers)
    return [u_sol, p_sol, pf_sol, None]


class PhaseFieldSchur:
    """Preconditioner for the (phi, g) system [[A, B], [D, Mg]], A = alpha Mphi
    + advection, B = mobility K, D = -sigma_bar epsilon K - the potential, Mg
    the mass matrix of g: the block factorization with the Schur complement
    S = A - B Mg^-1 D replaced by H Mphi^-1 H, H = sqrt(alpha) Mphi +
    sqrt(c) Kphi, c = mobility sigma_bar epsilon. Without advection and the
    potential, and with phi and g in one space, H Mphi^-1 H is spectrally
    equivalent to S with bounds 1/2 and 1, whatever h and dt (Boyanova,
    Do-Quang & Neytcheva, CMAM 12, 2012), so the iteration count does not grow
    as the mesh is refined. At phi's Dirichlet dofs, where the system's rows are
    the identity, H and Mphi are too. phase_field_krylov_solve gives alpha and
    c; H is constant while they are, so it and its solver are only set up again
    when they change, and B and Mg are taken from the system once."""

    def __init__(self, W, prefix, bcs):
        from petsc4py import PETSc

        def dofs(i):
            return np.asarray(W.sub(i).dofmap().dofs(), dtype=PETSc.IntType)

        def block(form):
            return as_backend_type(assemble(form)).mat().createSubMatrix(self.is_phi, self.is_phi)

        phi_dofs = dofs(0)
        self.is_phi = PETSc.IS().createGeneral(phi_dofs, comm=PETSc.COMM_WORLD)
        self.is_g = PETSc.IS().createGeneral(dofs(1), comm=PETSc.COMM_WORLD)
        phi, _ = TrialFunctions(W)
        xi, _ = TestFunctions(W)
        self.Mphi = block(phi * xi * dx)
        self.Kphi = block(inner(grad(phi), grad(xi)) * dx)
        # phi's Dirichlet dofs as rows of the phi block
        start, owned = W.dofmap().ownership_range()[0], W.dofmap().index_map().size(
            IndexMap.MapSize.OWNED)
        fixed = [start + k for bc in bcs for k in bc.get_boundary_values() if k < owned]
        self.fixed = (self.Mphi.getOwnershipRange()[0]
                      + np.flatnonzero(np.isin(phi_dofs, fixed))).astype(PETSc.IntType)
        self.Mphi.zeroRowsColumns(self.fixed, diag=1.)
        self.kH = self.inner_ksp(prefix + "H_")
        self.kM = self.inner_ksp(prefix + "M_")
        self.alpha = self.c = self.H_coefficients = self.B = None

    @staticmethod
    def inner_ksp(prefix):
        from petsc4py import PETSc
        ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
        ksp.setOptionsPrefix(prefix)
        ksp.setFromOptions()
        return ksp

    def setUp(self, pc):
        from petsc4py import PETSc
        A = pc.getOperators()[0]
        if self.B is None:
            self.B = A.createSubMatrix(self.is_phi, self.is_g)
            Mg = A.createSubMatrix(self.is_g, self.is_g)
            self.kM.setOperators(Mg)
            self.D = None
            x = A.createVecLeft()
            self.phi = self.Mphi.createVecs() + (self.Mphi.createVecLeft(),)
            self.g = Mg.createVecs() + (Mg.createVecLeft(),)
            self.to_phi = PETSc.Scatter().create(x, self.is_phi, self.phi[0], None)
            self.to_g = PETSc.Scatter().create(x, self.is_g, self.g[0], None)
        self.D = A.createSubMatrix(self.is_g, self.is_phi, submat=self.D)
        if (self.alpha, self.c) != self.H_coefficients:
            H = self.Mphi.copy()
            H.scale(math.sqrt(self.alpha))
            H.axpy(math.sqrt(self.c), self.Kphi)
            H.zeroRowsColumns(self.fixed, diag=1.)
            self.kH.setOperators(H)
            self.H_coefficients = self.alpha, self.c

    def apply(self, pc, r, z):
        from petsc4py import PETSc
        rphi, s1, s2 = self.phi
        rg, t0, t1 = self.g
        fwd, ins = PETSc.ScatterMode.FORWARD, PETSc.InsertMode.INSERT
        self.to_phi.scatter(r, rphi, ins, fwd)
        self.to_g.scatter(r, rg, ins, fwd)
        self.kM.solve(rg, t0)
        self.B.mult(t0, s1)
        rphi.axpy(-1., s1)                  # r_phi - B Mg^-1 r_g
        self.kH.solve(rphi, s1)             # S^-1 ~ H^-1 Mphi H^-1
        self.Mphi.mult(s1, s2)
        self.kH.solve(s2, s1)               # z_phi
        self.D.mult(s1, t0)
        rg.axpy(-1., t0)                    # r_g - D z_phi
        self.kM.solve(rg, t1)               # z_g
        rev = PETSc.ScatterMode.REVERSE
        self.to_phi.scatter(s1, z, ins, rev)
        self.to_g.scatter(t1, z, ins, rev)


def phase_field_schur_solver(W, phase_field_krylov_solver, krylov_solvers, bcs):
    """Krylov solver for the (phi, g) system with the PhaseFieldSchur
    preconditioner (bcs: the system's Dirichlet conditions). Its inner solvers
    take PETSc options under the prefixes H_ and M_ in petsc_options. H, a mass
    plus a stiffness matrix, is solved with one BoomerAMG cycle in 2D and with
    CG and Jacobi in 3D, where a BoomerAMG cycle costs more than the iterations
    it saves; Mg, a mass matrix, with four Chebyshev-Jacobi iterations. An inner
    Krylov solve is not a fixed linear map, so the outer method must then be a
    flexible one (fgmres or gcr)."""
    if W.mesh().geometry().dim() == 2:
        H = {"H_ksp_type": "preonly", "H_pc_type": "hypre", "H_pc_hypre_type": "boomeramg"}
    else:
        H = {"H_ksp_type": "cg", "H_pc_type": "jacobi", "H_ksp_rtol": 1e-2, "H_ksp_max_it": 50}
    pc_options = dict(H, pc_type="python", M_ksp_type="chebyshev", M_pc_type="jacobi",
                      M_ksp_max_it=4, M_ksp_norm_type="none",
                      M_ksp_chebyshev_esteig="0,0.1,0,1.1")
    name = "phase_field_krylov_solver"
    pf_sol = krylov_solver(name, phase_field_krylov_solver, krylov_solvers, pc_options)
    schur = PhaseFieldSchur(W, name + "_", bcs)
    if schur.kH.getType() != "preonly" and pf_sol.ksp().getType() not in ("fgmres", "gcr"):
        raise ValueError("{}: the 'schur' preconditioner's inner {} solve needs a flexible "
                         "outer method (fgmres or gcr), not {}".format(
                             name, schur.kH.getType(), pf_sol.ksp().getType()))
    pf_sol.ksp().getPC().setPythonContext(schur)
    return pf_sol


def phase_field_krylov_solve(pf_sol, Apf, x, b, alpha, M, sigma_bar, epsilon):
    """Solve the (phi, g) system, giving a PhaseFieldSchur preconditioner the
    time-derivative coefficient alpha and the coefficient of its stiffness
    term, mobility*sigma_bar*epsilon."""
    pc = pf_sol.ksp().getPC()
    if pc.getType() == "python":
        ctx = pc.getPythonContext()
        ctx.alpha, ctx.c = float(alpha), float(M) * float(sigma_bar) * float(epsilon)
    pf_sol.solve(Apf, x, b)


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


def velocity_tentative_solve(ui, A, Ai, bcs, x_, du_, u_sol, b, udiff, unorm,
                             u_components, inner_iter, **NS_namespace):
    """Solve for one component of the tentative velocity."""
    # When all components share the matrix A (Ai is None), it only changes at the
    # start of a timestep, so the preconditioner is rebuilt for the first solve of
    # the timestep and reused after that. With one matrix per component (Ai), the
    # solver switches matrices between solves and must rebuild each time.
    if Ai is None:
        u_sol.set_reuse_preconditioner(not (inner_iter == 1 and ui == u_components[0]))

    [apply_bc(bc, b[ui]) for bc in bcs[ui]]
    du_[ui].zero()
    du_[ui].axpy(1., x_[ui])
    t1 = Timer("Tentative Linear Algebra Solve")
    u_sol.solve(Ai[ui] if Ai else A, x_[ui], b[ui])
    t1.stop()
    du_[ui].axpy(-1., x_[ui])
    udiff[0] += du_[ui].norm('l2')
    unorm[0] += x_[ui].norm('l2')


def velocity_tentative_assemble(**NS_namespace):
    """Assemble remaining system for tentative velocity component."""
    pass

def pressure_assemble(**NS_namespace):
    """Assemble rhs of pressure equation."""
    pass


def pressure_solve(**NS_namespace):
    """Solve pressure equation."""
    pass


def pressure_update(**NS_namespace):
    """The pressure's update after each pressure solve: none by default."""
    pass


class AndersonPressure:
    """Anderson acceleration of the inner passes. Within a time step the momentum
    matrix and the convecting velocity are fixed, so a pass maps the pressure it
    starts from to the one it ends with, p -> G(p), an affine map: the pressure
    solve and its update. After each pass the pressure becomes the combination of
    the last depth + 1 results G(p_i) whose residuals G(p_i) - p_i combine to the
    least norm (Walker and Ni 2011). The coefficients sum to one, so prescribed
    pressures and a normalized mean stay. The first two passes are plain ones."""

    def __init__(self, depth):
        self.depth = depth

    def __call__(self, p, inner_iter):
        g = p.get_local()
        if inner_iter == 1:
            self.x, self.f, self.dF, self.dG = g, None, [], []
            return
        f = g - self.x
        if self.f is not None:
            self.dF = (self.dF + [f - self.f])[-self.depth:]
            self.dG = (self.dG + [g - self.g])[-self.depth:]
        self.f, self.g = f, g
        x = g
        if self.dF:
            F = np.array(self.dF).T
            comm = p.mpi_comm()
            gram = comm.allreduce(F.T @ F)
            gamma = np.linalg.lstsq(gram, comm.allreduce(F.T @ f), rcond=1e-12)[0]
            x = g - np.array(self.dG).T @ gamma
        p.set_local(x)
        p.apply('insert')
        p.update_ghost_values()
        self.x = x


def rotational_fields(rotational, q, Q, V, mu_):
    """What the rotational pressure update (rotational_pressure_update) needs, for
    a solver's setup: div(u*) at the pressure dofs is the integral of div(u*)*q
    divided by the lumped mass on Q, and the viscosity is needed on Q (mu_
    itself when Q is the velocity space)."""
    if not isinstance(rotational, bool):
        raise ValueError("rotational={!r}: True or False".format(rotational))
    if not rotational:
        return dict(mass_Q_inv=None, mu_Q=None, mu_mass_Q=None, rotational_work=None)
    if Q.ufl_element().degree() > 1:
        raise ValueError("rotational=True needs a P1 pressure: the lumped P{} mass has "
                         "zeros".format(Q.ufl_element().degree()))
    mass_Q_inv = assemble(q * dx)
    mass_Q_inv.set_local(1. / mass_Q_inv.get_local())
    mass_Q_inv.apply('insert')
    return dict(mass_Q_inv=mass_Q_inv, mu_Q=mu_ if Q is V else Function(Q),
                mu_mass_Q=Vector(mass_Q_inv), rotational_work=Vector(mass_Q_inv))


def rotational_pressure_update(rotational, x_, divu, mu_, mu_Q, mass_Q_inv, mu_mass_Q,
                               rotational_work, bcs, inner_iter, **NS_namespace):
    """The rotational pressure update (rotational=True), after each pressure
    solve: p = p* + phi - mu*div(u*), with u* the tentative velocity of this
    iteration. The same for the BDF and IPCS solvers, whose momentum steps treat
    the viscous term implicitly. divu.rhs holds the integral of div(u*)*q from
    pressure_assemble; divided by the lumped mass on Q it is div(u*) at the
    pressure dofs. The velocity update uses the increment phi alone, so it is
    unaffected. The pressure's Dirichlet conditions are applied again, so
    prescribed values stay, and without any the pressure is normalized as in
    pressure_solve. mu over the lumped mass, mu_mass_Q, is set on the first
    iteration, where mu_ changes."""
    if not rotational:
        return
    if inner_iter == 1:
        if mu_Q is not mu_:
            interpolate(mu_, mu_Q.function_space(), out=mu_Q)
        mu_mass_Q.zero()
        mu_mass_Q.axpy(1., mu_Q.vector())
        mu_mass_Q *= mass_Q_inv
    w = rotational_work
    w.zero()
    w.axpy(1., divu.rhs)
    w *= mu_mass_Q
    x_['p'].axpy(-1., w)
    [apply_bc(bc, x_['p']) for bc in bcs['p']]
    if bcs['p'] == []:
        normalize(x_['p'])


def velocity_update(**NS_namespace):
    """Update the velocity after finishing pressure velocity iterations."""
    pass

def print_velocity_pressure_info(num_iter, print_velocity_pressure_convergence,
                                 inner_iter, udiff, unorm, dp_, **NS_namespace):
    if num_iter > 1 and print_velocity_pressure_convergence:
        if inner_iter == 1:
            info_blue('  Inner iterations velocity pressure:')
            info_blue('                 error u  rel. u   error p')
        info_blue('    Iter = {0:4d}, {1:2.2e} {2:2.2e} {3:2.2e}'.format(
            inner_iter, udiff[0], udiff[0] / max(unorm[0], 1e-300), norm(dp_.vector())))

def phase_field_assemble(**NS_namespace):
    """Assemble the (phi, g) system: each scheme has its own."""
    pass


def phase_field_solve(pf_sol, Apf, Apf_alpha, M, sigma_bar, epsilon, x_, b, bcs,
                      **NS_namespace):
    """Solve the (phi, g) system for the phase field and chemical potential."""
    t1 = Timer("Phase field solve")
    [apply_bc(bc, Apf, b['phig']) for bc in bcs['phig']]
    phase_field_krylov_solve(pf_sol, Apf, x_['phig'], b['phig'], Apf_alpha[0], M, sigma_bar,
                             epsilon)
    t1.stop()


def potential_linear_derivative(phi, phi0):
    """ Linearised derivative of phase field potential. """
    return phi0**3 - phi0 + (3*phi0**2 - 1.) * (phi - phi0)

def wall_potential_linear_derivative(phi, phi0):
    return 3.*(1. - phi0**2)/4. - 3.*phi0/2.* (phi - phi0)

class PhaseFieldOnVelocitySpace:
    """The phase field phi, taken from the mixed (phi, g) function, as a Function
    c on the velocity space V, by femtools.fast_interpolation: a transfer matrix
    built once, then a sparse matrix-vector product per update."""

    def __init__(self, W, V):
        self.c = Function(V)

    def __call__(self, phig):
        return interpolate(phig.split()[0], self.c.function_space(), out=self.c)


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

def pressure_matrix(constant_poisson_coefficient, rho, rho_inv_, p, q, Q, bcs):
    """The pressure Poisson matrix Ap of the incremental schemes, with the
    coefficient 1/rho: 1/min(rho) when constant_poisson_coefficient, and Ap is
    assembled once here; else rho_inv_, and assemble_pressure_matrix assembles
    the form a_p at the start of every time step. Without a pressure Dirichlet
    condition the constant pressures are its null space, p_null (else None)."""
    coefficient = Constant(1. / min(rho)) if constant_poisson_coefficient else rho_inv_
    a_p = coefficient * inner(grad(q), grad(p)) * dx
    Ap = assemble(a_p)
    [bc.apply(Ap) for bc in bcs['p']]
    p_null = None if bcs['p'] else constant_nullspace(Ap, Q)
    return dict(Ap=Ap, a_p=a_p, p_null=p_null)


def assemble_pressure_matrix(Ap, a_p, p_null, bcs, constant_poisson_coefficient):
    """Assemble Ap again, for a density that changed (see pressure_matrix)."""
    if constant_poisson_coefficient:
        return
    t1 = Timer("Pressure assemble")
    assemble(a_p, tensor=Ap)
    [bc.apply(Ap) for bc in bcs['p']]
    if p_null is not None:
        as_backend_type(Ap).set_nullspace(p_null)
    t1.stop()


def incremental_pressure_assemble(b, x_, Ap, divu, alpha, **NS_namespace):
    """The pressure equation's right-hand side, Ap*p - alpha[0]*(div(u*), q),
    with p the pressure before the solve and alpha[0] the coefficient of the
    new velocity in the time derivative."""
    divu.assemble_rhs()
    b['p'][:] = divu.rhs
    b['p'] *= -alpha[0]
    b['p'].axpy(1., Ap * x_['p'])


def incremental_pressure_solve(dp_, x_, Ap, p_null, b, p_sol, bcs, inner_iter, tstep,
                               constant_poisson_coefficient, **NS_namespace):
    """Solve the pressure equation; dp_ is the change of the pressure."""
    # Rebuilding the AMG preconditioner is expensive, so it is reused while Ap is
    # unchanged: within a time step, or for the whole run when Ap is constant.
    p_sol.set_reuse_preconditioner(inner_iter > 1 or (constant_poisson_coefficient
                                                      and tstep > 1))
    [apply_bc(bc, b['p']) for bc in bcs['p']]
    dp = dp_.vector()
    dp.zero()
    dp.axpy(-1., x_['p'])
    # Without a pressure Dirichlet condition the pressure is defined up to a
    # constant: the right-hand side is made consistent, and the mean set to zero.
    if p_null is not None:
        p_null.orthogonalize(b['p'])
    t1 = Timer("Pressure Linear Algebra Solve")
    p_sol.solve(Ap, x_['p'], b['p'])
    t1.stop()
    if p_null is not None:
        normalize(x_['p'])
    dp.axpy(1., x_['p'])


def incremental_velocity_update(u_components, bcs, gradp, dp_, x_, rho, rho_inv_, alpha,
                                constant_poisson_coefficient, **NS_namespace):
    """The velocity's correction by the last pressure change dp:
    u -= grad(dp) / (alpha[0]*rho), with the pressure equation's 1/rho."""
    for ui in u_components:
        gradp[ui](dp_)
        if constant_poisson_coefficient:
            x_[ui].axpy(-1. / (alpha[0] * min(rho)), gradp[ui].vector())
        else:
            x_[ui].axpy(-1. / alpha[0], gradp[ui].vector() * rho_inv_.vector())
        [apply_bc(bc, x_[ui]) for bc in bcs[ui]]


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
