__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
# Modified for two-phase flow by Gaute Linga, 2022-2026.

from twoasis.backend import (
    apply_bc, as_backend_type, assemble, Constant, constant_nullspace, constant_on, div, dot, dx,
    element_degree, fixed_dofs, Function, function_space, geometric_dimension, grad, hmin,
    inner, interpolate, ksp, lhs, Matrix, MPI, nabla_grad, norm, normalize, owned_dofs,
    PETScKrylovSolver, rhs, space_mesh, TestFunctions, Timer, TrialFunctions, ufl, Vector)
from twoasis.common import dual, telemetry
from twoasis.problems import info_blue
from twoasis.common.utilities import axpy, mass_norm2, mult, nonzero_part, pointwise
from twoasis.solvers import scalar_assemble, scalar_solve, setup
import math
import numpy as np


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
           "phase_field_schur_solver", "phase_field_krylov_solve", "coupled_solve"]


def check_chns_options(chns_solve, initial_guess, dt_split_max):
    """ValueError for a CH-NS step option (problems/TPfracStep/__init__.py) that is
    unknown or does not fit the others."""
    if chns_solve not in ("split", "coupled", "auto"):
        raise ValueError("chns_solve={!r}: 'split', 'coupled' or 'auto'".format(chns_solve))
    if dt_split_max is not None and chns_solve != "auto":
        raise ValueError("dt_split_max is for chns_solve='auto' (here {!r})".format(
            chns_solve))
    if initial_guess not in ("previous", "extrapolated"):
        raise ValueError("initial_guess={!r}: 'previous' or 'extrapolated'".format(
            initial_guess))


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
    preconditioner_type (PETSc's names, or dolfin's: feluga.ksp), then its
    petsc_options, any further PETSc options, which are set under the prefix
    '<name>_' (e.g. {"sub_pc_type": "icc"} for bjacobi). `pc_options`, when
    given, replace the preconditioner_type's options."""
    options = ksp.solver_options(cfg['solver_type'], cfg['preconditioner_type'],
                                 cfg.get('petsc_options'), pc_options)
    sol = PETScKrylovSolver()
    ksp.configure(sol.ksp(), name + "_", options)
    sol.parameters.update(krylov_solver_parameters(name, cfg, krylov_solvers))
    return telemetry.current.solver(sol, name)  # the solver itself unless telemetry is on


def projection_solver(velocity_update_solver, krylov_solvers):
    """A Krylov solver for a velocity-update projection (the velocity_update_solver
    settings), started from zero: it projects the gradient of the pressure
    change, which the last result says nothing about, and as the inner
    iterations converge its right-hand side tends to zero, where a nonzero first
    guess stops PETSc at once (DIVERGED_DTOL)."""
    sol = krylov_solver("velocity_update_solver", velocity_update_solver, krylov_solvers)
    sol.parameters["nonzero_initial_guess"] = False
    return sol


# dt*nu/h^2 above which the tentative velocity takes BoomerAMG: block Jacobi ILU(0) needs
# about 4*sqrt(dt*nu/h^2) iterations once the viscous term dominates, some 30 here, where
# one AMG setup a step and its few cycles cost less
VISCOUS_DOMINATED = 50.


def velocity_preconditioner(cfg, dt, rho, mu, mesh, velocity_degree):
    """The tentative velocity's solver settings `cfg`, preconditioner 'auto' resolved:
    BoomerAMG when dt*nu/h^2 exceeds VISCOUS_DOMINATED (nu the largest of the phases'
    mu/rho, h the smallest cell diameter over the velocity's degree), block Jacobi
    otherwise. Collective (the mesh's smallest cell)."""
    if cfg.get('preconditioner_type') != 'auto':
        return cfg
    nu = max(m / r for m, r in zip(mu, rho))
    h = hmin(mesh) / velocity_degree
    number = dt * nu / h ** 2
    pc = 'hypre_amg' if number > VISCOUS_DOMINATED else 'bjacobi'
    info_blue("Velocity preconditioner: {} (dt*nu/h^2 = {:.3g}, {} {})".format(
        pc, number, "above" if pc == 'hypre_amg' else "at most", VISCOUS_DOMINATED))
    return dict(cfg, preconditioner_type=pc)


# PETSc's convergence test against the initial residual instead of the right-hand side.
# With the previous step's solution as the first guess, the initial residual measures the
# step's change, so a relative tolerance bounds the solve's error relative to that change
# whatever the size of the right-hand side: its pressure terms, large and balanced, make
# ||b|| a poor scale while the flow is slow.
INITIAL_RESIDUAL = {'ksp_converged_use_initial_residual_norm': True}


def with_initial_residual(cfg):
    """A Krylov solver's settings `cfg` with the test against the initial residual
    (INITIAL_RESIDUAL), which cfg's petsc_options may override."""
    return dict(cfg, petsc_options=dict(INITIAL_RESIDUAL, **(cfg.get('petsc_options') or {})))


def for_dimension(cfg, dim):
    """A Krylov solver's settings `cfg` for a mesh of dimension `dim`: in 3D a BoomerAMG
    takes feluga.ksp.BOOMERAMG_3D's settings, which cfg's petsc_options override."""
    if dim == 3 and ksp.PC_NAMES.get(cfg['preconditioner_type']) == ksp.BOOMERAMG:
        cfg = dict(cfg, petsc_options=dict(ksp.BOOMERAMG_3D, **cfg.get('petsc_options', {})))
    return cfg


def krylov_get_solvers(krylov_solvers, q_, mesh, bcs, velocity_krylov_solver,
                       velocity_update_solver, pressure_krylov_solver,
                       phase_field_krylov_solver, velocity_tolerance_factor, dt, rho, mu,
                       velocity_degree, dt_adapt=None, **NS_namespace):
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
    dim = geometric_dimension(mesh)
    # With an adaptive dt bounded by dt_max, the preconditioner for dt_max (else the first dt)
    dt_pc = max(dt, (dt_adapt or {}).get("dt_max") or dt)
    u_cfg = for_dimension(velocity_preconditioner(velocity_krylov_solver, dt_pc, rho, mu, mesh,
                                                  velocity_degree), dim)
    if velocity_tolerance_factor:
        u_cfg = with_initial_residual(dict(u_cfg, relative_tolerance=velocity_tolerance_factor))
    u_sol = krylov_solver("velocity_krylov_solver", u_cfg, krylov_solvers)
    p_cfg = for_dimension(pressure_krylov_solver, dim)
    p_sol = krylov_solver("pressure_krylov_solver", p_cfg, krylov_solvers)
    pf_sol = phase_field_solver(function_space(q_['phig']), phase_field_krylov_solver,
                                krylov_solvers, bcs['phig'])
    return [u_sol, p_sol, pf_sol, None]


def phase_field_solver(W, phase_field_krylov_solver, krylov_solvers, bcs,
                       name="phase_field_krylov_solver"):
    """The (phi, g) system's Krylov solver from its settings (tested against the initial
    residual), with the PhaseFieldSchur preconditioner for preconditioner_type 'schur'.
    `name` is its PETSc options prefix (with "_") and its name in telemetry."""
    pf_cfg = with_initial_residual(phase_field_krylov_solver)
    if pf_cfg['preconditioner_type'] == 'schur':
        return phase_field_schur_solver(W, pf_cfg, krylov_solvers, bcs, name=name)
    return krylov_solver(name, pf_cfg, krylov_solvers)


class PhaseFieldSchur:
    """Preconditioner for the (phi, g) system [[A, B], [D, Mg]], A = alpha Mphi
    + advection, B = mobility K, D = -sigma_bar epsilon K - the potential, Mg
    the mass matrix of g: the block factorization with the Schur complement
    S = A - B Mg^-1 D replaced by a product of two shifted stiffness matrices.

    In the bulk, where the potential's second derivative f'' is F2_BULK, S is
    alpha M + c K M^-1 K + s K, with c = mobility sigma_bar epsilon and
    s = mobility sigma_bar F2_BULK / epsilon. Where alpha <= s^2/(4c) (large steps) it
    factorises exactly as H1 M^-1 H2, H_i = sqrt(c) K + c_i M, c_1 c_2 = alpha,
    c_1 + c_2 = s/sqrt(c): two positive shifts, the roots of a quadratic. Otherwise
    H1 = H2 = sqrt(alpha) M + sqrt(c) K, whose product is within a factor 2 of S
    there (Boyanova, Do-Quang & Neytcheva, CMAM 12, 2012; s = 0 is their setting).
    The potential's bulk value in the preconditioner is as in Brenner, Diegel & Sung
    (J. Sci. Comput. 77, 2018). Both forms match the bulk's modes up to that factor,
    whatever h and dt; the interface, where f'' < F2_BULK, is not in them.

    That is the "bulk" approximation. The "system" one takes the interface too: with
    G = -D (stiffness, the potential's linearisation f''(phi^n) with its sign, the wall
    terms), S = alpha M + B Mg^-1 G, and S ~ (B + beta M) Mg^-1 (G + delta M) = S +
    delta B + beta G, beta delta = alpha, with beta = sqrt(alpha c)/(sigma_bar epsilon)
    and delta = sqrt(alpha c)/mobility, which balance the two error terms' stiffness
    parts. Both factors are symmetric, made from the system's own blocks (any variable
    mobility included), and rebuilt at every setup; it needs phi and g of one degree
    (their dofs node-aligned), and is used whenever they are, else "bulk".

    At phi's Dirichlet dofs, where the system's rows are the identity, H and Mphi
    are too. schur_coefficients sets alpha, c and s (and the mobility and sigma_bar
    epsilon, for "system"); for "bulk" the H are rebuilt (and their AMG set up) only
    when one of these changes by more than REBUILD_RTOL, and solved by one solver when
    they are the same. Mg is taken from the system once, B and D at every setup (B
    changes with a variable mobility)."""

    F2_BULK = 2.0
    REBUILD_RTOL = 1e-2

    def __init__(self, W, prefix, bcs):
        from petsc4py import PETSc

        def block(form):
            return as_backend_type(assemble(form)).mat().createSubMatrix(self.is_phi, self.is_phi)
        self.block = block

        phi_dofs = owned_dofs(W, 0)
        self.is_phi = PETSc.IS().createGeneral(phi_dofs, comm=PETSc.COMM_WORLD)
        self.is_g = PETSc.IS().createGeneral(owned_dofs(W, 1), comm=PETSc.COMM_WORLD)
        approximation = "system"
        if (element_degree(W.sub(0)) != element_degree(W.sub(1))
                or PETSc.COMM_WORLD.tompi4py().allreduce(int(self.is_phi.getLocalSize()
                                                != self.is_g.getLocalSize())) > 0):
            approximation = "bulk"
            info_blue("PhaseFieldSchur: phi and g differ in degree or dof layout; "
                      "approximation 'bulk' instead of 'system'")
        self.approximation = approximation
        phi, _ = TrialFunctions(W)
        xi, _ = TestFunctions(W)
        self.Mphi = block(phi * xi * dx)
        # phi's stiffness matrix, assembled only to build H (when its coefficients change)
        self.Kphi_form = inner(grad(phi), grad(xi)) * dx
        # phi's Dirichlet dofs as rows of the phi block
        self.fixed = (self.Mphi.getOwnershipRange()[0]
                      + np.flatnonzero(np.isin(phi_dofs, fixed_dofs(W, bcs)))).astype(
                          PETSc.IntType)
        self.Mphi.zeroRowsColumns(self.fixed, diag=1.)
        self.kH = self.inner_ksp(prefix + "H_")
        self.kH2 = self.inner_ksp(prefix + "H_")
        self.second = self.kH2      # the solver of H2: kH when H2 is H1
        self.kM = self.inner_ksp(prefix + "M_")
        self.alpha = self.c = self.H_coefficients = self.B = None
        self.s = 0.0
        self.mobility = self.k = None

    def inner(self, k):
        """Count inner solve k's iterations (telemetry's role phase_field_H), and warn
        once if it stopped at its iteration limit, which telemetry's outer counts would
        not show."""
        from petsc4py import PETSc
        telemetry.current.count("phase_field_H", k.getIterationNumber())
        if (k.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT
                and not getattr(self, "warned", False)):
            self.warned = True
            info_blue("PhaseFieldSchur: the inner H solve stopped at its {} iterations; "
                      "its preconditioner (H_ options) needs a stronger one".format(
                          k.getTolerances()[3]))

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
            Mg = A.createSubMatrix(self.is_g, self.is_g)
            self.kM.setOperators(Mg)
            self.D = None
            x = A.createVecLeft()
            self.phi = self.Mphi.createVecs() + (self.Mphi.createVecLeft(),)
            self.g = Mg.createVecs() + (Mg.createVecLeft(),)
            self.to_phi = PETSc.Scatter().create(x, self.is_phi, self.phi[0], None)
            self.to_g = PETSc.Scatter().create(x, self.is_g, self.g[0], None)
        self.B = A.createSubMatrix(self.is_phi, self.is_g, submat=self.B)
        self.D = A.createSubMatrix(self.is_g, self.is_phi, submat=self.D)
        if self.approximation == "system":
            self.system_H()
            return
        new = (self.alpha, self.c, self.s)
        old = self.H_coefficients
        if old is None or not all(math.isclose(a, b, rel_tol=self.REBUILD_RTOL)
                                  for a, b in zip(new, old)):
            Kphi = self.block(self.Kphi_form)
            Hs = []
            for shift in self.shifts():
                H = self.Mphi.copy()
                H.scale(shift)
                H.axpy(math.sqrt(self.c), Kphi)
                H.zeroRowsColumns(self.fixed, diag=1.)
                Hs.append(H)
            Kphi.destroy()
            self.kH.setOperators(Hs[0])
            if len(Hs) > 1:
                self.kH2.setOperators(Hs[1])
            self.second = self.kH2 if len(Hs) > 1 else self.kH
            self.H_coefficients = new

    def system_H(self):
        """H1 = B + beta Mphi and H2 = G + delta Mphi, G = -D, from this setup's B, D."""
        from petsc4py import PETSc
        r = math.sqrt(self.alpha * self.c)
        beta, delta = r / self.k, r / self.mobility
        same = PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN
        H1 = self.B.copy()
        H1.axpy(beta, self.Mphi, same)
        H2 = self.D.copy()
        H2.scale(-1.0)
        H2.axpy(delta, self.Mphi, same)
        for H in (H1, H2):
            H.zeroRowsColumns(self.fixed, diag=1.)
        self.kH.setOperators(H1)
        self.kH2.setOperators(H2)
        self.second = self.kH2

    def shifts(self):
        """The mass shifts of H1 and H2 (one when they are the same)."""
        a, c, s = self.alpha, self.c, self.s
        disc = s * s / (4 * c) - a
        if disc < 0:
            return [math.sqrt(a)]
        c1 = s / (2 * math.sqrt(c)) + math.sqrt(disc)
        return [c1, a / c1]

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
        self.kH.solve(rphi, s1)             # S^-1 ~ H2^-1 Mphi H1^-1
        self.inner(self.kH)
        self.Mphi.mult(s1, s2)
        self.second.solve(s2, s1)           # z_phi
        self.inner(self.second)
        self.D.mult(s1, t0)
        rg.axpy(-1., t0)                    # r_g - D z_phi
        self.kM.solve(rg, t1)               # z_g
        rev = PETSc.ScatterMode.REVERSE
        self.to_phi.scatter(s1, z, ins, rev)
        self.to_g.scatter(t1, z, ins, rev)


# BoomerAMG for the Schur preconditioner's H in 2D: HMIS and ext+i, without the 3D
# settings' aggressive coarsening
H_BOOMERAMG_2D = {"pc_hypre_boomeramg_coarsen_type": "HMIS",
                  "pc_hypre_boomeramg_interp_type": "ext+i",
                  "pc_hypre_boomeramg_strong_threshold": 0.25}


def phase_field_schur_solver(W, phase_field_krylov_solver, krylov_solvers, bcs,
                             name="phase_field_krylov_solver"):
    """Krylov solver for the (phi, g) system with the PhaseFieldSchur preconditioner
    (bcs: the system's Dirichlet conditions). Its inner solvers take PETSc options under
    the prefixes H_ and M_ in petsc_options. H, a mass plus a stiffness matrix, is
    solved by GMRES to a relative 1e-2, preconditioned by one BoomerAMG cycle, whose
    cost does not grow as the mesh is refined. GMRES adds cycles where one falls short:
    at large dt, where H2's shift is small and the potential's negative f'' in the
    interface can make it indefinite, a single cycle lets the outer iterations grow. The
    cycle uses HMIS coarsening and ext+i interpolation, which depend less on the order
    of the dofs than the default (H_BOOMERAMG_2D; feluga.ksp.BOOMERAMG_3D's settings
    in 3D, whose aggressive coarsening makes the outer iterations grow in 2D). Mg, a
    mass matrix, is solved with four Chebyshev-Jacobi iterations. An inner Krylov
    solve is not a fixed linear map, so the outer method must be a flexible one (fgmres,
    the default, or gcr), or H_ksp_type "preonly" for one cycle."""
    H = {"H_ksp_type": "gmres", "H_ksp_rtol": 1e-2, "H_ksp_max_it": 50, "H_pc_type": "hypre",
         "H_pc_hypre_type": "boomeramg"}
    three_d = geometric_dimension(space_mesh(W)) == 3
    H.update({"H_" + k: v for k, v in (ksp.BOOMERAMG_3D if three_d else H_BOOMERAMG_2D).items()})
    pc_options = dict(H, pc_type="python", M_ksp_type="chebyshev", M_pc_type="jacobi",
                      M_ksp_max_it=4, M_ksp_norm_type="none",
                      M_ksp_chebyshev_esteig="0,0.1,0,1.1")
    pf_sol = krylov_solver(name, phase_field_krylov_solver, krylov_solvers, pc_options)
    schur = PhaseFieldSchur(W, name + "_", bcs)
    if schur.kH.getType() != "preonly" and pf_sol.ksp().getType() not in ("fgmres", "gcr"):
        raise ValueError("{}: the 'schur' preconditioner's inner {} solve needs a flexible "
                         "outer method (fgmres or gcr), not {}, or H_ksp_type "
                         "preonly".format(
                             name, schur.kH.getType(), pf_sol.ksp().getType()))
    pf_sol.ksp().getPC().setPythonContext(schur)
    return pf_sol


def schur_coefficients(ctx, alpha, mobility, sigma_bar, epsilon):
    """Give a PhaseFieldSchur `ctx` the time-derivative coefficient alpha and, from
    the mobility, the coefficients c and s of its stiffness and potential terms."""
    ctx.alpha = float(alpha)
    ctx.mobility = float(mobility)
    ctx.k = float(sigma_bar) * float(epsilon)
    ctx.c = float(mobility) * float(sigma_bar) * float(epsilon)
    ctx.s = float(mobility) * float(sigma_bar) * ctx.F2_BULK / float(epsilon)


#: alpha0 at each phase-field KSP's last preconditioner build, by its PETSc handle (a
#: solver's ksp() gives a new petsc4py object each call)
_PF_BUILT = {}


def forget_phase_field_preconditioner(ksp):
    """Have the phase-field KSP's preconditioner rebuilt on its next solve: its operator
    changed in a way alpha0 does not show (chns_solve="auto" switching steps)."""
    _PF_BUILT.pop(ksp.handle, None)


def reuse_phase_field_preconditioner(ksp, reuse, alpha):
    """Keep the phase-field KSP's preconditioner from its last build when `reuse` and
    alpha0 is within 10 % of the alpha0 it was built with; else have it rebuilt (and
    remember alpha0). Returns whether it is kept."""
    built = _PF_BUILT.get(ksp.handle)
    keep = bool(reuse and built is not None and abs(alpha - built) <= 0.1 * abs(built))
    ksp.getPC().setReusePreconditioner(keep)
    if not keep:
        _PF_BUILT[ksp.handle] = float(alpha)
    return keep


def prepare_phase_field_preconditioner(ksp, reuse, alpha, M, sigma_bar, epsilon):
    """Keep the phase-field KSP's preconditioner or have it rebuilt
    (reuse_phase_field_preconditioner), and give a PhaseFieldSchur that is rebuilt its
    coefficients (schur_coefficients)."""
    keep = reuse_phase_field_preconditioner(ksp, reuse, alpha)
    pc = ksp.getPC()
    if pc.getType() == "python" and not keep:
        schur_coefficients(pc.getPythonContext(), alpha, M, sigma_bar, epsilon)


def phase_field_krylov_solve(pf_sol, Apf, x, b, alpha, M, sigma_bar, epsilon, reuse=False):
    """Solve the (phi, g) system (prepare_phase_field_preconditioner first)."""
    prepare_phase_field_preconditioner(pf_sol.ksp(), reuse, alpha, M, sigma_bar, epsilon)
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


def velocity_tentative_solve(ui, A, Ai, Md, bcs, x_, du_, u_sol, b, udiff, unorm,
                             u_components, inner_iter, tstep, velocity_preconditioner_steps,
                             **NS_namespace):
    """Solve for one component of the tentative velocity, adding the squares of the L2
    norms of its change and of its new value to udiff and unorm (the driver takes
    their roots once every component is solved)."""
    # When all components share the matrix A (Ai is None), it only changes at the
    # start of a timestep, so the preconditioner is rebuilt for the first solve of
    # a timestep, every velocity_preconditioner_steps timesteps, and reused
    # otherwise. With one matrix per component (Ai), the solver switches matrices
    # between solves and must rebuild each time.
    if Ai is None:
        u_sol.set_reuse_preconditioner(not (inner_iter == 1 and ui == u_components[0]
                                            and rebuild_step(tstep, velocity_preconditioner_steps)))

    [apply_bc(bc, b[ui]) for bc in bcs[ui]]
    du_[ui].zero()
    du_[ui].axpy(1., x_[ui])
    t1 = Timer("Tentative Linear Algebra Solve")
    u_sol.solve(Ai[ui] if Ai else A, x_[ui], b[ui])
    t1.stop()
    du_[ui].axpy(-1., x_[ui])
    udiff[0] += mass_norm2(Md, du_[ui])
    unorm[0] += mass_norm2(Md, x_[ui])


def rebuild_step(tstep, steps):
    """Whether a preconditioner rebuilt every `steps` timesteps (1: every timestep)
    is rebuilt at timestep tstep, the first one counted 1."""
    return (tstep - 1) % max(int(steps), 1) == 0


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
    if element_degree(Q) > 1:
        raise ValueError("rotational=True needs a P1 pressure: the lumped P{} mass has "
                         "zeros".format(element_degree(Q)))
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
            interpolate(mu_, function_space(mu_Q), out=mu_Q)
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


def coupled_solve(**NS_namespace):
    """Solve a step's velocity and pressure as one system (velocity_pressure_solve=
    "coupled"); a solver that has none raises."""
    raise NotImplementedError("velocity_pressure_solve='coupled': only the BDF solver has "
                              "a coupled velocity-pressure solve")

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
                      tstep=1, phase_field_preconditioner_steps=1, **NS_namespace):
    """Solve the (phi, g) system for the phase field and chemical potential."""
    t1 = Timer("Phase field solve")
    [apply_bc(bc, Apf, b['phig']) for bc in bcs['phig']]
    phase_field_krylov_solve(pf_sol, Apf, x_['phig'], b['phig'], Apf_alpha[0], M, sigma_bar,
                             epsilon,
                             reuse=not rebuild_step(tstep, phase_field_preconditioner_steps))
    t1.stop()


def potential_linear_derivative(phi, phi0):
    """ Linearised derivative of phase field potential. """
    return phi0**3 - phi0 + (3*phi0**2 - 1.) * (phi - phi0)

def wall_potential_linear_derivative(phi, phi0):
    return 3.*(1. - phi0**2)/4. - 3.*phi0/2.* (phi - phi0)

class PhaseFieldOnVelocitySpace:
    """The phase field phi, taken from the mixed (phi, g) function, as a Function
    c on the velocity space V, by feluga.fast_interpolation: a transfer matrix
    built once, then a sparse matrix-vector product per update. On the dual mesh,
    phi at V's nodes, which are the phase field's mesh's too (DualMesh.inject)."""

    def __init__(self, V):
        self.c = Function(V)
        self.dual = dual.current()

    def __call__(self, phig):
        if self.dual is not None:
            return self.dual.inject(phig, 0, self.c)
        return interpolate(phig.split()[0], function_space(self.c), out=self.c)


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
    axpy(b['p'], 1., mult(Ap, x_['p']))


def incremental_pressure_solve(dp_, x_, Ap, p_null, b, p_sol, bcs, inner_iter, tstep,
                               constant_poisson_coefficient, pressure_preconditioner_steps,
                               **NS_namespace):
    """Solve the pressure equation; dp_ is the change of the pressure."""
    # Rebuilding the AMG preconditioner is expensive, so it is reused while Ap is
    # unchanged (within a time step, or for the whole run when Ap is constant), and
    # rebuilt only every pressure_preconditioner_steps timesteps otherwise.
    p_sol.set_reuse_preconditioner(inner_iter > 1 or (constant_poisson_coefficient
                                                      and tstep > 1)
                                   or not rebuild_step(tstep, pressure_preconditioner_steps))
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
            axpy(x_[ui], -1. / alpha[0], pointwise(gradp[ui].vector(), rho_inv_.vector()))
        [apply_bc(bc, x_[ui]) for bc in bcs[ui]]


def convection_form(w, u, v, skew, coefficient=1.0):
    """The convection of the velocity component u by w, tested with v, times
    coefficient: (w.grad u) v, plus 1/2 (div w) u v when skew. w is continuous (it
    lives on the velocity space), so the skew form is antisymmetric up to the boundary
    flux 1/2 int (w.n) u v ds: convection adds no kinetic energy on walls and periodic
    sides even where w, extrapolated from discretely divergence-free fields, is not
    exactly divergence-free. At open ends the flux is the energy convected through
    them, as with the plain form."""
    form = dot(w, nabla_grad(u)) * v
    if skew:
        form = form + 0.5 * div(w) * u * v
    return coefficient * form * dx


def phase_field_system(phi, xi, g, eta, phi_1, u_adv, normal, M, sigma, sigma_bar,
                       epsilon, angles, ds, advection=True):
    """Forms and matrices of the (phi, g) system of the Cahn-Hilliard equation.

    The double-well potential is linearized about phi_1 (tangent line), with an
    error of O(dt^2). The matrix Apf is assembled every timestep from one form:
    advection by u_adv, the linearized potential with its wall terms, mobility
    diffusion, the mass matrix for g, the capillary term, and the time-derivative
    term alpha0 times phi's mass matrix. a_pf_full holds that form, with alpha0 a
    Constant on the phase field's mesh (its second item), so that another alpha0 (a
    variable time step) does not make another form to compile; Apf_alpha holds the
    alpha0 last set. Lpft is the right-hand side of the
    linearized potential. Mpf is phi's mass matrix for products (its nonzero entries
    only: the mixed space's pattern holds four blocks) and its form. Without
    `advection` the form has no advection term (the coupled CH-NS step, monolithic.py,
    has it in a block of its own). a_pf_forms holds both, "split" (with advection) and
    "coupled" (without), with the same alpha0, for steps that switch between the two
    (chns_solve="auto", switch.py).
    """
    fprime = potential_linear_derivative(phi, phi_1)
    fwprime = wall_potential_linear_derivative(phi, phi_1)
    Fpft = add_contact_angles(-sigma_bar / epsilon * fprime * eta * dx,
                              angles, sigma, fwprime, eta, ds)
    apft, Lpft = lhs(Fpft), rhs(Fpft)

    mass = inner(phi, xi) * dx
    Mphi = assemble(mass)
    # Conservative advection of phi: -int phi (u.grad xi) dx + int (u.n) phi xi ds.
    # The boundary term gives the right flux at open boundaries.
    a_pf = -inner(dot(grad(xi), u_adv), phi) * dx + dot(u_adv, normal) * xi * phi * ds
    a_rest = (M * inner(grad(g), grad(xi)) * dx + g * eta * dx
              - sigma_bar * epsilon * inner(grad(phi), grad(eta)) * dx)
    alpha0 = constant_on(0.0, space_mesh(
        ufl.algorithms.extract_arguments(phi)[0].ufl_function_space()))
    forms = {"split": (a_pf + apft + a_rest + alpha0 * mass, alpha0),
             "coupled": (apft + a_rest + alpha0 * mass, alpha0)}
    return dict(Apf=Mphi, Mpf=(nonzero_part(Mphi), mass), Lpft=Lpft,
                a_pf_full=forms["split" if advection else "coupled"], a_pf_forms=forms,
                Apf_alpha=np.full(1, np.nan))


def assemble_phase_field_system(alpha0, Apf, a_pf_full, Apf_alpha, Mpf,
                                Lpft, b, bpf, fpf, t):
    """Assemble the phase-field matrix Apf, with the time-derivative term
    alpha0*Mpf[0], and the parts of the right-hand side b that do not involve
    earlier timesteps: the linearized potential and the source term bpf (if any).
    The caller adds the time-derivative terms of the earlier timesteps to b."""
    if alpha0 != Apf_alpha[0]:
        a_pf_full[1].assign(float(alpha0))
        Apf_alpha[0] = alpha0
    assemble(a_pf_full[0], tensor=Apf)

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
