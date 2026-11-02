__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
from twoasis.solvers import *
import math
import ufl_legacy as ufl
if not isinstance(Constant, ufl.Coefficient):
    import ufl


"""Define all functions required by fractional step solver."""
__all__ = ["assemble_first_inner_iter", "velocity_tentative_assemble",
           "velocity_tentative_solve", "pressure_assemble",
           "pressure_solve", "velocity_update", "scalar_assemble",
           "scalar_solve", "get_solvers", "setup",
           "print_velocity_pressure_info", "phase_field_solve", "phase_field_assemble",
           "potential_linear_derivative", "potential_derivative_approx",
           "wall_potential_linear_derivative",
           "weighted_arithmetic_mean", "weighted_geometric_mean", 
           "weighted_arithmetic_derivative",
           "krylov_solver_parameters", "krylov_solver", "phase_field_fieldsplit_solver"]


def krylov_solver_parameters(name, cfg, krylov_solvers):
    """Parameters for the Krylov solver configured by the per-solver dict `cfg`
    (NS_parameters[name], e.g. name='pressure_krylov_solver').

    The global `krylov_solvers` dict applies to every solver. A per-solver dict may
    additionally carry 'relative_tolerance' and 'maximum_iterations'. For those keys
    the precedence is:
        per-solver value set by the user > global value set by the user
        > per-solver default > global default,
    where "set by the user" means it differs from the package default in
    problems.TPfracStep.KRYLOV_DEFAULTS. So krylov_solvers='{"relative_tolerance": 1e-10}'
    tightens every solver, while pressure_krylov_solver='{"relative_tolerance": 1e-10}'
    tightens only the pressure solve.
    """
    from twoasis.problems.TPfracStep import KRYLOV_DEFAULTS
    params = dict(krylov_solvers)
    for key in ("relative_tolerance", "maximum_iterations"):
        if key not in cfg:
            continue
        cfg_user_set = cfg[key] != KRYLOV_DEFAULTS.get(name, {}).get(key)
        global_user_set = (key in krylov_solvers and
                           krylov_solvers[key] != KRYLOV_DEFAULTS["krylov_solvers"].get(key))
        if cfg_user_set or not global_user_set:
            params[key] = cfg[key]
    return params


def krylov_solver(name, cfg, krylov_solvers):
    """PETScKrylovSolver with solver/preconditioner type from `cfg` and parameters
    from krylov_solver_parameters."""
    sol = PETScKrylovSolver(cfg['solver_type'], PETScPreconditioner(cfg['preconditioner_type']))
    sol.parameters.update(krylov_solver_parameters(name, cfg, krylov_solvers))
    return sol


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
    # Schur on (phi, g): g-block is a mass matrix; the Schur complement onto phi is
    # the stiff biharmonic-like operator -> AMG. (Multiplicative splits do poorly:
    # both diagonal blocks are well-conditioned mass matrices.)
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


def velocity_tentative_solve(**NS_namespace):
    """Linear algebra solve of tentative velocity component."""
    pass


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

def print_velocity_pressure_info(num_iter, print_velocity_pressure_convergence, norm,
                                             info_blue, inner_iter, udiff, dp_, **NS_namespace):
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

def potential_derivative_approx(phi_, phi_1, phi_2):
    phi_ab = 1.5 * phi_1 - 0.5 * phi_2
    phi_cn = 0.5 * (phi_ + phi_1)
    return 0.5 * (phi_**2 + phi_1**2) * phi_cn - phi_ab


def weighted_arithmetic_mean(a, phi):
    phi_bar = ufl.min_value(ufl.max_value(phi, -1.0), 1.0)
    #phi_mod = ufl.sin(math.pi/2 * phi_bar)
    #return 0.5 * (a[0] + a[1] + (a[0] - a[1]) * phi_mod)
    return 0.5 * (a[0] + a[1] + (a[0] - a[1]) * phi_bar)

def weighted_geometric_mean(a, phi):
    phi_bar = ufl.min_value(ufl.max_value(phi, -1.0), 1.0)
    chi = 0.5*(1 + ufl.sin(math.pi/2 * phi_bar))
    return a[0]**chi * a[1]**(1-chi)

def weighted_harmonic_mean(a, phi):
    return 1./weighted_arithmetic_mean([1./a[0], 1./a[1]], phi)

def weighted_arithmetic_derivative(a, phi):
    phi_bar = ufl.min_value(ufl.max_value(phi, -1.0), 1.0)
    dphi_mod = math.pi/2 * ufl.cos(math.pi/2 * phi_bar)
    return 0.5*(a[0]-a[1]) * dphi_mod